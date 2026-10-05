import gc
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# Import Decord only after PyTorch; the reverse order can crash CUDA lazy init.
import decord
import transformers
from decord import VideoReader, cpu

# Compatibility for Qwen2.5-VL with the project's torch version.
if not hasattr(torch, "compiler"):
    class _CompilerCompat:
        @staticmethod
        def disable(*args, **kwargs):
            def decorator(fn):
                return fn
            return decorator

    torch.compiler = _CompilerCompat()

for _name in ("float8_e4m3fn", "float8_e5m2", "float8_e4m3fnuz", "float8_e5m2fnuz"):
    if not hasattr(torch, _name):
        setattr(torch, _name, torch.float16)

from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


VIDEO_EXTS = (".mp4", ".avi", ".mkv", ".webm", ".mov", ".m4v")
PROMPT = (
    "This is a single frame from an untrimmed video in THUMOS14."
    "Identify the most relevant THUMOS14 action category suggested by the visual content of this frame. "
    "Focus on human pose, objects, and scene context. "
    "Do NOT invent new categories.\n\n"
    "THUMOS14 action categories:\n"
    "BaseballPitch, BasketballDunk, Billiards, CleanAndJerk, CliffDiving, "
    "CricketBowling, CricketShot, Diving, FrisbeeCatch, GolfSwing, "
    "HammerThrow, HighJump, JavelinThrow, LongJump, PoleVault, Shotput, "
    "SoccerPenalty, TennisSwing, ThrowDiscus, VolleyballSpiking.\n\n"
    "If NO specific action from the list can be confidently inferred from this single frame, output null."
    "Output:\n"
    "<category or null>\n"
    "Do not include any actions outside the THUMOS14 category list. "
    "Do not add explanations beyond the required output."
)


def parse_hidden_layers(value):
    if value.strip().lower() == "all":
        return None
    layers = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not layers:
        raise ValueError("QWEN_HIDDEN_LAYERS must contain at least one layer index.")
    if any(layer <= 0 for layer in layers):
        raise ValueError(f"QWEN_HIDDEN_LAYERS must be positive, got {layers}.")
    if len(set(layers)) != len(layers):
        raise ValueError(f"QWEN_HIDDEN_LAYERS contains duplicate indices: {layers}.")
    return tuple(sorted(layers))


# hidden_states[0] is the embedding output. Positive indices 1..N represent
# decoder-block outputs; index N includes the final norm and matches the last
# hidden state used to generate experiment 08's frames_pooled features.
REQUESTED_HIDDEN_LAYERS = parse_hidden_layers(
    os.getenv("QWEN_HIDDEN_LAYERS", "all")
)


def list_videos(video_root):
    return sorted(
        path for path in video_root.rglob("*") if path.is_file() and path.suffix.lower() in VIDEO_EXTS
    )


@torch.no_grad()
def infer_token_features(
    model,
    processor,
    pil_images,
    max_pixels,
    hidden_layers,
):
    messages = [
        [{
            "role": "user",
            "content": [
                {"type": "image", "image": image, "max_pixels": max_pixels},
                {"type": "text", "text": PROMPT},
            ],
        }]
        for image in pil_images
    ]
    texts = [processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True) for message in messages]
    image_inputs, video_inputs = process_vision_info(messages)
    device = next(model.parameters()).device
    inputs = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(device)

    output = model(**inputs, output_hidden_states=True, return_dict=True)
    hidden_states = output.hidden_states
    last_layer_index = len(hidden_states) - 1
    configured_layers = int(model.config.num_hidden_layers)
    if last_layer_index != configured_layers:
        raise RuntimeError(
            "Qwen hidden-state count does not match its config: "
            f"config={configured_layers}, hidden_states={last_layer_index}."
        )
    invalid_layers = [layer for layer in hidden_layers if layer > last_layer_index]
    if invalid_layers:
        raise RuntimeError(
            f"Requested hidden layers {invalid_layers}, but the model has "
            f"{last_layer_index} decoder layers."
        )
    # Preserve experiment 08 exactly: average all valid image and text tokens.
    token_mask = inputs.attention_mask.unsqueeze(-1)

    pooled_by_layer = {}
    for layer_index in hidden_layers:
        hidden = hidden_states[layer_index]
        mask = token_mask.to(dtype=hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        pooled_by_layer[layer_index] = pooled.to(device="cpu", dtype=torch.float16)
    return pooled_by_layer


def validate_last_layer_against_reference(
    pooled,
    frame_idx,
    reference_path,
    max_abs_tolerance,
):
    if not reference_path.exists():
        raise FileNotFoundError(f"Missing experiment-08 reference: {reference_path}")
    reference = torch.load(reference_path, map_location="cpu")
    reference_idx = reference["frame_idx"]
    reference_pooled = reference["pooled"]
    if not torch.equal(frame_idx, reference_idx):
        raise RuntimeError(f"Frame indices differ from experiment 08: {reference_path}")
    if pooled.shape != reference_pooled.shape:
        raise RuntimeError(
            "Last-layer shape differs from experiment 08: "
            f"new={tuple(pooled.shape)}, old={tuple(reference_pooled.shape)}"
        )
    difference = (pooled.float() - reference_pooled.float()).abs()
    max_abs = float(difference.max())
    mean_abs = float(difference.mean())
    cosine = float(torch.nn.functional.cosine_similarity(
        pooled.float().flatten(), reference_pooled.float().flatten(), dim=0
    ))
    print(
        f"[REFERENCE] {reference_path.parent.name} max_abs={max_abs:.8g} "
        f"mean_abs={mean_abs:.8g} cosine={cosine:.10f}"
    )
    if max_abs > max_abs_tolerance:
        raise RuntimeError(
            "Last-layer feature does not reproduce experiment 08: "
            f"max_abs={max_abs:.8g} > tolerance={max_abs_tolerance:.8g}"
        )


def main():
    video_root = Path(os.getenv(
        "THUMOS_VIDEO_ROOT",
        "/mnt/nvme1n1/zhouzhewen/OpenTAD-main/data/thumos-14/raw_data/video",
    ))
    save_root = Path(os.getenv(
        "QWEN_FEATURE_ROOT",
        "/mnt/nvme1n1/zhouzhewen/actionteller/qwen/data/thumos_hidden/"
        "frames_baseline08_all_layers_1_28",
    ))
    model_dir = os.getenv(
        "QWEN_MODEL_DIR",
        "/mnt/nvme1n1/zhouzhewen/actionteller/qwen/Qwen2.5-VL-7B-Instruct",
    )
    stride = int(os.getenv("QWEN_FRAME_STRIDE", "4"))
    qwen_batch = int(os.getenv("QWEN_BATCH_SIZE", "10"))
    decode_chunk = int(os.getenv("QWEN_DECODE_CHUNK", "128"))
    max_pixels = int(os.getenv("QWEN_MAX_PIXELS", str(160 * 160)))
    video_limit = int(os.getenv("QWEN_VIDEO_LIMIT", "0"))
    num_shards = int(os.getenv("QWEN_NUM_SHARDS", "1"))
    shard_index = int(os.getenv("QWEN_SHARD_INDEX", "0"))
    reference_root_value = os.getenv("QWEN_REFERENCE_ROOT", "").strip()
    reference_root = Path(reference_root_value) if reference_root_value else None
    reference_max_abs_tolerance = float(os.getenv("QWEN_REFERENCE_MAX_ABS_TOL", "0"))

    if num_shards <= 0:
        raise ValueError(f"QWEN_NUM_SHARDS must be positive, got {num_shards}.")
    if not 0 <= shard_index < num_shards:
        raise ValueError(
            f"QWEN_SHARD_INDEX must be in [0, {num_shards}), got {shard_index}."
        )

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_dir)
    model.eval()

    num_hidden_layers = int(model.config.num_hidden_layers)
    hidden_layers = REQUESTED_HIDDEN_LAYERS or tuple(range(1, num_hidden_layers + 1))
    invalid_layers = [
        layer for layer in hidden_layers if layer < 1 or layer > num_hidden_layers
    ]
    if invalid_layers:
        raise ValueError(
            f"Requested invalid hidden layers {invalid_layers}; "
            f"valid decoder-block outputs are 1..{num_hidden_layers}."
        )
    if reference_root is not None and num_hidden_layers not in hidden_layers:
        raise ValueError("QWEN_REFERENCE_ROOT requires the final hidden layer.")

    save_root.mkdir(parents=True, exist_ok=True)
    layer_roots = {
        layer_index: save_root / f"layer_{layer_index}"
        for layer_index in hidden_layers
    }
    for layer_root in layer_roots.values():
        layer_root.mkdir(parents=True, exist_ok=True)
    decord.bridge.set_bridge("native")

    all_videos = list_videos(video_root)
    if video_limit > 0:
        all_videos = all_videos[:video_limit]
    videos = all_videos[shard_index::num_shards]
    print(
        f"[INFO] found {len(all_videos)} videos; assigned={len(videos)}; "
        f"shard={shard_index}/{num_shards}; layers={hidden_layers}; output={save_root}"
    )
    print(
        "[PROTOCOL] experiment-08 prompt; Decord RGB frames; "
        "attention-masked mean over all valid image+text tokens; "
        f"stride={stride}; batch={qwen_batch}; decode_chunk={decode_chunk}; "
        f"max_pixels={max_pixels}; torch={torch.__version__}; "
        f"transformers={transformers.__version__}; decord={decord.__version__}"
    )

    errors = []
    for number, video in enumerate(videos, 1):
        packed_paths = {
            layer_index: layer_roots[layer_index] / video.stem / "packed.pt"
            for layer_index in hidden_layers
        }
        if all(path.exists() for path in packed_paths.values()):
            print(f"[SKIP] ({number}/{len(videos)}) {video.stem}")
            continue

        try:
            reader = VideoReader(str(video), ctx=cpu(0))
            frame_count = len(reader)
            if frame_count <= 0:
                raise RuntimeError(f"Decord reports no frames: {video}")

            features = {layer_index: [] for layer_index in hidden_layers}
            indices = []
            sampled_indices = np.arange(0, frame_count, stride, dtype=np.int64)
            for start in range(0, len(sampled_indices), decode_chunk):
                index_chunk = sampled_indices[start:start + decode_chunk]
                frames = reader.get_batch(index_chunk).asnumpy()
                for batch_start in range(0, len(index_chunk), qwen_batch):
                    batch_end = min(batch_start + qwen_batch, len(index_chunk))
                    images = [
                        Image.fromarray(frames[position])
                        for position in range(batch_start, batch_end)
                    ]
                    batch_features = infer_token_features(
                        model, processor, images, max_pixels, hidden_layers
                    )
                    for layer_index in hidden_layers:
                        features[layer_index].append(batch_features[layer_index])
                    indices.append(torch.from_numpy(
                        index_chunk[batch_start:batch_end].copy()
                    ).to(torch.int32))

            if not features[hidden_layers[-1]]:
                raise RuntimeError(f"No sampled frames: {video}")

            pooled_by_layer = {
                layer_index: torch.cat(features[layer_index], dim=0)
                for layer_index in hidden_layers
            }
            frame_idx = torch.cat(indices, dim=0)
            if reference_root is not None:
                validate_last_layer_against_reference(
                    pooled_by_layer[num_hidden_layers],
                    frame_idx,
                    reference_root / video.stem / "packed.pt",
                    reference_max_abs_tolerance,
                )
            for layer_index, pooled in pooled_by_layer.items():
                packed_path = packed_paths[layer_index]
                if packed_path.exists():
                    continue
                packed_path.parent.mkdir(parents=True, exist_ok=True)
                pack = {
                    "video_name": video.stem,
                    "video_path": str(video),
                    "stride": stride,
                    "frame_idx": frame_idx,
                    # The baseline SSA reader consumes this key, so every layer
                    # directory can be selected without changing detector code.
                    "pooled": pooled,
                    "hidden_layer": layer_index,
                    "num_hidden_layers": num_hidden_layers,
                    "hidden_layer_ratio": layer_index / num_hidden_layers,
                    "hidden_dim": pooled.shape[-1],
                    "max_pixels": max_pixels,
                    "feature_type": (
                        f"qwen2_5_vl_hidden_layer_{layer_index}_all_valid_token_mean"
                    ),
                    "pooling_type": "attention_masked_mean_all_valid_tokens",
                    "frame_decoder": f"decord_{decord.__version__}_rgb",
                    "extractor_protocol": "actionteller_experiment_08",
                    "torch_version": torch.__version__,
                    "transformers_version": transformers.__version__,
                    "prompt": PROMPT,
                    "model_dir": model_dir,
                }
                temporary_path = packed_path.with_name(
                    f"{packed_path.name}.tmp.{os.getpid()}"
                )
                torch.save(pack, temporary_path)
                os.replace(temporary_path, packed_path)

            layer_shapes = {
                layer_index: tuple(pooled_by_layer[layer_index].shape)
                for layer_index in hidden_layers
            }
            print(f"[OK] ({number}/{len(videos)}) {video.stem} {layer_shapes}")
        except Exception as error:
            print(f"[ERR] ({number}/{len(videos)}) {video.name}: {error!r}")
            errors.append((video.name, repr(error)))
            torch.cuda.empty_cache()
            gc.collect()

    if errors:
        preview = ", ".join(f"{name}: {error}" for name, error in errors[:5])
        raise RuntimeError(
            f"Feature extraction failed for {len(errors)} videos on shard "
            f"{shard_index}/{num_shards}. First errors: {preview}"
        )


def parse_integer_list(value):
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def set_probe_seed(seed):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def load_probe_annotations(annotation_path, class_map_path):
    import json

    database = json.loads(Path(annotation_path).read_text())["database"]
    class_names = [
        line.strip() for line in Path(class_map_path).read_text().splitlines()
        if line.strip()
    ]
    if len(class_names) != 20 or len(set(class_names)) != len(class_names):
        raise RuntimeError(f"Expected 20 unique THUMOS14 classes, got {class_names}")
    return database, class_names, {name: index for index, name in enumerate(class_names)}


def limit_probe_database(database, videos_per_subset):
    if videos_per_subset <= 0:
        return database
    limited = {}
    for subset in ("training", "validation"):
        names = [
            name for name, record in sorted(database.items())
            if record.get("subset") == subset
        ][:videos_per_subset]
        limited.update({name: database[name] for name in names})
    return limited


def resolve_probe_layers(feature_root, value):
    if value.strip().lower() == "all":
        layers = tuple(
            sorted(
                int(path.name.split("_", 1)[1])
                for path in feature_root.glob("layer_*")
                if path.is_dir() and path.name.split("_", 1)[1].isdigit()
            )
        )
    else:
        layers = parse_integer_list(value)
    if not layers:
        raise RuntimeError(f"No probe layers found under {feature_root}")
    return layers


def probe_feature_path(feature_root, layer_index, video_name):
    layered_path = feature_root / f"layer_{layer_index}" / video_name / "packed.pt"
    if layered_path.exists():
        return layered_path
    direct_path = feature_root / video_name / "packed.pt"
    if layer_index == 28 and direct_path.exists():
        return direct_path
    raise FileNotFoundError(
        f"Missing layer {layer_index} feature for {video_name} under {feature_root}"
    )


def build_classification_probe_dataset(
    feature_root,
    layer_index,
    database,
    subset,
    label_to_index,
):
    feature_rows = []
    labels = []
    video_names = []
    fallback_segments = 0
    for video_name, record in sorted(database.items()):
        if record.get("subset") != subset:
            continue
        packed_path = probe_feature_path(feature_root, layer_index, video_name)
        pack = torch.load(packed_path, map_location="cpu")
        pooled = pack["pooled"].float()
        frame_idx = pack["frame_idx"].float()
        duration = float(record["duration"])
        frame_count = float(record["frame"])
        if duration <= 0 or frame_count <= 0:
            raise RuntimeError(f"Invalid video metadata for {video_name}: {record}")
        fps = frame_count / duration
        frame_times = frame_idx / fps
        for annotation in record.get("annotations", []):
            label = annotation["label"]
            if label not in label_to_index:
                continue
            start, end = map(float, annotation["segment"])
            selected = (frame_times >= start) & (frame_times <= end)
            if selected.any():
                segment_feature = pooled[selected].mean(dim=0)
            else:
                fallback_segments += 1
                center = (start + end) / 2.0
                nearest = int(torch.argmin((frame_times - center).abs()))
                segment_feature = pooled[nearest]
            feature_rows.append(segment_feature)
            labels.append(label_to_index[label])
            video_names.append(video_name)
    if not feature_rows:
        raise RuntimeError(f"No {subset} classification samples for layer {layer_index}")
    return (
        torch.stack(feature_rows),
        torch.tensor(labels, dtype=torch.long),
        np.asarray(video_names),
        fallback_segments,
    )


def binary_average_precision(scores, targets):
    scores = np.asarray(scores, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int64)
    positive_count = int(targets.sum())
    if positive_count == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    sorted_targets = targets[order]
    true_positives = np.cumsum(sorted_targets)
    precision = true_positives / np.arange(1, len(sorted_targets) + 1)
    return float((precision * sorted_targets).sum() / positive_count)


def classification_probe_metrics(probabilities, targets, num_classes):
    targets = np.asarray(targets, dtype=np.int64)
    predictions = np.asarray(probabilities).argmax(axis=1)
    top1 = float((predictions == targets).mean())
    recalls = []
    average_precisions = []
    for class_index in range(num_classes):
        class_targets = targets == class_index
        recalls.append(float((predictions[class_targets] == class_index).mean()))
        average_precisions.append(binary_average_precision(
            probabilities[:, class_index], class_targets
        ))
    return {
        "top1_accuracy": top1,
        "balanced_accuracy": float(np.mean(recalls)),
        "class_mAP": float(np.nanmean(average_precisions)),
        "per_class_recall": recalls,
        "per_class_AP": average_precisions,
    }


def train_classification_linear_probe(
    train_features,
    train_labels,
    test_features,
    test_labels,
    seed,
    device,
    epochs,
    batch_size,
    learning_rate,
    weight_decay,
    num_classes,
):
    set_probe_seed(seed)
    model = torch.nn.Linear(train_features.shape[1], num_classes).to(device)
    counts = torch.bincount(train_labels, minlength=num_classes).float()
    class_weights = counts.sum() / (num_classes * counts.clamp_min(1.0))
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    final_loss = None
    model.train()
    for _ in range(epochs):
        permutation = torch.randperm(len(train_labels), generator=generator)
        loss_sum = 0.0
        sample_count = 0
        for start in range(0, len(permutation), batch_size):
            indices = permutation[start:start + batch_size]
            batch_features = train_features[indices].to(device, non_blocking=True)
            batch_labels = train_labels[indices].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_features)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(indices)
            sample_count += len(indices)
        final_loss = loss_sum / sample_count
    model.eval()
    probability_batches = []
    with torch.no_grad():
        for start in range(0, len(test_labels), batch_size):
            batch_features = test_features[start:start + batch_size].to(
                device, non_blocking=True
            )
            probability_batches.append(
                torch.softmax(model(batch_features), dim=1).cpu()
            )
    probabilities = torch.cat(probability_batches).numpy()
    metrics = classification_probe_metrics(
        probabilities, test_labels.numpy(), num_classes
    )
    metrics["final_train_loss"] = final_loss
    return metrics, probabilities


def classification_probe_main():
    import json

    annotation_path = os.getenv(
        "QWEN_PROBE_ANNOTATION",
        "/mnt/nvme1n1/zhouzhewen/OpenTAD-main/data/thumos-14/annotations/thumos_14_anno.json",
    )
    class_map_path = os.getenv(
        "QWEN_PROBE_CLASS_MAP",
        "/mnt/nvme1n1/zhouzhewen/OpenTAD-main/data/thumos-14/annotations/category_idx.txt",
    )
    feature_root = Path(os.getenv(
        "QWEN_PROBE_FEATURE_ROOT",
        "/mnt/nvme1n1/zhouzhewen/actionteller/qwen/data/thumos_hidden/frames_baseline08_all_layers_1_28",
    ))
    output_root = Path(os.getenv(
        "QWEN_PROBE_OUTPUT_ROOT",
        "exps/thumos/open-git/20_qwen_layerwise_classification_probe",
    ))
    layers = resolve_probe_layers(
        feature_root, os.getenv("QWEN_PROBE_LAYERS", "all")
    )
    seeds = parse_integer_list(os.getenv("QWEN_PROBE_SEEDS", "42,2026,3407"))
    epochs = int(os.getenv("QWEN_PROBE_EPOCHS", "100"))
    batch_size = int(os.getenv("QWEN_PROBE_BATCH_SIZE", "256"))
    learning_rate = float(os.getenv("QWEN_PROBE_LR", "0.01"))
    weight_decay = float(os.getenv("QWEN_PROBE_WEIGHT_DECAY", "0.0001"))
    overwrite = os.getenv("QWEN_PROBE_OVERWRITE", "0") == "1"
    requested_device = os.getenv("QWEN_PROBE_DEVICE", "cuda")
    device = torch.device(
        requested_device if requested_device != "cuda" or torch.cuda.is_available()
        else "cpu"
    )
    output_root.mkdir(parents=True, exist_ok=True)
    database, class_names, label_to_index = load_probe_annotations(
        annotation_path, class_map_path
    )
    database = limit_probe_database(
        database, int(os.getenv("QWEN_PROBE_VIDEO_LIMIT_PER_SPLIT", "0"))
    )
    print(
        f"[CLASS-PROBE] layers={layers}; seeds={seeds}; device={device}; "
        f"epochs={epochs}; batch={batch_size}; lr={learning_rate}; wd={weight_decay}"
    )
    for layer_index in layers:
        result_path = output_root / f"layer_{layer_index:02d}.json"
        if result_path.exists() and not overwrite:
            print(f"[CLASS-SKIP] layer={layer_index} result exists")
            continue
        train_x, train_y, train_videos, train_fallback = build_classification_probe_dataset(
            feature_root, layer_index, database, "training", label_to_index
        )
        test_x, test_y, test_videos, test_fallback = build_classification_probe_dataset(
            feature_root, layer_index, database, "validation", label_to_index
        )
        train_mean = train_x.mean(dim=0)
        train_std = train_x.std(dim=0, unbiased=False).clamp_min(1e-6)
        train_x = torch.nan_to_num((train_x - train_mean) / train_std)
        test_x = torch.nan_to_num((test_x - train_mean) / train_std)
        runs = []
        for seed in seeds:
            metrics, probabilities = train_classification_linear_probe(
                train_x,
                train_y,
                test_x,
                test_y,
                seed,
                device,
                epochs,
                batch_size,
                learning_rate,
                weight_decay,
                len(class_names),
            )
            prediction_path = output_root / (
                f"layer_{layer_index:02d}_seed_{seed}_predictions.npz"
            )
            np.savez_compressed(
                prediction_path,
                probabilities=probabilities,
                targets=test_y.numpy(),
                video_names=test_videos,
                class_names=np.asarray(class_names),
            )
            runs.append({"seed": seed, **metrics})
            print(
                f"[CLASS-RESULT] layer={layer_index} seed={seed} "
                f"top1={metrics['top1_accuracy']:.6f} "
                f"balanced={metrics['balanced_accuracy']:.6f} "
                f"mAP={metrics['class_mAP']:.6f}"
            )
        summary = {
            key: {
                "mean": float(np.mean([run[key] for run in runs])),
                "std": float(np.std([run[key] for run in runs], ddof=1)),
            }
            for key in ("top1_accuracy", "balanced_accuracy", "class_mAP")
        }
        result = {
            "probe": "classification_linear_gt_segment_mean",
            "layer": layer_index,
            "feature_root": str(feature_root),
            "annotation_path": annotation_path,
            "class_map_path": class_map_path,
            "train_subset": "training",
            "test_subset": "validation",
            "train_samples": int(len(train_y)),
            "test_samples": int(len(test_y)),
            "train_videos": int(len(set(train_videos.tolist()))),
            "test_videos": int(len(set(test_videos.tolist()))),
            "fallback_segments": {
                "train": train_fallback,
                "test": test_fallback,
            },
            "normalization": "per_dimension_zscore_from_training_split",
            "optimizer": "AdamW",
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "seeds": list(seeds),
            "class_names": class_names,
            "runs": runs,
            "summary": summary,
        }
        temporary_path = result_path.with_suffix(".json.tmp")
        temporary_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
        os.replace(temporary_path, result_path)
        print(f"[CLASS-DONE] layer={layer_index} summary={summary}")


def normalize_boundary_features(features):
    features = features.float()
    return torch.nn.functional.layer_norm(features, (features.shape[-1],))


def boundary_targets(frame_times, record, positive_tolerance):
    start_targets = torch.zeros(len(frame_times), dtype=torch.bool)
    end_targets = torch.zeros(len(frame_times), dtype=torch.bool)
    start_times = []
    end_times = []
    for annotation in record.get("annotations", []):
        start_time, end_time = map(float, annotation["segment"])
        start_times.append(start_time)
        end_times.append(end_time)
        for target, boundary_time in (
            (start_targets, start_time),
            (end_targets, end_time),
        ):
            if positive_tolerance > 0:
                selected = (frame_times - boundary_time).abs() <= positive_tolerance
            else:
                selected = torch.zeros_like(target)
            if not selected.any():
                selected[int(torch.argmin((frame_times - boundary_time).abs()))] = True
            target |= selected
    return start_targets, end_targets, start_times, end_times


def build_boundary_probe_dataset(
    feature_root,
    layer_index,
    database,
    subset,
    positive_tolerance,
    negative_exclusion,
    negative_ratio,
    dilation,
    data_seed,
):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(data_seed)
    windows = []
    targets = []
    positive_counts = torch.zeros(2, dtype=torch.long)
    for video_name, record in sorted(database.items()):
        if record.get("subset") != subset:
            continue
        pack = torch.load(
            probe_feature_path(feature_root, layer_index, video_name),
            map_location="cpu",
        )
        features = normalize_boundary_features(pack["pooled"])
        frame_idx = pack["frame_idx"].float()
        fps = float(record["frame"]) / float(record["duration"])
        frame_times = frame_idx / fps
        start_target, end_target, start_times, end_times = boundary_targets(
            frame_times, record, positive_tolerance
        )
        target = torch.stack((start_target, end_target), dim=1)
        positive = target.any(dim=1)
        positive_indices = torch.nonzero(positive, as_tuple=False).flatten()
        if not len(positive_indices):
            continue
        excluded = torch.zeros(len(frame_times), dtype=torch.bool)
        for boundary_time in start_times + end_times:
            excluded |= (frame_times - boundary_time).abs() <= negative_exclusion
        negative_candidates = torch.nonzero(~excluded, as_tuple=False).flatten()
        requested_negatives = negative_ratio * len(positive_indices)
        if len(negative_candidates) > requested_negatives:
            order = torch.randperm(len(negative_candidates), generator=generator)
            negative_indices = negative_candidates[order[:requested_negatives]]
        else:
            negative_indices = negative_candidates
        selected_indices = torch.cat((positive_indices, negative_indices))
        order = torch.randperm(len(selected_indices), generator=generator)
        selected_indices = selected_indices[order]
        previous_indices = (selected_indices - dilation).clamp_min(0)
        next_indices = (selected_indices + dilation).clamp_max(len(features) - 1)
        windows.append(torch.stack((
            features[previous_indices],
            features[selected_indices],
            features[next_indices],
        ), dim=1).to(torch.float16))
        selected_targets = target[selected_indices].float()
        targets.append(selected_targets)
        positive_counts += selected_targets.sum(dim=0).long()
    if not windows:
        raise RuntimeError(f"No {subset} boundary samples for layer {layer_index}")
    return torch.cat(windows), torch.cat(targets), positive_counts


def train_boundary_linear_probe(
    train_windows,
    train_targets,
    seed,
    device,
    epochs,
    batch_size,
    learning_rate,
    weight_decay,
):
    set_probe_seed(seed)
    input_dimension = train_windows.shape[1] * train_windows.shape[2]
    model = torch.nn.Linear(input_dimension, 2).to(device)
    positive_counts = train_targets.sum(dim=0)
    negative_counts = len(train_targets) - positive_counts
    criterion = torch.nn.BCEWithLogitsLoss(
        pos_weight=(negative_counts / positive_counts.clamp_min(1.0)).to(device)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    final_loss = None
    model.train()
    for _ in range(epochs):
        permutation = torch.randperm(len(train_targets), generator=generator)
        loss_sum = 0.0
        sample_count = 0
        for start in range(0, len(permutation), batch_size):
            indices = permutation[start:start + batch_size]
            batch_windows = train_windows[indices].to(
                device=device, dtype=torch.float32, non_blocking=True
            ).flatten(start_dim=1)
            batch_targets = train_targets[indices].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_windows)
            loss = criterion(logits, batch_targets)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(indices)
            sample_count += len(indices)
        final_loss = loss_sum / sample_count
    return model, final_loss


def dense_boundary_scores(model, features, dilation, device):
    features = normalize_boundary_features(features).to(device)
    length, hidden_dimension = features.shape
    positions = torch.arange(length, device=device)
    previous = features[(positions - dilation).clamp_min(0)]
    following = features[(positions + dilation).clamp_max(length - 1)]
    weight = model.weight.view(2, 3, hidden_dimension)
    logits = (
        previous @ weight[:, 0, :].T
        + features @ weight[:, 1, :].T
        + following @ weight[:, 2, :].T
        + model.bias
    )
    return torch.sigmoid(logits).cpu()


def temporal_iou(predicted_start, predicted_end, target_start, target_end):
    if predicted_end <= predicted_start:
        return 0.0
    intersection = max(
        0.0,
        min(predicted_end, target_end) - max(predicted_start, target_start),
    )
    union = max(predicted_end, target_end) - min(predicted_start, target_start)
    return intersection / union if union > 0 else 0.0


def evaluate_boundary_probe(
    model,
    feature_root,
    layer_index,
    database,
    subset,
    device,
    dilation,
    evaluation_tolerance,
    search_radius,
):
    dense_scores = [[], []]
    dense_targets = [[], []]
    segment_video_names = []
    predicted_starts = []
    predicted_ends = []
    target_starts = []
    target_ends = []
    start_errors = []
    end_errors = []
    temporal_ious = []
    model.eval()
    with torch.no_grad():
        for video_name, record in sorted(database.items()):
            if record.get("subset") != subset:
                continue
            pack = torch.load(
                probe_feature_path(feature_root, layer_index, video_name),
                map_location="cpu",
            )
            frame_idx = pack["frame_idx"].float()
            fps = float(record["frame"]) / float(record["duration"])
            frame_times = frame_idx / fps
            scores = dense_boundary_scores(
                model, pack["pooled"], dilation, device
            )
            start_target, end_target, _, _ = boundary_targets(
                frame_times, record, evaluation_tolerance
            )
            for output_index, target in enumerate((start_target, end_target)):
                dense_scores[output_index].append(scores[:, output_index].numpy())
                dense_targets[output_index].append(target.numpy())
            for annotation in record.get("annotations", []):
                target_start, target_end = map(float, annotation["segment"])
                predictions = []
                for output_index, boundary_time in enumerate((target_start, target_end)):
                    search = (frame_times - boundary_time).abs() <= search_radius
                    if not search.any():
                        search[int(torch.argmin((frame_times - boundary_time).abs()))] = True
                    candidates = torch.nonzero(search, as_tuple=False).flatten()
                    best = candidates[int(torch.argmax(scores[candidates, output_index]))]
                    predictions.append(float(frame_times[best]))
                predicted_start, predicted_end = predictions
                segment_video_names.append(video_name)
                predicted_starts.append(predicted_start)
                predicted_ends.append(predicted_end)
                target_starts.append(target_start)
                target_ends.append(target_end)
                start_errors.append(abs(predicted_start - target_start))
                end_errors.append(abs(predicted_end - target_end))
                temporal_ious.append(temporal_iou(
                    predicted_start, predicted_end, target_start, target_end
                ))
    start_errors = np.asarray(start_errors, dtype=np.float64)
    end_errors = np.asarray(end_errors, dtype=np.float64)
    temporal_ious = np.asarray(temporal_ious, dtype=np.float64)
    metrics = {
        "start_AP": binary_average_precision(
            np.concatenate(dense_scores[0]), np.concatenate(dense_targets[0])
        ),
        "end_AP": binary_average_precision(
            np.concatenate(dense_scores[1]), np.concatenate(dense_targets[1])
        ),
        "start_MAE_seconds": float(start_errors.mean()),
        "end_MAE_seconds": float(end_errors.mean()),
        "mean_boundary_MAE_seconds": float(np.mean((start_errors + end_errors) / 2.0)),
        "start_recall_at_0.5s": float((start_errors <= 0.5).mean()),
        "end_recall_at_0.5s": float((end_errors <= 0.5).mean()),
        "start_recall_at_1.0s": float((start_errors <= 1.0).mean()),
        "end_recall_at_1.0s": float((end_errors <= 1.0).mean()),
        "oracle_tIoU_mean": float(temporal_ious.mean()),
        "oracle_tIoU_recall_at_0.5": float((temporal_ious >= 0.5).mean()),
        "oracle_tIoU_recall_at_0.7": float((temporal_ious >= 0.7).mean()),
    }
    predictions = {
        "video_names": np.asarray(segment_video_names),
        "predicted_starts": np.asarray(predicted_starts),
        "predicted_ends": np.asarray(predicted_ends),
        "target_starts": np.asarray(target_starts),
        "target_ends": np.asarray(target_ends),
        "start_errors": start_errors,
        "end_errors": end_errors,
        "temporal_ious": temporal_ious,
    }
    return metrics, predictions


def boundary_probe_main():
    import json

    annotation_path = os.getenv(
        "QWEN_PROBE_ANNOTATION",
        "/mnt/nvme1n1/zhouzhewen/OpenTAD-main/data/thumos-14/annotations/thumos_14_anno.json",
    )
    class_map_path = os.getenv(
        "QWEN_PROBE_CLASS_MAP",
        "/mnt/nvme1n1/zhouzhewen/OpenTAD-main/data/thumos-14/annotations/category_idx.txt",
    )
    feature_root = Path(os.getenv(
        "QWEN_PROBE_FEATURE_ROOT",
        "/mnt/nvme1n1/zhouzhewen/actionteller/qwen/data/thumos_hidden/frames_baseline08_all_layers_1_28",
    ))
    output_root = Path(os.getenv(
        "QWEN_PROBE_OUTPUT_ROOT",
        "exps/thumos/open-git/21_qwen_layerwise_boundary_probe",
    ))
    layers = resolve_probe_layers(feature_root, os.getenv("QWEN_PROBE_LAYERS", "all"))
    seeds = parse_integer_list(os.getenv("QWEN_PROBE_SEEDS", "42,2026,3407"))
    epochs = int(os.getenv("QWEN_PROBE_EPOCHS", "50"))
    batch_size = int(os.getenv("QWEN_PROBE_BATCH_SIZE", "512"))
    learning_rate = float(os.getenv("QWEN_PROBE_LR", "0.001"))
    weight_decay = float(os.getenv("QWEN_PROBE_WEIGHT_DECAY", "0.0001"))
    positive_tolerance = float(os.getenv("QWEN_BOUNDARY_TRAIN_POS_TOL", "0"))
    evaluation_tolerance = float(os.getenv("QWEN_BOUNDARY_EVAL_POS_TOL", "0.5"))
    negative_exclusion = float(os.getenv("QWEN_BOUNDARY_NEG_EXCLUSION", "0.5"))
    negative_ratio = int(os.getenv("QWEN_BOUNDARY_NEG_RATIO", "5"))
    dilation = int(os.getenv("QWEN_BOUNDARY_DILATION", "4"))
    search_radius = float(os.getenv("QWEN_BOUNDARY_SEARCH_RADIUS", "2.0"))
    data_seed = int(os.getenv("QWEN_BOUNDARY_DATA_SEED", "12345"))
    overwrite = os.getenv("QWEN_PROBE_OVERWRITE", "0") == "1"
    requested_device = os.getenv("QWEN_PROBE_DEVICE", "cuda")
    device = torch.device(
        requested_device if requested_device != "cuda" or torch.cuda.is_available()
        else "cpu"
    )
    output_root.mkdir(parents=True, exist_ok=True)
    database, _, _ = load_probe_annotations(annotation_path, class_map_path)
    database = limit_probe_database(
        database, int(os.getenv("QWEN_PROBE_VIDEO_LIMIT_PER_SPLIT", "0"))
    )
    print(
        f"[BOUNDARY-PROBE] layers={layers}; seeds={seeds}; device={device}; "
        f"epochs={epochs}; batch={batch_size}; dilation={dilation}; "
        f"search_radius={search_radius}s"
    )
    for layer_index in layers:
        result_path = output_root / f"layer_{layer_index:02d}.json"
        if result_path.exists() and not overwrite:
            print(f"[BOUNDARY-SKIP] layer={layer_index} result exists")
            continue
        train_windows, train_targets, positive_counts = build_boundary_probe_dataset(
            feature_root,
            layer_index,
            database,
            "training",
            positive_tolerance,
            negative_exclusion,
            negative_ratio,
            dilation,
            data_seed,
        )
        runs = []
        for seed in seeds:
            model, final_loss = train_boundary_linear_probe(
                train_windows,
                train_targets,
                seed,
                device,
                epochs,
                batch_size,
                learning_rate,
                weight_decay,
            )
            metrics, predictions = evaluate_boundary_probe(
                model,
                feature_root,
                layer_index,
                database,
                "validation",
                device,
                dilation,
                evaluation_tolerance,
                search_radius,
            )
            metrics["final_train_loss"] = final_loss
            prediction_path = output_root / (
                f"layer_{layer_index:02d}_seed_{seed}_predictions.npz"
            )
            np.savez_compressed(prediction_path, **predictions)
            runs.append({"seed": seed, **metrics})
            print(
                f"[BOUNDARY-RESULT] layer={layer_index} seed={seed} "
                f"mae={metrics['mean_boundary_MAE_seconds']:.6f}s "
                f"startAP={metrics['start_AP']:.6f} endAP={metrics['end_AP']:.6f} "
                f"oracle_tIoU={metrics['oracle_tIoU_mean']:.6f}"
            )
        summary_keys = (
            "start_AP",
            "end_AP",
            "start_MAE_seconds",
            "end_MAE_seconds",
            "mean_boundary_MAE_seconds",
            "start_recall_at_0.5s",
            "end_recall_at_0.5s",
            "start_recall_at_1.0s",
            "end_recall_at_1.0s",
            "oracle_tIoU_mean",
            "oracle_tIoU_recall_at_0.5",
            "oracle_tIoU_recall_at_0.7",
        )
        summary = {
            key: {
                "mean": float(np.mean([run[key] for run in runs])),
                "std": float(np.std([run[key] for run in runs], ddof=1)),
            }
            for key in summary_keys
        }
        result = {
            "probe": "class_agnostic_three_point_linear_boundary",
            "layer": layer_index,
            "feature_root": str(feature_root),
            "annotation_path": annotation_path,
            "train_subset": "training",
            "test_subset": "validation",
            "normalization": "per_frame_layer_norm_without_affine",
            "train_samples": int(len(train_targets)),
            "positive_counts_start_end": positive_counts.tolist(),
            "data_seed": data_seed,
            "model_seeds": list(seeds),
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "train_positive_tolerance_seconds": positive_tolerance,
            "evaluation_positive_tolerance_seconds": evaluation_tolerance,
            "negative_exclusion_seconds": negative_exclusion,
            "negative_ratio": negative_ratio,
            "dilation_in_sampled_frames": dilation,
            "oracle_search_radius_seconds": search_radius,
            "runs": runs,
            "summary": summary,
        }
        temporary_path = result_path.with_suffix(".json.tmp")
        temporary_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
        os.replace(temporary_path, result_path)
        print(f"[BOUNDARY-DONE] layer={layer_index} summary={summary}")
        del train_windows, train_targets
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


def probe_pipeline_main():
    import datetime
    import subprocess
    import sys
    import time

    feature_root = Path(os.getenv(
        "QWEN_PROBE_FEATURE_ROOT",
        "/mnt/nvme1n1/zhouzhewen/actionteller/qwen/data/thumos_hidden/frames_baseline08_all_layers_1_28",
    ))
    extraction_log_root = Path(os.getenv(
        "QWEN_EXTRACTION_LOG_ROOT",
        "exps/thumos/open-git/19_qwen_baseline08_all_layers_feature_extraction",
    ))
    classification_root = Path(os.getenv(
        "QWEN_CLASS_PROBE_OUTPUT_ROOT",
        "exps/thumos/open-git/20_qwen_layerwise_classification_probe",
    ))
    boundary_root = Path(os.getenv(
        "QWEN_BOUNDARY_PROBE_OUTPUT_ROOT",
        "exps/thumos/open-git/21_qwen_layerwise_boundary_probe",
    ))
    pipeline_root = Path(os.getenv(
        "QWEN_PROBE_PIPELINE_ROOT",
        "exps/thumos/open-git/22_qwen_probe_pipeline",
    ))
    expected_videos = int(os.getenv("QWEN_EXPECTED_VIDEOS", "413"))
    poll_seconds = int(os.getenv("QWEN_PIPELINE_POLL_SECONDS", "30"))
    pipeline_root.mkdir(parents=True, exist_ok=True)
    classification_root.mkdir(parents=True, exist_ok=True)
    boundary_root.mkdir(parents=True, exist_ok=True)
    status_path = pipeline_root / "status.log"

    def status(message):
        timestamp = datetime.datetime.now().isoformat(timespec="seconds")
        line = f"[{timestamp}] {message}"
        print(line, flush=True)
        with status_path.open("a") as stream:
            stream.write(line + "\n")

    status("WAITING for complete H1-H28 baseline08 feature extraction")
    while True:
        error_lines = []
        for log_path in sorted(extraction_log_root.glob("shard*.log")):
            for line in log_path.read_text(errors="replace").splitlines():
                if line.startswith("[ERR]"):
                    error_lines.append(f"{log_path.name}: {line}")
        if error_lines:
            status("BLOCKED because extraction errors were found: " + " | ".join(error_lines[:4]))
            raise RuntimeError("Feature extraction contains errors; probes were not launched.")
        counts = {
            layer: len(list((feature_root / f"layer_{layer}").glob("*/packed.pt")))
            for layer in range(1, 29)
        }
        minimum_count = min(counts.values()) if counts else 0
        maximum_count = max(counts.values()) if counts else 0
        if all(count == expected_videos for count in counts.values()):
            status(f"FEATURES complete: all 28 layers have {expected_videos} videos")
            break
        status(
            f"FEATURES progress: min_layer_count={minimum_count}, "
            f"max_layer_count={maximum_count}, expected={expected_videos}"
        )
        time.sleep(poll_seconds)

    # Allow extraction processes to save final logs and release their GPU contexts.
    time.sleep(30)
    layer_groups = (
        "1,5,9,13,17,21,25",
        "2,6,10,14,18,22,26",
        "3,7,11,15,19,23,27",
        "4,8,12,16,20,24,28",
    )

    def launch_stage(operation, output_root, log_prefix, extra_environment):
        processes = []
        log_streams = []
        for gpu_index, layer_group in enumerate(layer_groups):
            environment = os.environ.copy()
            environment.update({
                "CUDA_VISIBLE_DEVICES": str(gpu_index),
                "QWEN_OPERATION": operation,
                "QWEN_PROBE_FEATURE_ROOT": str(feature_root),
                "QWEN_PROBE_OUTPUT_ROOT": str(output_root),
                "QWEN_PROBE_LAYERS": layer_group,
                "QWEN_PROBE_DEVICE": "cuda",
                "QWEN_PROBE_SEEDS": "42,2026,3407",
                "PYTHONUNBUFFERED": "1",
                **extra_environment,
            })
            log_path = pipeline_root / f"{log_prefix}_gpu{gpu_index}.log"
            stream = log_path.open("a")
            process = subprocess.Popen(
                [sys.executable, "-u", str(Path(__file__).resolve())],
                cwd=str(Path(__file__).resolve().parents[1]),
                env=environment,
                stdout=stream,
                stderr=subprocess.STDOUT,
            )
            processes.append(process)
            log_streams.append(stream)
            status(
                f"LAUNCHED {operation} gpu={gpu_index} layers={layer_group} "
                f"pid={process.pid} log={log_path}"
            )
        return_codes = [process.wait() for process in processes]
        for stream in log_streams:
            stream.close()
        status(f"FINISHED {operation} return_codes={return_codes}")
        if any(return_code != 0 for return_code in return_codes):
            raise RuntimeError(
                f"{operation} failed with return codes {return_codes}; next stage not launched."
            )

    launch_stage(
        "classification_probe",
        classification_root,
        "classification",
        {
            "QWEN_PROBE_EPOCHS": "100",
            "QWEN_PROBE_BATCH_SIZE": "256",
            "QWEN_PROBE_LR": "0.01",
            "QWEN_PROBE_WEIGHT_DECAY": "0.0001",
        },
    )
    launch_stage(
        "boundary_probe",
        boundary_root,
        "boundary",
        {
            "QWEN_PROBE_EPOCHS": "50",
            "QWEN_PROBE_BATCH_SIZE": "512",
            "QWEN_PROBE_LR": "0.001",
            "QWEN_PROBE_WEIGHT_DECAY": "0.0001",
            "QWEN_BOUNDARY_TRAIN_POS_TOL": "0",
            "QWEN_BOUNDARY_EVAL_POS_TOL": "0.5",
            "QWEN_BOUNDARY_NEG_EXCLUSION": "0.5",
            "QWEN_BOUNDARY_NEG_RATIO": "5",
            "QWEN_BOUNDARY_DILATION": "4",
            "QWEN_BOUNDARY_SEARCH_RADIUS": "2.0",
            "QWEN_BOUNDARY_DATA_SEED": "12345",
        },
    )
    status("COMPLETE: classification and boundary probes finished successfully")


def _cluster_sums(video_names, values, ordered_videos):
    video_names = np.asarray(video_names).astype(str)
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    video_to_index = {name: index for index, name in enumerate(ordered_videos)}
    cluster_indices = np.asarray(
        [video_to_index[name] for name in video_names], dtype=np.int64
    )
    sums = np.zeros((len(ordered_videos), values.shape[1]), dtype=np.float64)
    np.add.at(sums, cluster_indices, values)
    return sums


def _bootstrap_summary(point_delta, bootstrap_delta):
    bootstrap_delta = np.asarray(bootstrap_delta, dtype=np.float64)
    lower, upper = np.percentile(bootstrap_delta, (2.5, 97.5))
    return {
        "point_delta": float(point_delta),
        "ci95_percentile": [float(lower), float(upper)],
        "favorable_probability": float((bootstrap_delta > 0).mean()),
        "ci_excludes_zero": bool(lower > 0 or upper < 0),
    }


def paired_video_bootstrap_main():
    import json

    classification_root = Path(os.getenv(
        "QWEN_CLASS_PROBE_OUTPUT_ROOT",
        "exps/thumos/open-git/20_qwen_layerwise_classification_probe",
    ))
    boundary_root = Path(os.getenv(
        "QWEN_BOUNDARY_PROBE_OUTPUT_ROOT",
        "exps/thumos/open-git/21_qwen_layerwise_boundary_probe",
    ))
    output_root = Path(os.getenv(
        "QWEN_BOOTSTRAP_OUTPUT_ROOT",
        "exps/thumos/open-git/23_qwen_probe_paired_video_bootstrap",
    ))
    layer_a = int(os.getenv("QWEN_BOOTSTRAP_LAYER_A", "21"))
    layer_b = int(os.getenv("QWEN_BOOTSTRAP_LAYER_B", "28"))
    seeds = parse_integer_list(os.getenv("QWEN_PROBE_SEEDS", "42,2026,3407"))
    repetitions = int(os.getenv("QWEN_BOOTSTRAP_REPETITIONS", "10000"))
    bootstrap_seed = int(os.getenv("QWEN_BOOTSTRAP_SEED", "20260904"))
    if repetitions <= 0:
        raise ValueError("QWEN_BOOTSTRAP_REPETITIONS must be positive")
    if not seeds:
        raise ValueError("QWEN_PROBE_SEEDS must not be empty")
    output_root.mkdir(parents=True, exist_ok=True)

    classification_pairs = {}
    boundary_pairs = {}
    reference_videos = None
    reference_targets = None
    for seed in seeds:
        class_pair = []
        boundary_pair = []
        for layer in (layer_a, layer_b):
            class_path = classification_root / (
                f"layer_{layer:02d}_seed_{seed}_predictions.npz"
            )
            boundary_path = boundary_root / (
                f"layer_{layer:02d}_seed_{seed}_predictions.npz"
            )
            if not class_path.exists() or not boundary_path.exists():
                raise FileNotFoundError(
                    f"Missing Probe predictions for layer={layer}, seed={seed}: "
                    f"class={class_path}, boundary={boundary_path}"
                )
            with np.load(class_path, allow_pickle=False) as packed:
                class_item = {key: packed[key] for key in packed.files}
            with np.load(boundary_path, allow_pickle=False) as packed:
                boundary_item = {key: packed[key] for key in packed.files}
            class_pair.append(class_item)
            boundary_pair.append(boundary_item)

        for item in class_pair:
            videos = np.asarray(item["video_names"]).astype(str)
            targets = np.asarray(item["targets"], dtype=np.int64)
            if reference_videos is None:
                reference_videos = videos
                reference_targets = targets
            elif not np.array_equal(videos, reference_videos):
                raise RuntimeError("Classification Probe video order is not aligned")
            elif not np.array_equal(targets, reference_targets):
                raise RuntimeError("Classification Probe targets are not aligned")
        for item in boundary_pair:
            videos = np.asarray(item["video_names"]).astype(str)
            if not np.array_equal(videos, reference_videos):
                raise RuntimeError("Boundary Probe video order is not aligned")
        classification_pairs[seed] = class_pair
        boundary_pairs[seed] = boundary_pair

    ordered_videos = tuple(sorted(set(reference_videos.tolist())))
    num_videos = len(ordered_videos)
    rng = np.random.default_rng(bootstrap_seed)
    sampled_video_counts = rng.multinomial(
        num_videos,
        np.full(num_videos, 1.0 / num_videos),
        size=repetitions,
    ).astype(np.float64)

    class_totals = np.eye(20, dtype=np.float64)[reference_targets]
    video_class_totals = _cluster_sums(
        reference_videos, class_totals, ordered_videos
    )
    bootstrap_class_totals = sampled_video_counts @ video_class_totals
    bootstrap_segment_totals = bootstrap_class_totals.sum(axis=1)
    distributions = {}
    report = {
        "analysis": "paired_video_cluster_bootstrap",
        "layer_a": layer_a,
        "layer_b": layer_b,
        "delta_convention": {
            "classification": "layer_a - layer_b; positive favors layer_a",
            "boundary_MAE": "layer_b - layer_a; positive favors layer_a",
            "boundary_tIoU_and_recall": "layer_a - layer_b; positive favors layer_a",
        },
        "seeds": list(seeds),
        "bootstrap_seed": bootstrap_seed,
        "repetitions": repetitions,
        "sampling_unit": "video",
        "num_videos": num_videos,
        "num_segments": int(len(reference_videos)),
        "classification": {},
        "boundary": {},
        "limitations": [
            "The bootstrap quantifies validation-video sampling uncertainty conditional on each trained Probe.",
            "Variation across the three model seeds is reported separately.",
            "H21 versus H28 was pre-specified; post-hoc best layers H15/H27 require independent confirmation.",
        ],
    }

    class_metric_deltas = {
        "top1_accuracy": [],
        "balanced_accuracy": [],
    }
    for seed, pair in classification_pairs.items():
        metric_values = []
        for item in pair:
            predictions = np.asarray(item["probabilities"]).argmax(axis=1)
            correct = predictions == reference_targets
            video_correct = _cluster_sums(
                reference_videos, correct.astype(np.float64), ordered_videos
            )[:, 0]
            top1 = (
                sampled_video_counts @ video_correct
                / np.maximum(bootstrap_segment_totals, 1.0)
            )
            correct_by_class = (correct[:, None] * class_totals).astype(np.float64)
            video_correct_by_class = _cluster_sums(
                reference_videos, correct_by_class, ordered_videos
            )
            bootstrap_correct_by_class = sampled_video_counts @ video_correct_by_class
            with np.errstate(divide="ignore", invalid="ignore"):
                recalls = np.divide(
                    bootstrap_correct_by_class,
                    bootstrap_class_totals,
                    out=np.full_like(bootstrap_correct_by_class, np.nan),
                    where=bootstrap_class_totals > 0,
                )
            balanced = np.nanmean(recalls, axis=1)
            full_metrics = classification_probe_metrics(
                item["probabilities"], reference_targets, 20
            )
            metric_values.append({
                "top1_accuracy": (full_metrics["top1_accuracy"], top1),
                "balanced_accuracy": (
                    full_metrics["balanced_accuracy"], balanced
                ),
                "class_mAP_point_only": full_metrics["class_mAP"],
            })

        seed_report = {
            "layer_a_point": {
                "top1_accuracy": metric_values[0]["top1_accuracy"][0],
                "balanced_accuracy": metric_values[0]["balanced_accuracy"][0],
                "class_mAP": metric_values[0]["class_mAP_point_only"],
            },
            "layer_b_point": {
                "top1_accuracy": metric_values[1]["top1_accuracy"][0],
                "balanced_accuracy": metric_values[1]["balanced_accuracy"][0],
                "class_mAP": metric_values[1]["class_mAP_point_only"],
            },
            "paired_deltas": {},
        }
        for metric_name in class_metric_deltas:
            point_delta = (
                metric_values[0][metric_name][0]
                - metric_values[1][metric_name][0]
            )
            bootstrap_delta = (
                metric_values[0][metric_name][1]
                - metric_values[1][metric_name][1]
            )
            class_metric_deltas[metric_name].append(bootstrap_delta)
            distributions[f"classification_{metric_name}_seed_{seed}"] = bootstrap_delta
            seed_report["paired_deltas"][metric_name] = _bootstrap_summary(
                point_delta, bootstrap_delta
            )
        report["classification"][f"seed_{seed}"] = seed_report

    for metric_name, per_seed in class_metric_deltas.items():
        seed_mean_delta = np.mean(np.stack(per_seed, axis=0), axis=0)
        point_deltas = [
            report["classification"][f"seed_{seed}"]["paired_deltas"][metric_name]["point_delta"]
            for seed in seeds
        ]
        report["classification"][f"seed_mean_{metric_name}"] = {
            **_bootstrap_summary(float(np.mean(point_deltas)), seed_mean_delta),
            "point_delta_seed_std": float(np.std(point_deltas, ddof=1)),
            "direction_consistent_across_seeds": bool(
                all(value > 0 for value in point_deltas)
                or all(value < 0 for value in point_deltas)
            ),
        }
        distributions[f"classification_{metric_name}_seed_mean"] = seed_mean_delta

    boundary_metric_deltas = {
        "mean_boundary_MAE_seconds": [],
        "oracle_tIoU_mean": [],
        "oracle_tIoU_recall_at_0.7": [],
    }
    for seed, pair in boundary_pairs.items():
        metric_values = []
        for item in pair:
            boundary_mae = (
                np.asarray(item["start_errors"], dtype=np.float64)
                + np.asarray(item["end_errors"], dtype=np.float64)
            ) / 2.0
            temporal_ious = np.asarray(item["temporal_ious"], dtype=np.float64)
            per_segment = {
                "mean_boundary_MAE_seconds": boundary_mae,
                "oracle_tIoU_mean": temporal_ious,
                "oracle_tIoU_recall_at_0.7": (temporal_ious >= 0.7).astype(np.float64),
            }
            layer_metrics = {}
            for metric_name, values in per_segment.items():
                video_sums = _cluster_sums(
                    reference_videos, values, ordered_videos
                )[:, 0]
                video_counts = _cluster_sums(
                    reference_videos, np.ones_like(values), ordered_videos
                )[:, 0]
                bootstrap_values = (
                    sampled_video_counts @ video_sums
                    / np.maximum(sampled_video_counts @ video_counts, 1.0)
                )
                layer_metrics[metric_name] = (float(values.mean()), bootstrap_values)
            metric_values.append(layer_metrics)

        seed_report = {
            "layer_a_point": {
                key: metric_values[0][key][0] for key in boundary_metric_deltas
            },
            "layer_b_point": {
                key: metric_values[1][key][0] for key in boundary_metric_deltas
            },
            "paired_deltas": {},
        }
        for metric_name in boundary_metric_deltas:
            if metric_name == "mean_boundary_MAE_seconds":
                point_delta = (
                    metric_values[1][metric_name][0]
                    - metric_values[0][metric_name][0]
                )
                bootstrap_delta = (
                    metric_values[1][metric_name][1]
                    - metric_values[0][metric_name][1]
                )
            else:
                point_delta = (
                    metric_values[0][metric_name][0]
                    - metric_values[1][metric_name][0]
                )
                bootstrap_delta = (
                    metric_values[0][metric_name][1]
                    - metric_values[1][metric_name][1]
                )
            boundary_metric_deltas[metric_name].append(bootstrap_delta)
            distributions[f"boundary_{metric_name}_seed_{seed}"] = bootstrap_delta
            seed_report["paired_deltas"][metric_name] = _bootstrap_summary(
                point_delta, bootstrap_delta
            )
        report["boundary"][f"seed_{seed}"] = seed_report

    for metric_name, per_seed in boundary_metric_deltas.items():
        seed_mean_delta = np.mean(np.stack(per_seed, axis=0), axis=0)
        point_deltas = [
            report["boundary"][f"seed_{seed}"]["paired_deltas"][metric_name]["point_delta"]
            for seed in seeds
        ]
        report["boundary"][f"seed_mean_{metric_name}"] = {
            **_bootstrap_summary(float(np.mean(point_deltas)), seed_mean_delta),
            "point_delta_seed_std": float(np.std(point_deltas, ddof=1)),
            "direction_consistent_across_seeds": bool(
                all(value > 0 for value in point_deltas)
                or all(value < 0 for value in point_deltas)
            ),
        }
        distributions[f"boundary_{metric_name}_seed_mean"] = seed_mean_delta

    summary_path = output_root / f"layer_{layer_a:02d}_vs_{layer_b:02d}_summary.json"
    distribution_path = output_root / (
        f"layer_{layer_a:02d}_vs_{layer_b:02d}_bootstrap_distributions.npz"
    )
    temporary_path = summary_path.with_suffix(".json.tmp")
    temporary_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    os.replace(temporary_path, summary_path)
    np.savez_compressed(distribution_path, **distributions)
    print(f"[BOOTSTRAP-DONE] summary={summary_path}")
    print(f"[BOOTSTRAP-DONE] distributions={distribution_path}")


def probe_pdf_report_main():
    import datetime
    import json
    import textwrap

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.font_manager import FontProperties, fontManager

    classification_root = Path(os.getenv(
        "QWEN_CLASS_PROBE_OUTPUT_ROOT",
        "exps/thumos/open-git/20_qwen_layerwise_classification_probe",
    ))
    boundary_root = Path(os.getenv(
        "QWEN_BOUNDARY_PROBE_OUTPUT_ROOT",
        "exps/thumos/open-git/21_qwen_layerwise_boundary_probe",
    ))
    bootstrap_path = Path(os.getenv(
        "QWEN_BOOTSTRAP_SUMMARY",
        "exps/thumos/open-git/23_qwen_probe_paired_video_bootstrap/"
        "layer_21_vs_28_summary.json",
    ))
    output_path = Path(os.getenv(
        "QWEN_PROBE_PDF_PATH",
        "文档/Qwen分层语义_分类与边界Probe实验报告.pdf",
    ))
    font_path = Path(os.getenv(
        "QWEN_REPORT_FONT",
        "/mnt/nvme1n1/zhouzhewen/video-mamba-suite-main/video-mamba-suite/"
        "video-dense-captioning/visualization/NotoSansCJK-Bold.otf",
    ))
    if not font_path.exists():
        raise FileNotFoundError(f"Missing Chinese font: {font_path}")
    fontManager.addfont(str(font_path))
    chinese_font = FontProperties(fname=str(font_path))
    font_family = chinese_font.get_name()
    matplotlib.rcParams.update({
        "font.family": font_family,
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
    })

    class_results = {}
    boundary_results = {}
    for layer in range(1, 29):
        class_path = classification_root / f"layer_{layer:02d}.json"
        boundary_path = boundary_root / f"layer_{layer:02d}.json"
        if not class_path.exists() or not boundary_path.exists():
            raise FileNotFoundError(
                f"Missing Probe result for layer {layer}: "
                f"class={class_path}, boundary={boundary_path}"
            )
        class_results[layer] = json.loads(class_path.read_text())
        boundary_results[layer] = json.loads(boundary_path.read_text())
    bootstrap = json.loads(bootstrap_path.read_text())
    output_path.parent.mkdir(parents=True, exist_ok=True)

    page_size = (8.27, 11.69)
    navy = "#16324F"
    blue = "#246BCE"
    orange = "#D97706"
    green = "#17803D"
    red = "#B42318"
    gray = "#52606D"
    light = "#EEF3F8"

    def new_page(title, subtitle=None):
        fig = plt.figure(figsize=page_size, facecolor="white")
        fig.text(
            0.07, 0.955, title, fontsize=19, fontweight="bold",
            color=navy, fontproperties=chinese_font, va="top",
        )
        if subtitle:
            fig.text(
                0.07, 0.92, subtitle, fontsize=9.5, color=gray,
                fontproperties=chinese_font, va="top",
            )
        fig.lines.append(plt.Line2D(
            [0.07, 0.93], [0.9, 0.9], transform=fig.transFigure,
            color=blue, linewidth=1.4,
        ))
        fig.text(
            0.93, 0.025, "Qwen分层语义 Probe 实验报告",
            ha="right", fontsize=8, color="#718096",
            fontproperties=chinese_font,
        )
        return fig

    def add_bullets(fig, items, x=0.09, y=0.84, width=48, fontsize=10.8,
                    line_height=0.032, color="#263746"):
        current_y = y
        for item in items:
            wrapped = textwrap.wrap(item, width=width) or [item]
            for line_index, line in enumerate(wrapped):
                prefix = "• " if line_index == 0 else "  "
                fig.text(
                    x, current_y, prefix + line, fontsize=fontsize,
                    color=color, fontproperties=chinese_font, va="top",
                )
                current_y -= line_height
            current_y -= line_height * 0.35
        return current_y

    def add_section(fig, title, y):
        fig.text(
            0.08, y, title, fontsize=13, fontweight="bold", color=navy,
            fontproperties=chinese_font, va="top",
        )
        return y - 0.045

    def add_table(fig, bbox, columns, rows, font_size=9):
        ax = fig.add_axes(bbox)
        ax.axis("off")
        table = ax.table(
            cellText=rows, colLabels=columns, cellLoc="center",
            colLoc="center", loc="center", bbox=[0, 0, 1, 1],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(font_size)
        for (row, _), cell in table.get_celld().items():
            cell.get_text().set_fontproperties(chinese_font)
            cell.set_edgecolor("#CBD5E1")
            cell.set_linewidth(0.6)
            if row == 0:
                cell.set_facecolor("#DCE8F5")
                cell.get_text().set_weight("bold")
                cell.get_text().set_color(navy)
            elif row % 2 == 0:
                cell.set_facecolor("#F7FAFC")
        return table

    layers = np.arange(1, 29)
    top1 = np.asarray([
        class_results[layer]["summary"]["top1_accuracy"]["mean"]
        for layer in layers
    ])
    top1_std = np.asarray([
        class_results[layer]["summary"]["top1_accuracy"]["std"]
        for layer in layers
    ])
    balanced = np.asarray([
        class_results[layer]["summary"]["balanced_accuracy"]["mean"]
        for layer in layers
    ])
    class_map = np.asarray([
        class_results[layer]["summary"]["class_mAP"]["mean"]
        for layer in layers
    ])
    boundary_mae = np.asarray([
        boundary_results[layer]["summary"]["mean_boundary_MAE_seconds"]["mean"]
        for layer in layers
    ])
    boundary_mae_std = np.asarray([
        boundary_results[layer]["summary"]["mean_boundary_MAE_seconds"]["std"]
        for layer in layers
    ])
    oracle_tiou = np.asarray([
        boundary_results[layer]["summary"]["oracle_tIoU_mean"]["mean"]
        for layer in layers
    ])
    oracle_recall_07 = np.asarray([
        boundary_results[layer]["summary"]["oracle_tIoU_recall_at_0.7"]["mean"]
        for layer in layers
    ])

    selected_layers = (7, 14, 21, 28)
    best_top1_layer = int(layers[np.argmax(top1)])
    best_map_layer = int(layers[np.argmax(class_map)])
    best_mae_layer = int(layers[np.argmin(boundary_mae)])
    best_tiou_layer = int(layers[np.argmax(oracle_tiou)])

    metadata = {
        "Title": "Qwen分层语义：分类与边界Probe实验报告",
        "Author": "ActionTeller实验记录",
        "Subject": "THUMOS14 Qwen2.5-VL 28层语义诊断",
        "Keywords": "Qwen, THUMOS14, classification probe, boundary probe",
        "CreationDate": datetime.datetime.now(),
    }
    with PdfPages(output_path, metadata=metadata) as pdf:
        fig = new_page(
            "Qwen分层语义：分类与边界 Probe 实验报告",
            f"THUMOS14｜H1–H28｜3个模型seed｜生成日期：{datetime.date.today().isoformat()}",
        )
        fig.text(
            0.08, 0.83, "核心问题", fontsize=15, fontweight="bold",
            color=navy, fontproperties=chinese_font,
        )
        fig.text(
            0.08, 0.775,
            "Qwen不同深度是否保留了对动作分类和时间边界定位不同程度的有效信息？",
            fontsize=12.5, color="#1F2937", fontproperties=chinese_font,
        )
        ax = fig.add_axes([0.08, 0.57, 0.84, 0.15])
        ax.axis("off")
        ax.add_patch(plt.Rectangle((0, 0), 1, 1, facecolor=light, edgecolor="#B8CCE2"))
        ax.text(
            0.04, 0.72, "当前结论（严格表述）", fontsize=13, fontweight="bold",
            color=navy, fontproperties=chinese_font, transform=ax.transAxes,
        )
        ax.text(
            0.04, 0.43,
            "分类Top-1整体偏向后层；边界指标在H15–H21附近更好。",
            fontsize=11.5, color="#243B53", fontproperties=chinese_font,
            transform=ax.transAxes,
        )
        ax.text(
            0.04, 0.17,
            "结果支持继续验证任务特定静态融合，但尚不能证明视频级或时间级动态路由有效。",
            fontsize=10.8, color=red, fontproperties=chinese_font,
            transform=ax.transAxes,
        )
        y = add_section(fig, "关键数值", 0.51)
        add_bullets(fig, [
            f"分类Top-1最优为H{best_top1_layer}：{top1[best_top1_layer-1]*100:.2f}%；"
            f"H28为{top1[27]*100:.2f}%。",
            f"边界MAE最优为H{best_mae_layer}：{boundary_mae[best_mae_layer-1]:.4f}秒；"
            f"H28为{boundary_mae[27]:.4f}秒。",
            f"Oracle tIoU最优为H{best_tiou_layer}：{oracle_tiou[best_tiou_layer-1]:.4f}；"
            f"H28为{oracle_tiou[27]:.4f}。",
            "H21与H28的预设比较通过视频级配对Bootstrap进一步检验，边界差异区间不跨0。",
        ], y=y, width=52)
        fig.text(
            0.08, 0.10,
            "注意：Probe是冻结特征上的诊断实验，不等同于完整动作检测器的最终mAP。",
            fontsize=10.5, color=red, fontproperties=chinese_font,
        )
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig = new_page("1. 实验条件与 Probe 定义", "所有层均来自一次统一的Qwen前向，排除Prompt和pooling差异")
        y = add_section(fig, "统一特征条件", 0.86)
        y = add_bullets(fig, [
            "模型：Qwen2.5-VL-7B，共28个decoder block；H28对应08使用的last hidden state。",
            "Prompt：限定THUMOS14的20个动作类别，不提供单个视频的真实标签。",
            "视频读取：Decord RGB；采样步长为4帧；每帧独立构造视觉—文本输入。",
            "Token pooling：按attention mask平均所有有效图像token与文本token。",
            "完整性：每层413个视频；新H28与08原frames_pooled逐数值核对，最大误差为0。",
        ], y=y, width=53)
        y = add_section(fig, "分类 Probe", y - 0.01)
        y = add_bullets(fig, [
            "以真实动作片段为单位，将片段内部的帧级Qwen特征做时间平均。",
            "训练集：200个视频、3003个片段；测试集：211个视频、3325个片段。",
            "冻结Qwen特征，仅训练一个线性分类头；训练集逐维z-score标准化。",
            "AdamW训练100轮，batch size 256，学习率0.01，weight decay 1e-4。",
            "指标：Top-1 Accuracy、Balanced Accuracy、20类平均AP。",
        ], y=y, width=53)
        y = add_section(fig, "边界 Probe", y - 0.01)
        add_bullets(fig, [
            "类别无关的三点时序线性Probe，用局部上下文判断动作起点与终点。",
            "训练正样本取最接近真实边界的采样位置；0.5秒内不作为负样本；负正比5:1。",
            "每帧特征做无仿射LayerNorm；AdamW训练50轮，batch size 512，学习率0.001。",
            "指标包含Start/End AP、边界MAE、±2秒局部搜索下的Oracle tIoU及Recall。",
            "模型seed均为42、2026、3407；边界负样本数据seed固定为12345。",
        ], y=y, width=53)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig = new_page("2. 分类 Probe：28层结果", "纵轴为3个seed的均值；误差条仅展示Top-1的seed标准差")
        ax = fig.add_axes([0.10, 0.48, 0.82, 0.36])
        ax.errorbar(
            layers, top1 * 100, yerr=top1_std * 100,
            color=blue, marker="o", markersize=3, linewidth=1.5,
            capsize=2, label="Top-1",
        )
        ax.plot(layers, balanced * 100, color=green, linewidth=1.4, label="Balanced Acc.")
        ax.plot(layers, class_map * 100, color=orange, linewidth=1.4, label="Class mAP")
        for layer in selected_layers:
            ax.axvline(layer, color="#B7C4D1", linewidth=0.7, linestyle="--")
        ax.set_xlabel("Qwen层级")
        ax.set_ylabel("指标（%）")
        ax.set_xticks(list(range(1, 29, 2)) + [28])
        ax.grid(alpha=0.2)
        ax.legend(loc="lower right", frameon=False)
        rows = []
        for layer in selected_layers:
            summary = class_results[layer]["summary"]
            rows.append([
                f"H{layer}",
                f"{summary['top1_accuracy']['mean']*100:.2f} ± {summary['top1_accuracy']['std']*100:.2f}",
                f"{summary['balanced_accuracy']['mean']*100:.2f}",
                f"{summary['class_mAP']['mean']*100:.2f} ± {summary['class_mAP']['std']*100:.2f}",
            ])
        add_table(
            fig, [0.09, 0.26, 0.83, 0.16],
            ["层级", "Top-1（%）", "Balanced（%）", "Class mAP（%）"],
            rows, font_size=9,
        )
        y = add_section(fig, "解释", 0.21)
        add_bullets(fig, [
            f"Top-1最优为H{best_top1_layer}（{top1[best_top1_layer-1]*100:.2f}%），"
            f"H28为{top1[27]*100:.2f}%，二者仅差{(top1[best_top1_layer-1]-top1[27])*100:.2f}个百分点。",
            "从H7、H14、H21到H28，Top-1总体上升，支持“分类更偏后层”的趋势判断。",
            "部分层的Class mAP seed波动较大，因此不能仅依据单个seed或mAP最高层做强结论。",
        ], y=y, width=55, fontsize=9.8, line_height=0.029)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig = new_page("3. 边界 Probe：28层结果", "MAE越低越好；Oracle tIoU与Recall越高越好")
        ax1 = fig.add_axes([0.10, 0.55, 0.82, 0.29])
        ax1.errorbar(
            layers, boundary_mae, yerr=boundary_mae_std,
            color=red, marker="o", markersize=3, capsize=2,
            linewidth=1.5, label="Mean boundary MAE",
        )
        ax1.axvline(21, color=blue, linestyle="--", linewidth=1, label="H21")
        ax1.axvline(28, color=gray, linestyle="--", linewidth=1, label="H28")
        ax1.set_ylabel("MAE（秒，越低越好）")
        ax1.set_xticks(list(range(1, 29, 2)) + [28])
        ax1.grid(alpha=0.2)
        ax1.legend(frameon=False, ncol=3, fontsize=8)
        ax2 = fig.add_axes([0.10, 0.30, 0.82, 0.20])
        ax2.plot(layers, oracle_tiou * 100, color=blue, marker="o", markersize=3, label="Oracle tIoU")
        ax2.plot(layers, oracle_recall_07 * 100, color=orange, marker="o", markersize=3, label="Oracle R@0.7")
        ax2.set_xlabel("Qwen层级")
        ax2.set_ylabel("指标（%）")
        ax2.set_xticks(list(range(1, 29, 2)) + [28])
        ax2.grid(alpha=0.2)
        ax2.legend(frameon=False, ncol=2, fontsize=8)
        rows = []
        for layer in selected_layers:
            summary = boundary_results[layer]["summary"]
            rows.append([
                f"H{layer}",
                f"{summary['mean_boundary_MAE_seconds']['mean']:.4f}",
                f"{summary['oracle_tIoU_mean']['mean']:.4f}",
                f"{summary['oracle_tIoU_recall_at_0.7']['mean']*100:.2f}",
            ])
        add_table(
            fig, [0.09, 0.135, 0.83, 0.12],
            ["层级", "边界MAE（秒）", "Oracle tIoU", "Oracle R@0.7（%）"],
            rows, font_size=8.8,
        )
        fig.text(
            0.09, 0.085,
            f"最优MAE：H{best_mae_layer}={boundary_mae[best_mae_layer-1]:.4f}秒；"
            f"最优Oracle tIoU：H{best_tiou_layer}={oracle_tiou[best_tiou_layer-1]:.4f}。"
            " 中后层达到峰值后，H28出现回落。",
            fontsize=9.5, color="#263746", fontproperties=chinese_font,
        )
        fig.text(
            0.09, 0.055,
            "限制：25毫秒是3325个片段的平均误差差值，小于一个采样时间步；实际意义必须由完整检测器验证。",
            fontsize=9.3, color=red, fontproperties=chinese_font,
        )
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig = new_page("4. H21 与 H28：按视频配对 Bootstrap", "10,000次重采样｜211个视频｜3325个动作片段｜3个模型seed")
        class_top1_boot = bootstrap["classification"]["seed_mean_top1_accuracy"]
        class_bal_boot = bootstrap["classification"]["seed_mean_balanced_accuracy"]
        boundary_mae_boot = bootstrap["boundary"]["seed_mean_mean_boundary_MAE_seconds"]
        boundary_tiou_boot = bootstrap["boundary"]["seed_mean_oracle_tIoU_mean"]
        boundary_r07_boot = bootstrap["boundary"]["seed_mean_oracle_tIoU_recall_at_0.7"]
        rows = [
            [
                "分类Top-1",
                f"{class_top1_boot['point_delta']*100:+.3f} pp",
                f"[{class_top1_boot['ci95_percentile'][0]*100:+.3f}, "
                f"{class_top1_boot['ci95_percentile'][1]*100:+.3f}] pp",
                "H28更好" if class_top1_boot["point_delta"] < 0 else "H21更好",
            ],
            [
                "分类Balanced Acc.",
                f"{class_bal_boot['point_delta']*100:+.3f} pp",
                f"[{class_bal_boot['ci95_percentile'][0]*100:+.3f}, "
                f"{class_bal_boot['ci95_percentile'][1]*100:+.3f}] pp",
                "不确定" if not class_bal_boot["ci_excludes_zero"] else "存在差异",
            ],
            [
                "边界MAE改善",
                f"{boundary_mae_boot['point_delta']*1000:+.2f} ms",
                f"[{boundary_mae_boot['ci95_percentile'][0]*1000:+.2f}, "
                f"{boundary_mae_boot['ci95_percentile'][1]*1000:+.2f}] ms",
                "H21更好",
            ],
            [
                "Oracle tIoU",
                f"{boundary_tiou_boot['point_delta']*100:+.3f} pp",
                f"[{boundary_tiou_boot['ci95_percentile'][0]*100:+.3f}, "
                f"{boundary_tiou_boot['ci95_percentile'][1]*100:+.3f}] pp",
                "H21更好",
            ],
            [
                "Oracle Recall@0.7",
                f"{boundary_r07_boot['point_delta']*100:+.3f} pp",
                f"[{boundary_r07_boot['ci95_percentile'][0]*100:+.3f}, "
                f"{boundary_r07_boot['ci95_percentile'][1]*100:+.3f}] pp",
                "H21更好",
            ],
        ]
        add_table(
            fig, [0.07, 0.58, 0.86, 0.25],
            ["指标", "H21相对H28差值", "95%置信区间", "解释"],
            rows, font_size=8.7,
        )
        y = add_section(fig, "边界结果的逐seed一致性", 0.52)
        seed_rows = []
        for seed in (42, 2026, 3407):
            seed_result = bootstrap["boundary"][f"seed_{seed}"]["paired_deltas"]
            seed_rows.append([
                str(seed),
                f"{seed_result['mean_boundary_MAE_seconds']['point_delta']*1000:.2f} ms",
                f"{seed_result['oracle_tIoU_mean']['point_delta']*100:.3f} pp",
                f"{seed_result['oracle_tIoU_recall_at_0.7']['point_delta']*100:.3f} pp",
            ])
        add_table(
            fig, [0.12, 0.33, 0.76, 0.15],
            ["seed", "MAE改善", "tIoU提升", "R@0.7提升"],
            seed_rows, font_size=9,
        )
        y = add_section(fig, "统计解释", 0.27)
        add_bullets(fig, [
            "H21的三个边界指标在三个seed下均优于H28，并且各自的95%视频Bootstrap区间均不跨0。",
            "分类Top-1中H21在三个seed下均低于H28；seed均值差为-1.45个百分点，支持分类更偏后层。",
            "Balanced Accuracy区间跨0，说明该指标不足以支持H21与H28存在稳定差异。",
            "Bootstrap处理测试视频采样不确定性；三个模型seed处理Probe训练随机性，两者不能相互替代。",
            "Class mAP本页仅保留点估计，未做排序型AP的cluster bootstrap，不能据此给出其置信区间。",
        ], y=y, width=55, fontsize=9.4, line_height=0.027)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig = new_page("5. 结论边界：结果能证明什么", "以下表述可直接用于后续论文方法动机，但不能替代完整检测实验")
        y = add_section(fig, "目前得到支持的结论", 0.85)
        y = add_bullets(fig, [
            "在统一08提取条件下，Qwen层级与任务有效信息之间存在可重复的关联。",
            "预设比较H21对H28显示：H28分类Top-1更高，而H21边界MAE、Oracle tIoU和R@0.7更好。",
            "因此，分类与定位共享同一个固定Qwen层可能不是最优选择，任务特定静态融合值得进一步验证。",
        ], y=y, width=54)
        y = add_section(fig, "目前不能证明的结论", y - 0.015)
        y = add_bullets(fig, [
            "不能证明动态Router一定有效；Probe没有验证不同视频或不同时间点需要不同权重。",
            "不能将25毫秒Probe MAE改善直接等同于完整检测器mAP提升。",
            "不能宣称H15或H27是普适最优层：它们是查看28层结果后得到的探索性最优。",
            "不能只报告单次最好结果；完整检测器旧08同seed重跑仍有明显波动。",
        ], y=y, width=54, color="#5B2533")
        y = add_section(fig, "计算与存储", y - 0.01)
        add_bullets(fig, [
            "得到H28时Qwen本来已经经过H7/H14/H21，因此固定使用少数中间层不会再次运行Qwen主干。",
            "额外成本主要来自保存中间特征、I/O以及少量投影和融合模块；不应一开始保留全部28层做复杂路由。",
        ], y=y, width=54)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig = new_page("6. 下一阶段：S0–S4完整检测实验", "候选层固定为H7/H14/H21/H28；先验证静态任务分层，再决定是否做动态路由")
        rows = [
            ["S0", "仅H28", "严格baseline"],
            ["S1", "四层等权平均", "多层信息本身是否有效"],
            ["S2", "共享的可学习静态权重", "学习权重是否优于平均"],
            ["S3", "分类H28、定位H21", "最简单的任务硬选层是否有效"],
            ["S4", "分类/定位分别学习静态权重", "核心：任务特定权重是否优于共享权重"],
        ]
        add_table(
            fig, [0.08, 0.57, 0.84, 0.25],
            ["配置", "方法", "验证目标"], rows, font_size=9,
        )
        y = add_section(fig, "统一运行规范", 0.51)
        y = add_bullets(fig, [
            "全部使用严格复现08的新特征；相同数据、优化器、训练轮数和checkpoint选择规则。",
            "每种配置100 epoch，使用seed 42、2026、3407，共15次完整训练；不选择其中最好一次。",
            "报告mAP@0.3–0.7、平均mAP、均值±标准差、配对差值、参数量、显存和推理时间。",
            "S4必须稳定超过S0和S2，且提升超出baseline波动，才有依据继续视频级Router。",
            "视频级Router必须超过S4，才考虑时间级动态选层。",
        ], y=y, width=55)
        fig.text(
            0.08, 0.18, "核心判据", fontsize=13, fontweight="bold",
            color=navy, fontproperties=chinese_font,
        )
        fig.text(
            0.08, 0.135,
            "本研究最终是否成立，不取决于Probe的25毫秒本身，而取决于S4是否以可接受代价"
            "稳定提高完整检测器，尤其是高tIoU mAP。",
            fontsize=11, color=red, fontproperties=chinese_font,
        )
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    print(f"[PDF-DONE] {output_path.resolve()}")


def average_hidden_layers_main():
    """Build S1 features by equally averaging already extracted hidden layers.

    The average is computed in float32 and stored in the source feature dtype.
    Keeping the detector unchanged makes S0/S1 differ only in their Qwen input.
    """
    source_root = Path(os.getenv(
        "QWEN_AVERAGE_SOURCE_ROOT",
        "/mnt/nvme1n1/zhouzhewen/actionteller/qwen/data/thumos_hidden/"
        "frames_llm_image_tokens_single_layers_7_14_21_28",
    ))
    output_root = Path(os.getenv(
        "QWEN_AVERAGE_OUTPUT_ROOT",
        "/mnt/nvme1n1/zhouzhewen/actionteller/qwen/data/thumos_hidden/"
        "frames_llm_image_tokens_mean_layers_7_14_21_28",
    ))
    layers = parse_hidden_layers(os.getenv("QWEN_AVERAGE_LAYERS", "7,14,21,28"))
    if layers is None:
        raise ValueError("QWEN_AVERAGE_LAYERS must be an explicit layer list, not 'all'.")
    if output_root.resolve() == source_root.resolve():
        raise ValueError("S1 output root must differ from its source root.")

    layer_roots = {layer: source_root / f"layer_{layer}" for layer in layers}
    missing_roots = [str(path) for path in layer_roots.values() if not path.is_dir()]
    if missing_roots:
        raise FileNotFoundError(f"Missing source layer directories: {missing_roots}")

    video_sets = {
        layer: {path.parent.name for path in root.glob("*/packed.pt")}
        for layer, root in layer_roots.items()
    }
    reference_videos = video_sets[layers[0]]
    for layer in layers[1:]:
        if video_sets[layer] != reference_videos:
            missing = sorted(reference_videos - video_sets[layer])
            extra = sorted(video_sets[layer] - reference_videos)
            raise RuntimeError(
                f"Layer {layer} video set differs from layer {layers[0]}: "
                f"missing={missing[:5]}, extra={extra[:5]}"
            )
    if not reference_videos:
        raise RuntimeError(f"No packed.pt files found under {source_root}")

    output_root.mkdir(parents=True, exist_ok=True)
    layer_label = "_".join(str(layer) for layer in layers)
    completed = 0
    skipped = 0
    for video_name in sorted(reference_videos):
        output_path = output_root / video_name / "packed.pt"
        if output_path.exists():
            existing = torch.load(output_path, map_location="cpu")
            if (
                existing.get("aggregation_protocol") != "s1_fixed_equal_mean_v1"
                or tuple(existing.get("source_layers", ())) != layers
            ):
                raise RuntimeError(f"Existing S1 output has incompatible metadata: {output_path}")
            skipped += 1
            continue

        packs = [
            torch.load(layer_roots[layer] / video_name / "packed.pt", map_location="cpu")
            for layer in layers
        ]
        reference = packs[0]
        reference_idx = reference["frame_idx"]
        reference_pooled = reference["pooled"]
        reference_prompt = reference.get("prompt")
        for layer, pack in zip(layers[1:], packs[1:]):
            if not torch.equal(reference_idx, pack["frame_idx"]):
                raise RuntimeError(f"{video_name}: frame indices differ at layer {layer}")
            if pack["pooled"].shape != reference_pooled.shape:
                raise RuntimeError(
                    f"{video_name}: pooled shape differs at layer {layer}: "
                    f"{tuple(pack['pooled'].shape)} != {tuple(reference_pooled.shape)}"
                )
            if pack["pooled"].dtype != reference_pooled.dtype:
                raise RuntimeError(
                    f"{video_name}: pooled dtype differs at layer {layer}: "
                    f"{pack['pooled'].dtype} != {reference_pooled.dtype}"
                )
            if pack.get("prompt") != reference_prompt:
                raise RuntimeError(f"{video_name}: prompt differs at layer {layer}")

        mean_pooled = torch.stack(
            [pack["pooled"].to(torch.float32) for pack in packs], dim=0
        ).mean(dim=0).to(reference_pooled.dtype)
        output_pack = dict(packs[-1])
        output_pack.update({
            "pooled": mean_pooled,
            "hidden_layer": None,
            "hidden_layer_ratio": None,
            "feature_type": (
                f"qwen2_5_vl_fixed_equal_mean_layers_{layer_label}_image_token_mean"
            ),
            "pooling_type": "fixed_equal_mean_of_per_layer_image_token_means",
            "aggregation_protocol": "s1_fixed_equal_mean_v1",
            "aggregation_compute_dtype": "float32",
            "aggregation_storage_dtype": str(reference_pooled.dtype),
            "source_layers": list(layers),
            "source_root": str(source_root.resolve()),
        })
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = output_path.with_suffix(".pt.tmp")
        torch.save(output_pack, temporary_path)
        os.replace(temporary_path, output_path)
        completed += 1
        if completed == 1 or completed % 25 == 0:
            print(
                f"[S1-MEAN] completed={completed} skipped={skipped} "
                f"video={video_name} shape={tuple(mean_pooled.shape)} "
                f"dtype={mean_pooled.dtype}"
            )

    print(
        f"[S1-MEAN-DONE] layers={layers} videos={len(reference_videos)} "
        f"written={completed} skipped={skipped} output={output_root.resolve()}"
    )


def run_selected_operation():
    operation = os.getenv("QWEN_OPERATION", "extract").strip().lower()
    if operation == "extract":
        main()
    elif operation == "classification_probe":
        classification_probe_main()
    elif operation == "boundary_probe":
        boundary_probe_main()
    elif operation == "probe_pipeline":
        probe_pipeline_main()
    elif operation == "paired_video_bootstrap":
        paired_video_bootstrap_main()
    elif operation == "probe_pdf_report":
        probe_pdf_report_main()
    elif operation == "average_hidden_layers":
        average_hidden_layers_main()
    else:
        raise ValueError(f"Unsupported QWEN_OPERATION={operation!r}")


if __name__ == "__main__":
    run_selected_operation()
