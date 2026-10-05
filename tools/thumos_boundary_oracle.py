import argparse
import copy
import json
import os
import sys
from collections import defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment


sys.dont_write_bytecode = True
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from opentad.evaluations import build_evaluator
from opentad.evaluations.builder import remove_duplicate_annotations


def parse_args():
    parser = argparse.ArgumentParser(
        description="Measure local temporal-boundary Oracle room on THUMOS14 predictions."
    )
    parser.add_argument("--ground-truth", required=True, help="THUMOS14 annotation JSON")
    parser.add_argument("--prediction", required=True, help="OpenTAD result_detection.json")
    parser.add_argument("--output-dir", required=True, help="Directory for report and edited predictions")
    parser.add_argument("--subset", default="validation")
    parser.add_argument("--feature-stride", type=float, default=4.0, help="Frames per feature step")
    parser.add_argument("--radii", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--tiou-thresholds", type=float, nargs="+", default=[0.3, 0.4, 0.5, 0.6, 0.7])
    parser.add_argument("--crossing-thresholds", type=float, nargs="+", default=[0.6, 0.7])
    parser.add_argument("--eval-workers", type=int, default=4)
    parser.add_argument(
        "--oracle-chunk-size",
        type=int,
        default=256,
        help="Prediction rows per vectorized local-search chunk",
    )
    parser.add_argument(
        "--save-edited-predictions",
        action="store_true",
        help="Also save one prediction JSON for every mode/radius pair",
    )
    return parser.parse_args()


def segment_iou(segment, candidates):
    candidates = np.asarray(candidates, dtype=np.float64)
    if candidates.size == 0:
        return np.empty(0, dtype=np.float64)
    left = np.maximum(float(segment[0]), candidates[:, 0])
    right = np.minimum(float(segment[1]), candidates[:, 1])
    intersection = np.maximum(right - left, 0.0)
    union = (
        float(segment[1])
        - float(segment[0])
        + candidates[:, 1]
        - candidates[:, 0]
        - intersection
    )
    return intersection / np.maximum(union, 1e-8)


def load_ground_truth(path, subset):
    with open(path, "r") as f:
        data = json.load(f)

    ground_truth = defaultdict(list)
    metadata = {}
    for video_id, video in data["database"].items():
        if video["subset"] != subset:
            continue
        duration = float(video["duration"])
        frame_count = float(video["frame"])
        if duration <= 0 or frame_count <= 0:
            raise ValueError(f"Invalid duration/frame metadata for {video_id}")
        metadata[video_id] = {
            "duration": duration,
            "frame": frame_count,
            "fps": frame_count / duration,
        }
        for annotation in remove_duplicate_annotations(video["annotations"]):
            if annotation["label"] == "Ambiguous":
                continue
            ground_truth[video_id].append(
                {
                    "segment": np.asarray(annotation["segment"], dtype=np.float64),
                    "label": annotation["label"],
                }
            )
    return ground_truth, metadata


def load_predictions(path):
    with open(path, "r") as f:
        data = json.load(f)
    if "results" not in data:
        raise ValueError("Prediction JSON must contain a 'results' field")
    return data


def evaluate(prediction, ground_truth_path, subset, thresholds, workers):
    evaluator = build_evaluator(
        dict(
            type="mAP",
            ground_truth_filename=ground_truth_path,
            prediction_filename=prediction,
            subset=subset,
            tiou_thresholds=thresholds,
            thread=workers,
        )
    )
    metrics = evaluator.evaluate()
    return {key: float(value) for key, value in metrics.items()}


def prediction_arrays(predictions):
    arrays = {}
    for video_id, items in predictions["results"].items():
        arrays[video_id] = {
            "segments": np.asarray([item["segment"] for item in items], dtype=np.float64).reshape(-1, 2),
            "labels": np.asarray([item["label"] for item in items], dtype=object),
            "scores": np.asarray([float(item["score"]) for item in items], dtype=np.float64),
        }
    return arrays


def proposal_recall(ground_truth, prediction_index, thresholds, class_aware):
    best_ious = []
    for video_id, gt_items in ground_truth.items():
        pred = prediction_index.get(video_id)
        for gt in gt_items:
            if pred is None or len(pred["segments"]) == 0:
                best_ious.append(0.0)
                continue
            mask = np.ones(len(pred["segments"]), dtype=bool)
            if class_aware:
                mask = pred["labels"] == gt["label"]
            candidates = pred["segments"][mask]
            ious = segment_iou(gt["segment"], candidates)
            best_ious.append(float(ious.max()) if len(ious) else 0.0)

    best_ious = np.asarray(best_ious, dtype=np.float64)
    return {
        "num_ground_truth": int(len(best_ious)),
        "mean_best_iou": float(best_ious.mean()) if len(best_ious) else 0.0,
        "recall": {str(t): float(np.mean(best_ious >= t)) for t in thresholds},
    }


def action_grid(radius, mode):
    values = range(-radius, radius + 1)
    if mode == "start":
        actions = [(delta, 0) for delta in values]
    elif mode == "end":
        actions = [(0, delta) for delta in values]
    elif mode == "both":
        actions = [(left, right) for left in values for right in values]
    else:
        raise ValueError(f"Unknown Oracle mode: {mode}")
    return sorted(actions, key=lambda x: (abs(x[0]) + abs(x[1]), abs(x[0]), abs(x[1]), x[0], x[1]))


def best_local_edits(pred_segments, gt_segments, step_seconds, duration, actions, chunk_size):
    pred_segments = np.asarray(pred_segments, dtype=np.float64)
    gt_segments = np.asarray(gt_segments, dtype=np.float64)
    actions = np.asarray(actions, dtype=np.float64)
    num_predictions = len(pred_segments)
    num_ground_truth = len(gt_segments)
    best_ious = np.empty((num_predictions, num_ground_truth), dtype=np.float64)
    best_starts = np.empty_like(best_ious)
    best_ends = np.empty_like(best_ious)

    gt_starts = gt_segments[:, 0][None, None, :]
    gt_ends = gt_segments[:, 1][None, None, :]
    gt_lengths = gt_ends - gt_starts
    start_deltas = actions[:, 0][None, :]
    end_deltas = actions[:, 1][None, :]

    for chunk_start in range(0, num_predictions, chunk_size):
        chunk_end = min(chunk_start + chunk_size, num_predictions)
        chunk = pred_segments[chunk_start:chunk_end]
        starts = np.clip(chunk[:, :1] + start_deltas * step_seconds, 0.0, duration)
        ends = np.clip(chunk[:, 1:] + end_deltas * step_seconds, 0.0, duration)
        valid = ends > starts

        intersection = np.maximum(
            np.minimum(ends[:, :, None], gt_ends) - np.maximum(starts[:, :, None], gt_starts),
            0.0,
        )
        union = (ends - starts)[:, :, None] + gt_lengths - intersection
        ious = intersection / np.maximum(union, 1e-8)
        ious = np.where(valid[:, :, None], ious, -1.0)

        best_action_indices = np.argmax(ious, axis=1)
        row_indices = np.arange(len(chunk))[:, None]
        best_ious[chunk_start:chunk_end] = np.take_along_axis(
            ious, best_action_indices[:, None, :], axis=1
        )[:, 0, :]
        best_starts[chunk_start:chunk_end] = starts[row_indices, best_action_indices]
        best_ends[chunk_start:chunk_end] = ends[row_indices, best_action_indices]

    return best_ious, best_starts, best_ends


def build_local_oracle(
    prediction,
    ground_truth,
    metadata,
    feature_stride,
    radius,
    mode,
    crossing_thresholds,
    chunk_size,
):
    edited = copy.deepcopy(prediction)
    actions = action_grid(radius, mode)
    stats = {
        "matched_pairs": 0,
        "improved_pairs": 0,
        "unchanged_pairs": 0,
        "mean_iou_before": 0.0,
        "mean_iou_after": 0.0,
        "mean_iou_gain": 0.0,
        "crossings": {str(t): 0 for t in crossing_thresholds},
        "mean_abs_start_move_steps": 0.0,
        "mean_abs_end_move_steps": 0.0,
    }
    before_values = []
    after_values = []
    start_moves = []
    end_moves = []

    for video_id, gt_items in ground_truth.items():
        pred_items = edited["results"].get(video_id, [])
        if not pred_items or not gt_items:
            continue
        step_seconds = feature_stride / metadata[video_id]["fps"]
        duration = metadata[video_id]["duration"]

        gt_by_label = defaultdict(list)
        pred_by_label = defaultdict(list)
        for gt_index, gt in enumerate(gt_items):
            gt_by_label[gt["label"]].append(gt_index)
        for pred_index, pred in enumerate(pred_items):
            pred_by_label[pred["label"]].append(pred_index)

        for label, label_gt_indices in gt_by_label.items():
            label_pred_indices = pred_by_label.get(label, [])
            if not label_pred_indices:
                continue

            label_pred_segments = np.asarray(
                [pred_items[index]["segment"] for index in label_pred_indices], dtype=np.float64
            )
            label_gt_segments = np.asarray(
                [gt_items[index]["segment"] for index in label_gt_indices], dtype=np.float64
            )
            quality, best_starts, best_ends = best_local_edits(
                label_pred_segments,
                label_gt_segments,
                step_seconds,
                duration,
                actions,
                chunk_size,
            )

            # Maximize total reachable IoU with one prediction assigned to at most one GT.
            # A tiny score term only breaks exact IoU ties in favour of higher-confidence detections.
            scores = np.asarray([float(pred_items[i]["score"]) for i in label_pred_indices])
            if len(scores) and scores.max() > scores.min():
                score_tie_break = (scores - scores.min()) / (scores.max() - scores.min())
                quality_for_assignment = quality + score_tie_break[:, None] * 1e-9
            else:
                quality_for_assignment = quality
            rows, cols = linear_sum_assignment(quality_for_assignment, maximize=True)

            for row, col in zip(rows, cols):
                pred_index = label_pred_indices[int(row)]
                gt_index = label_gt_indices[int(col)]
                pred_segment = np.asarray(pred_items[pred_index]["segment"], dtype=np.float64)
                gt_segment = gt_items[gt_index]["segment"]
                current_iou = float(segment_iou(gt_segment, pred_segment[None, :])[0])
                candidate_segment = np.asarray(
                    [best_starts[int(row), int(col)], best_ends[int(row), int(col)]],
                    dtype=np.float64,
                )
                candidate_iou = float(quality[int(row), int(col)])

                # Oracle never accepts an edit that reduces IoU for its assigned pair.
                if candidate_iou > current_iou + 1e-12:
                    pred_items[pred_index]["segment"] = [float(candidate_segment[0]), float(candidate_segment[1])]
                    stats["improved_pairs"] += 1
                else:
                    candidate_segment = pred_segment
                    candidate_iou = current_iou
                    stats["unchanged_pairs"] += 1

                stats["matched_pairs"] += 1
                before_values.append(current_iou)
                after_values.append(candidate_iou)
                start_moves.append(abs(candidate_segment[0] - pred_segment[0]) / step_seconds)
                end_moves.append(abs(candidate_segment[1] - pred_segment[1]) / step_seconds)
                for threshold in crossing_thresholds:
                    if current_iou < threshold <= candidate_iou:
                        stats["crossings"][str(threshold)] += 1

    if before_values:
        stats["mean_iou_before"] = float(np.mean(before_values))
        stats["mean_iou_after"] = float(np.mean(after_values))
        stats["mean_iou_gain"] = float(np.mean(np.asarray(after_values) - np.asarray(before_values)))
        stats["mean_abs_start_move_steps"] = float(np.mean(start_moves))
        stats["mean_abs_end_move_steps"] = float(np.mean(end_moves))
    return edited, stats


def main():
    args = parse_args()
    if any(radius < 0 for radius in args.radii):
        raise ValueError("Oracle radii must be non-negative")
    if args.oracle_chunk_size <= 0:
        raise ValueError("Oracle chunk size must be positive")
    os.makedirs(args.output_dir, exist_ok=True)

    ground_truth, metadata = load_ground_truth(args.ground_truth, args.subset)
    prediction = load_predictions(args.prediction)
    prediction_index = prediction_arrays(prediction)

    report = {
        "inputs": {
            "ground_truth": os.path.abspath(args.ground_truth),
            "prediction": os.path.abspath(args.prediction),
            "subset": args.subset,
            "feature_stride_frames": args.feature_stride,
            "radii_feature_steps": args.radii,
        },
        "definition": {
            "step_seconds": "feature_stride_frames * video_duration / video_frame_count",
            "matching": "per-video, per-class maximum-total-reachable-IoU one-to-one assignment",
            "unmatched_predictions": "kept unchanged",
            "scores_and_labels": "kept unchanged",
            "nms": "not rerun; Oracle edits the saved post-NMS detections",
        },
        "baseline": evaluate(
            prediction,
            args.ground_truth,
            args.subset,
            args.tiou_thresholds,
            args.eval_workers,
        ),
        "proposal_recall": {
            "class_agnostic": proposal_recall(
                ground_truth, prediction_index, args.tiou_thresholds, class_aware=False
            ),
            "class_aware": proposal_recall(
                ground_truth, prediction_index, args.tiou_thresholds, class_aware=True
            ),
        },
        "local_oracle": {},
    }

    for mode in ("start", "end", "both"):
        report["local_oracle"][mode] = {}
        for radius in args.radii:
            edited, stats = build_local_oracle(
                prediction,
                ground_truth,
                metadata,
                args.feature_stride,
                radius,
                mode,
                args.crossing_thresholds,
                args.oracle_chunk_size,
            )
            metrics = evaluate(
                edited,
                args.ground_truth,
                args.subset,
                args.tiou_thresholds,
                args.eval_workers,
            )
            report["local_oracle"][mode][str(radius)] = {
                "metrics": metrics,
                "diagnostics": stats,
            }
            if args.save_edited_predictions:
                output_path = os.path.join(args.output_dir, f"oracle_{mode}_r{radius}.json")
                with open(output_path, "w") as f:
                    json.dump(edited, f)
            print(
                f"{mode:>5} r={radius:<2d} "
                f"average_mAP={metrics['average_mAP'] * 100:.2f}% "
                f"mAP@0.6={metrics.get('mAP@0.6', float('nan')) * 100:.2f}% "
                f"mAP@0.7={metrics.get('mAP@0.7', float('nan')) * 100:.2f}%"
            )

    report_path = os.path.join(args.output_dir, "oracle_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved report to {report_path}")


if __name__ == "__main__":
    main()
