import copy
import os
import pickle
import random
import torch
import random
import pandas as pd
import numpy as np

from ..builder import PIPELINES
from torch.nn import functional as F

import os, json, time, torch

RT_LOG = "/nvme3/stu04/open-git/.rt_3return.log"
def _log_three_return(results, extra=None):
    wi  = torch.utils.data.get_worker_info()
    rec = {
        "ts": time.strftime("%F %T"),
        "pid": os.getpid(),
        "wid": (wi.id if wi else -1),
        "video_name": results.get("video_name", "<unk>"),
        "total_frames": int(results.get("total_frames", -1)),
        "snippet_stride": int(results.get("snippet_stride", -1)) if results.get("snippet_stride", None) is not None else None,
        "scale_factor": int(getattr(getattr(results, "_cfg", None), "scale_factor", -1))  # 可选：如果你塞在cfg里
    }
    if isinstance(results.get("gt_segments", None), (list, tuple)):
        rec["n_gt_segments"] = len(results["gt_segments"])
    elif getattr(results.get("gt_segments", None), "shape", None) is not None:
        rec["n_gt_segments"] = int(results["gt_segments"].shape[0])
    else:
        rec["n_gt_segments"] = None
    rec["n_gt_labels"] = (len(results.get("gt_labels", []))
                          if results.get("gt_labels", None) is not None else None)
    rec["n_gt_labels_text"] = (len(results.get("gt_labels_text", []))
                               if results.get("gt_labels_text", None) is not None else None)
    if extra:
        rec.update(extra)
    with open(RT_LOG, "a", buffering=1) as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

@PIPELINES.register_module()
class PrepareVideoInfo:
    def __init__(self, format="mp4", modality="RGB", prefix=""):
        self.format = format
        self.modality = modality
        self.prefix = prefix

    def __call__(self, results):
        results["modality"] = self.modality
        results["filename"] = os.path.join(
            results["data_path"],
            self.prefix + results["video_name"] + "." + self.format,
        )
        return results


@PIPELINES.register_module()
class LoadSnippetFrames:
    """Load the snippet frame, the output should follows the format:
    snippet_num x channel x clip_len x height x width
    """

    def __init__(
        self,
        clip_len,
        frame_interval=1,
        method="resize",
        trunc_len=None,
        trunc_thresh=None,
        crop_ratio=None,
    ):
        self.clip_len = clip_len
        self.frame_interval = frame_interval
        self.method = method  # resize or padding or sliding window
        # todo: support to  change FPS
        # random_trunc settings
        self.trunc_len = trunc_len
        self.trunc_thresh = trunc_thresh
        self.crop_ratio = crop_ratio

    def random_trunc(self, feats, trunc_len, gt_segments, gt_labels, offset=0, max_num_trials=200):
        feat_len = feats.shape[0]
        num_segs = gt_segments.shape[0]

        trunc_len = trunc_len
        if feat_len <= trunc_len:
            if self.crop_ratio == None:  # do nothing
                return feats, gt_segments, gt_labels
            else:  # randomly crop the seq by setting trunc_len to a value in [l, r]
                trunc_len = random.randint(
                    max(round(self.crop_ratio[0] * feat_len), 1),
                    min(round(self.crop_ratio[1] * feat_len), feat_len),
                )
                # corner case
                if feat_len == trunc_len:
                    return feats, gt_segments, gt_labels

        # try a few times till a valid truncation with at least one action
        for _ in range(max_num_trials):
            # sample a random truncation of the video feats
            st = random.randint(0, feat_len - trunc_len)
            ed = st + trunc_len
            window = np.array([st, ed], dtype=np.float32)

            # compute the intersection between the sampled window and all segments
            window = np.repeat(window[None, :], num_segs, axis=0)
            left = np.maximum(window[:, 0] - offset, gt_segments[:, 0])
            right = np.minimum(window[:, 1] + offset, gt_segments[:, 1])
            inter = np.clip(right - left, a_min=0, a_max=None)
            area_segs = np.abs(gt_segments[:, 1] - gt_segments[:, 0])
            inter_ratio = inter / area_segs

            # only select those segments over the thresh
            seg_idx = inter_ratio >= self.trunc_thresh

            # with at least one action
            if seg_idx.sum().item() > 0:
                break

        feats = feats[st:ed]
        gt_segments = np.stack((left[seg_idx], right[seg_idx]), axis=1)  # [N,2] in feature grids
        gt_segments = gt_segments - st  # shift the time stamps due to truncation
        gt_labels = gt_labels[seg_idx]  # [N]
        return feats, gt_segments, gt_labels

    def __call__(self, results):
        assert "total_frames" in results.keys(), "should have total_frames as a key"
        total_frames = results["total_frames"]
        fps = results["avg_fps"]

        if self.method == "resize":
            assert "resize_length" in results.keys(), "should have resize_length as a key"
            snippet_num = results["resize_length"]
            snippet_stride = total_frames / snippet_num
            snippet_center = np.arange(
                snippet_stride / 2 - 0.5,
                total_frames + snippet_stride / 2 - 0.5,
                snippet_stride,
            )
            masks = torch.ones(results["resize_length"]).bool()

            # don't forget to resize the ground truth segments
            if "gt_segments" in results.keys():
                # convert gt seconds to feature grid
                results["gt_segments"] = np.clip(results["gt_segments"] / results["duration"], 0.0, 1.0)
                results["gt_segments"] *= results["resize_length"]

        elif self.method == "random_trunc":
            snippet_num = self.trunc_len
            snippet_center = np.arange(0, total_frames, results["snippet_stride"])

            # trunc the snippet_center
            snippet_center, gt_segments, gt_labels = self.random_trunc(
                snippet_center,
                trunc_len=snippet_num,
                gt_segments=results["gt_segments"],
                gt_labels=results["gt_labels"],
            )

            # update the gt_segments
            results["gt_segments"] = gt_segments
            results["gt_labels"] = gt_labels

            # pad the snippet_center
            if len(snippet_center) < snippet_num:
                valid_len = len(snippet_center)
                snippet_center = np.pad(snippet_center, (0, snippet_num - valid_len), mode="edge")
                masks = torch.cat([torch.ones(valid_len), torch.zeros(snippet_num - valid_len)]).bool()
            else:
                masks = torch.ones(snippet_num).bool()

        elif self.method == "sliding_window":
            snippet_num = results["window_size"]
            snippet_center = np.arange(0, total_frames, results["snippet_stride"])

            start_idx = min(results["feature_start_idx"], len(snippet_center))
            end_idx = min((results["feature_end_idx"] + 1), len(snippet_center))

            snippet_center = snippet_center[start_idx:end_idx]

            if len(snippet_center) < snippet_num:
                valid_len = len(snippet_center)
                snippet_center = np.pad(snippet_center, (0, snippet_num - valid_len), mode="edge")
                masks = torch.cat([torch.ones(valid_len), torch.zeros(snippet_num - valid_len)]).bool()
            else:
                masks = torch.ones(snippet_num).bool()
        elif self.method == "padding":
            raise NotImplementedError

        # extend snippet center to a clip
        clip_idxs = np.arange(-(self.clip_len // 2), self.clip_len // 2)
        frame_idxs = snippet_center[:, None] + self.frame_interval * clip_idxs[None, :]  # [snippet_num, clip_len]

        # truncate to [0, total_frames-1], and round to int
        frame_idxs = np.clip(frame_idxs, 0, total_frames - 1).round()

        assert frame_idxs.shape[0] == snippet_num, "snippet center number should be equal to snippet number"
        assert frame_idxs.shape[1] == self.clip_len, "snippet length should be equal to clip length"

        results["frame_inds"] = frame_idxs.astype(int)
        results["num_clips"] = snippet_num
        results["clip_len"] = self.clip_len
        results["masks"] = masks
        return results


@PIPELINES.register_module()
class LoadFrames:
    def __init__(
        self,
        num_clips=1,
        scale_factor=1,
        method="resize",
        trunc_len=None,
        trunc_thresh=None,
        crop_ratio=None,
    ):
        self.num_clips = num_clips
        self.scale_factor = scale_factor  # multiply by the frame number, if backbone has downsampling
        self.method = method  # resize or padding or random_trunc or sliding_window
        # random_trunc settings
        self.trunc_len = trunc_len
        self.trunc_thresh = trunc_thresh
        self.crop_ratio = crop_ratio

    def random_trunc(self, feats, trunc_len, gt_segments, gt_labels, gt_labels_text, offset=0, max_num_trials=200):  # for txt
        feat_len = feats.shape[0]
        num_segs = gt_segments.shape[0]

        trunc_len = trunc_len
        if feat_len <= trunc_len:
            if self.crop_ratio == None:  # do nothing
                return feats, gt_segments, gt_labels
            else:  # randomly crop the seq by setting trunc_len to a value in [l, r]
                trunc_len = random.randint(
                    max(round(self.crop_ratio[0] * feat_len), 1),
                    min(round(self.crop_ratio[1] * feat_len), feat_len),
                )
                # corner case
                if feat_len == trunc_len:
                    return feats, gt_segments, gt_labels, gt_labels_text   # for txt

        # try a few times till a valid truncation with at least one action
        for _ in range(max_num_trials):
            # sample a random truncation of the video feats
            st = random.randint(0, feat_len - trunc_len)
            ed = st + trunc_len
            window = np.array([st, ed], dtype=np.float32)

            # compute the intersection between the sampled window and all segments
            window = np.repeat(window[None, :], num_segs, axis=0)
            left = np.maximum(window[:, 0] - offset, gt_segments[:, 0])
            right = np.minimum(window[:, 1] + offset, gt_segments[:, 1])
            inter = np.clip(right - left, a_min=0, a_max=None)
            area_segs = np.abs(gt_segments[:, 1] - gt_segments[:, 0])
            inter_ratio = inter / area_segs

            # only select those segments over the thresh
            seg_idx = inter_ratio >= self.trunc_thresh

            # with at least one action
            if seg_idx.sum().item() > 0:
                break

        feats = feats[st:ed]
        gt_segments = np.stack((left[seg_idx], right[seg_idx]), axis=1)  # [N,2] in feature grids
        gt_segments = gt_segments - st  # shift the time stamps due to truncation
        gt_labels = gt_labels[seg_idx]  # [N]
        gt_labels_text = [text for text, m in zip(gt_labels_text, seg_idx) if m]  # [N], for txt
        return feats, gt_segments, gt_labels, gt_labels_text

    def sliding_get_gt(self, results, scale_factor, frame_idxs, frame_num, window_size, max_num_trials=200):
        """
        for txt，eval也需要label过tokenizer
        """

        snippet_stride = int(results["snippet_stride"])
        start_snip = int(results["feature_start_idx"])
        end_snip = int(results["feature_end_idx"]) + 1  # 半开区间 [start_snip, end_snip)

        # 当前窗口idx映射到帧数
        start_f = start_snip * snippet_stride
        end_f = end_snip * snippet_stride

        # 取 GT（全局 snippet → 帧域）
        gt_segments = results["gt_segments"]
        if hasattr(gt_segments, "detach"):
            gt_segments = gt_segments.detach().cpu().numpy()
        else:
            gt_segments = np.asarray(gt_segments, dtype=np.float32)
        segs_f = gt_segments * float(snippet_stride)  # [N,2] in frame

        # 帧域求交
        if segs_f.size == 0:
            # 本视频本就无 GT
            empty_segments = np.zeros((0, 2), dtype=np.float32)
            empty_labels = results["gt_labels"][:0] if hasattr(results["gt_labels"], "__getitem__") else np.asarray([],
                                                                                                                    dtype=np.int64)
            return empty_segments, empty_labels, []

        l = np.maximum(segs_f[:, 0], start_f)
        r = np.minimum(segs_f[:, 1], end_f)
        keep = (r > l)  # 正长度交集

        if not np.any(keep):
            # 允许空窗：返回空
            empty_segments = np.zeros((0, 2), dtype=np.float32)
            empty_labels = results["gt_labels"][:0] if hasattr(results["gt_labels"], "__getitem__") else np.asarray([], dtype=np.int64)
            return empty_segments, empty_labels, []

        # 裁到局部帧域 → 转回局部 snippet 坐标
        l_loc_f = l[keep] - start_f
        r_loc_f = r[keep] - start_f
        segs_loc_snip = (np.stack([l_loc_f, r_loc_f], axis=1) / float(snippet_stride)).astype(np.float32)

        # 同步筛 labels / texts
        gt_labels = results["gt_labels"]
        gt_texts = results["gt_labels_text"]

        if isinstance(gt_labels, np.ndarray):
            labels_loc = gt_labels[keep]
        else:
            labels_loc = np.asarray(gt_labels)[keep]

        if isinstance(gt_texts, list):
            idxs = np.nonzero(keep)[0].tolist()
            texts_loc = [gt_texts[i] for i in idxs]
        else:
            texts_loc = np.asarray(gt_texts)[keep].tolist()

        return segs_loc_snip, labels_loc, texts_loc

    def __call__(self, results):
        assert "total_frames" in results.keys(), "should have total_frames as a key"
        total_frames = results["total_frames"]
        fps = results["avg_fps"]

        if self.method == "resize":
            assert "resize_length" in results.keys(), "should have resize_length as a key"
            frame_num = results["resize_length"] * self.scale_factor
            frame_stride = total_frames / frame_num
            frame_idxs = np.arange(
                frame_stride / 2 - 0.5,
                total_frames + frame_stride / 2 - 0.5,
                frame_stride,
            )
            masks = torch.ones(results["resize_length"]).bool()  # should not multiply by scale_factor

            # don't forget to resize the ground truth segments
            if "gt_segments" in results.keys():
                # convert gt seconds to feature grid
                results["gt_segments"] = np.clip(results["gt_segments"] / results["duration"], 0.0, 1.0)
                results["gt_segments"] *= results["resize_length"]

        elif self.method == "random_trunc":
            assert results["snippet_stride"] >= self.scale_factor, "snippet_stride should be larger than scale_factor"
            assert (
                results["snippet_stride"] % self.scale_factor == 0
            ), "snippet_stride should be divisible by scale_factor"

            frame_num = self.trunc_len * self.scale_factor
            frame_stride = results["snippet_stride"] // self.scale_factor
            frame_idxs = np.arange(0, total_frames, frame_stride)

            # debug
            # raise RuntimeError("[SENTINEL] reached before random_trunc "
            #                    f"video={results.get('video_name', '<unk>')}")
            # try:
            #     # trunc the frame_idxs
            #     frame_idxs, gt_segments, gt_labels, gt_labels_text = self.random_trunc(
            #         frame_idxs,
            #         trunc_len=frame_num,
            #         gt_segments=results["gt_segments"] * self.scale_factor,  # gt to frame level
            #         gt_labels=results["gt_labels"],
            #         gt_labels_text=results["gt_labels_text"],  # for txt
            #     )
            # except ValueError as e:
            #     vn = results.get("video_name", "<unknown>")
            #     ss = results.get("snippet_stride", None)
            #     print(
            #         "[random_trunc] ValueError @ video:",
            #         vn,
            #         "| total_frames:", total_frames,
            #         "| snippet_stride:", ss,
            #         "| scale_factor:", self.scale_factor,
            #         "| frame_num:", frame_num,
            #         "| len(frame_idxs):", len(frame_idxs),
            #         "| segs shape:", getattr(results.get("gt_segments", None), "shape", None),
            #         "| labels len:",
            #         (len(results.get("gt_labels", [])) if results.get("gt_labels", None) is not None else None),
            #         "| texts len:", (len(results.get("gt_labels_text", [])) if results.get("gt_labels_text",
            #                                                                                None) is not None else None),
            #         flush=True,
            #     )
            #     raise   # debug

            # trunc the frame_idxs
            frame_idxs, gt_segments, gt_labels, gt_labels_text = self.random_trunc(    # for txt
                frame_idxs,
                trunc_len=frame_num,
                gt_segments=results["gt_segments"] * self.scale_factor,  # gt segment should be mapped to frame level
                gt_labels=results["gt_labels"],
                gt_labels_text=results["gt_labels_text"],  # for txt
            )

            # ---- call random_trunc, robust unpack + logging ----
            # out = self.random_trunc(
            #     frame_idxs,
            #     trunc_len=frame_num,
            #     gt_segments=results["gt_segments"] * self.scale_factor,
            #     gt_labels=results["gt_labels"],
            #     gt_labels_text=results.get("gt_labels_text", []),
            # )
            #
            # if not isinstance(out, tuple):
            #     # 极端兜底：写日志再报错
            #     _log_three_return(results, {"error": f"non-tuple return: {type(out)}"})
            #     raise RuntimeError(
            #         f"random_trunc returned non-tuple: {type(out)} for video={results.get('video_name')}")
            #
            # n = len(out)
            # if n == 4:
            #     frame_idxs, gt_segments, gt_labels, gt_labels_text = out
            # elif n == 3:
            #     # 命中“仅3返回”，写入日志，附上下文
            #     _log_three_return(results, {"return_items": 3})
            #     # # 决策1：不中断，先让训练继续（空文本）
            #     # frame_idxs, gt_segments, gt_labels = out
            #     # gt_labels_text = []
            #     # 决策2：如果你想当场停掉，改成：
            #     raise RuntimeError(f"[3-return] video={results.get('video_name')}")
            # else:
            #     _log_three_return(results, {"return_items": n})
            #     raise RuntimeError(
            #         f"random_trunc returned {n} values (expected 3 or 4) | video={results.get('video_name')}")  # debug结束

            results["gt_segments"] = gt_segments / self.scale_factor  # convert back to original scale
            results["gt_labels"] = gt_labels
            results["gt_labels_text"] = gt_labels_text  # for txt

            # pad the frame_idxs
            if len(frame_idxs) < frame_num:
                valid_len = len(frame_idxs) // self.scale_factor
                frame_idxs = np.pad(frame_idxs, (0, frame_num - len(frame_idxs)), mode="edge")
                masks = torch.cat([torch.ones(valid_len), torch.zeros(self.trunc_len - valid_len)]).bool()
            else:
                masks = torch.ones(self.trunc_len).bool()

        elif self.method == "sliding_window":
            assert results["snippet_stride"] >= self.scale_factor, "snippet_stride should be larger than scale_factor"
            assert (
                results["snippet_stride"] % self.scale_factor == 0
            ), "snippet_stride should be divisible by scale_factor"

            window_size = results["window_size"]
            frame_num = window_size * self.scale_factor
            frame_stride = results["snippet_stride"] // self.scale_factor
            frame_idxs = np.arange(0, total_frames, frame_stride)

            start_idx = min(results["feature_start_idx"] * self.scale_factor, len(frame_idxs))
            end_idx = min((results["feature_end_idx"] + 1) * self.scale_factor, len(frame_idxs))

            frame_idxs = frame_idxs[start_idx:end_idx]

            if len(frame_idxs) < frame_num:
                valid_len = len(frame_idxs) // self.scale_factor
                frame_idxs = np.pad(frame_idxs, (0, frame_num - len(frame_idxs)), mode="edge")
                masks = torch.cat([torch.ones(valid_len), torch.zeros(window_size - valid_len)]).bool()
            else:
                masks = torch.ones(window_size).bool()

            # for txt, eval也需要label经过tokenizer
            gt_segments, gt_labels, gt_labels_text = self.sliding_get_gt(
                results=results,
                scale_factor=self.scale_factor,
                frame_idxs=frame_idxs,
                frame_num=frame_num,
                window_size=window_size,
            )
            results["gt_segments"] = gt_segments
            results["gt_labels"] = gt_labels
            results["gt_labels_text"] = gt_labels_text

        elif self.method == "padding":
            raise NotImplementedError

        # truncate to [0, total_frames-1], and round to int
        frame_idxs = np.clip(frame_idxs, 0, total_frames - 1).round()

        assert frame_idxs.shape[0] == frame_num, "snippet center number should be equal to snippet number"

        results["frame_inds"] = frame_idxs.astype(int)
        results["num_clips"] = self.num_clips
        results["clip_len"] = frame_num // self.num_clips
        results["masks"] = masks
        return results


@PIPELINES.register_module()
class Interpolate:
    def __init__(self, keys, size=128, mode="linear"):
        self.keys = keys
        self.size = size
        self.mode = mode

    def __call__(self, results):
        for key in self.keys:
            if results[key].shape[2:] != self.size:
                results[key] = F.interpolate(
                    results[key],
                    size=self.size,
                    mode=self.mode,
                    align_corners=False,
                )
        return results
