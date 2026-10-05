import numpy as np
from .base import SlidingWindowDataset, PaddingDataset, filter_same_annotation, filter_same_annotation_text
from .builder import DATASETS


@DATASETS.register_module()
class ThumosSlidingDataset(SlidingWindowDataset):
    def get_gt(self, video_info, thresh=0.0):
        gt_segment = []
        gt_label = []
        gt_label_text = []  # for txt
        for anno in video_info["annotations"]:
            if anno["label"] == "Ambiguous":
                continue
            gt_start = int(anno["segment"][0] / video_info["duration"] * video_info["frame"])
            gt_end = int(anno["segment"][1] / video_info["duration"] * video_info["frame"])

            if (not self.filter_gt) or (gt_end - gt_start > thresh):
                gt_segment.append([gt_start, gt_end])
                gt_label.append(self.class_map.index(anno["label"]))
                gt_label_text.append(anno["label"])  # for txt

        if len(gt_segment) == 0:  # have no valid gt
            return None
        else:
            annotation = dict(
                gt_segments=np.array(gt_segment, dtype=np.float32),
                gt_labels=np.array(gt_label, dtype=np.int32),
                gt_labels_text=gt_label_text,  # for txt
            )
            return filter_same_annotation_text(annotation)      # origin:return filter_same_annotation(annotation), for txt:filter_same_annotation_text(annotation)

    def __getitem__(self, index):
        video_name, video_info, video_anno, window_snippet_centers = self.data_list[index]

        if video_anno != {}:
            # frame divided by snippet stride inside current window
            # this is only valid gt inside this window
            video_anno["gt_segments"] = video_anno["gt_segments"] - window_snippet_centers[0] - self.offset_frames
            video_anno["gt_segments"] = video_anno["gt_segments"] / self.snippet_stride

        results = self.pipeline(
            dict(
                video_name=video_name,
                data_path=self.data_path,
                window_size=self.window_size,
                # trunc window setting
                feature_start_idx=int(window_snippet_centers[0] / self.snippet_stride),
                feature_end_idx=int(window_snippet_centers[-1] / self.snippet_stride),
                sample_stride=self.sample_stride,
                # sliding post process setting
                fps=video_info["frame"] / video_info["duration"],
                snippet_stride=self.snippet_stride,
                window_start_frame=window_snippet_centers[0],
                duration=video_info["duration"],
                offset_frames=self.offset_frames,
                # training setting
                **video_anno,
            )
        )

        # for txt debug
        # if "gt_segments" in results and len(results["gt_segments"]) == 0:
        #     print(f"[WARN] Empty GT in __getitem__ at index={index}")
        #     print(f"  video_name = {video_name}")
        #     print(f"  window_start_frame = {window_snippet_centers[0]}")
        #     print(f"  window_end_frame = {window_snippet_centers[-1]}")
        #     print(f"  duration = {video_info['duration']}")
        #     print(f"  frame = {video_info['frame']}")
        #     print(f"  original annotations: {video_info['annotations']}")
        #     print("-" * 80)

        return results


@DATASETS.register_module()
class ThumosPaddingDataset(PaddingDataset):  # train
    def get_gt(self, video_info, thresh=0.0):
        gt_segment = []
        gt_label = []
        gt_label_text = []    # for txt
        for anno in video_info["annotations"]:
            if anno["label"] == "Ambiguous":
                continue
            gt_start = int(anno["segment"][0] / video_info["duration"] * video_info["frame"])
            gt_end = int(anno["segment"][1] / video_info["duration"] * video_info["frame"])

            if (not self.filter_gt) or (gt_end - gt_start > thresh):
                gt_segment.append([gt_start, gt_end])
                gt_label.append(self.class_map.index(anno["label"]))
                gt_label_text.append(anno["label"])   # for txt

        if len(gt_segment) == 0:  # have no valid gt
            return None
        else:
            annotation = dict(
                gt_segments=np.array(gt_segment, dtype=np.float32),
                gt_labels=np.array(gt_label, dtype=np.int32),
                gt_labels_text=gt_label_text,     # for txt
            )
            return filter_same_annotation_text(annotation)    # origin:return filter_same_annotation(annotation), for txt

    def __getitem__(self, index):
        video_name, video_info, video_anno = self.data_list[index]

        if video_anno != {}:
            video_anno["gt_segments"] = video_anno["gt_segments"] - self.offset_frames
            video_anno["gt_segments"] = video_anno["gt_segments"] / self.snippet_stride

            # # 1) 进入 pipeline 前先看一眼 keys,看看text还在不在
            # pre = dict(
            #     video_name=video_name,
            #     data_path=self.data_path,
            #     sample_stride=self.sample_stride,
            #     snippet_stride=self.snippet_stride,
            #     fps=video_info["frame"] / video_info["duration"],
            #     duration=video_info["duration"],
            #     offset_frames=self.offset_frames,
            #     **video_anno,
            # )
            # if index == 0:  # 只打印第一个样本，避免刷屏
            #     print('[DEBUG] pre-pipeline keys:', sorted(list(pre.keys())))
            #     print('[DEBUG] pre has gt_labels_text:', 'gt_labels_text' in pre)
            # # 逐个 transform 看看有没有丢失 gt_labels_text
            # data = pre
            # for i, t in enumerate(getattr(self.pipeline, 'transforms', [])):
            #     data = t(data)
            #     if isinstance(data, dict):
            #         lost = 'gt_labels_text' not in data
            #         if index == 0:
            #             print(f'[DEBUG] [{i:02d}] {t.__class__.__name__:<24} -> dict, lost_text={lost}')
            #     else:
            #         # Pack* 可能返回 DataSample
            #         meta = getattr(data, 'metainfo', {})
            #         has = isinstance(meta, dict) and ('gt_labels_text' in meta)
            #         if index == 0:
            #             print(
            #                 f'[DEBUG] [{i:02d}] {t.__class__.__name__:<24} -> {type(data).__name__}, meta_has_text={has}')
            #
            # return data

        results = self.pipeline(
            dict(
                video_name=video_name,
                data_path=self.data_path,
                sample_stride=self.sample_stride,
                snippet_stride=self.snippet_stride,
                fps=video_info["frame"] / video_info["duration"],
                duration=video_info["duration"],
                offset_frames=self.offset_frames,
                **video_anno,
            )
        )
        return results
