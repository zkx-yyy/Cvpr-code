import torch
from ..utils.post_processing import load_predictions, save_predictions


class BaseDetector(torch.nn.Module):
    """Base class for detectors."""

    def __init__(self):
        super(BaseDetector, self).__init__()

    def forward(
        self,
        inputs,
        masks,
        metas,
        gt_segments=None,
        gt_labels=None,
        gt_labels_text=None,  # for txt
        return_loss=True,
        infer_cfg=None,
        post_cfg=None,
        **kwargs
    ):
        if return_loss:
            return self.forward_train(inputs, masks, metas, gt_segments=gt_segments, gt_labels=gt_labels, gt_labels_text=gt_labels_text, **kwargs)  # for txt :gt_labels_text=gt_labels_text,
        else:
            return self.forward_detection(inputs, masks, metas, infer_cfg, post_cfg, gt_segments=gt_segments, gt_labels=gt_labels, gt_labels_text=gt_labels_text, **kwargs)     # for txt : gt_segments=gt_segments, gt_labels=gt_labels, gt_labels_text=gt_labels_text,

    def forward_detection(self, inputs, masks, metas, infer_cfg, post_cfg, gt_segments, gt_labels, gt_labels_text, **kwargs):  # for txt : gt_segments, gt_labels, gt_labels_text,
        # step1: inference the model
        if infer_cfg.load_from_raw_predictions:  # easier and faster to tune the hyper parameter in postprocessing
            predictions = load_predictions(metas, infer_cfg)
        else:
            predictions = self.forward_test(inputs, masks, metas, infer_cfg, gt_segments=gt_segments, gt_labels=gt_labels, gt_labels_text=gt_labels_text,)   # for txt : gt_segments=gt_segments, gt_labels=gt_labels, gt_labels_text=gt_labels_text,

            if infer_cfg.save_raw_prediction:  # save the predictions to disk
                save_predictions(predictions, metas, infer_cfg.folder)

        # step2: detection post processing
        results = self.post_processing(predictions, metas, post_cfg, **kwargs)
        return results
