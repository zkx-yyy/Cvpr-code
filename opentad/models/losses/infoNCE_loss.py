import torch
import torch.nn as nn
import torch.nn.functional as F
# print("[DEBUG] import info_nce_loss from", __name__)

class InfoNCELoss(nn.Module):
    """
    CLIP-style InfoNCE loss with optional bidirectional training
    and automatic masking of [NULL] (zero-norm) samples.

    Args:
        temperature (float): Softmax temperature. Default 0.07.
        bidirectional (bool): If True, compute symmetric loss
            (caption→label + label→caption)/2.
        detach_label (bool): If True, stop gradient for label branch.
    Usage:
        loss_fn = InfoNCELoss(temperature=0.07, bidirectional=True)   # , detach_label=True
        loss = loss_fn(caption, label_gt)
        # caption, label_gt: [B, C]
    """

    def __init__(self, temperature=0.07, bidirectional=False):
        super().__init__()
        self.temperature = temperature
        self.bidirectional = bidirectional

    def forward(self, caption, label_gt, mask=None, lambda_bg=0.0):
        """
        Args:
            caption: [N, C]
            label_gt: [N, C]
        """
        if caption.numel() == 0 or label_gt.numel() == 0:
            return torch.tensor(0.0, device=caption.device, requires_grad=True)

        # 相似度矩阵
        logits = (caption @ label_gt.t()) / self.temperature
        logits = logits - logits.max(dim=-1, keepdim=True).values.detach()  # 数值稳定

        # 自然匹配索引
        targets = torch.arange(logits.size(0), device=logits.device)

        # Caption→Label
        loss_i2t = F.cross_entropy(logits, targets)

        if self.bidirectional:
            # Label→Caption
            loss_t2i = F.cross_entropy(logits.t(), targets)
            loss = 0.5 * (loss_i2t + loss_t2i)
        else:
            loss = loss_i2t

        return loss

try:
    from ..builder import LOSSES
    LOSSES.register_module()(InfoNCELoss)
except Exception:
    pass