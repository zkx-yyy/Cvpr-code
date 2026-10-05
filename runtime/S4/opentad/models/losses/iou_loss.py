import torch
import torch.nn as nn
from ..builder import LOSSES
from ..utils.iou_tools import compute_diou_torch, compute_giou_torch
import torch.nn.functional as F   #


@LOSSES.register_module()
class DIOULoss(nn.Module):
    def __init__(self):
        super(DIOULoss, self).__init__()

    def forward(
        self,
        input_bboxes: torch.Tensor,
        target_bboxes: torch.Tensor,
        reduction: str = "none",
        eps: float = 1e-8,
    ) -> torch.Tensor:
        loss = 1 - torch.diag(compute_diou_torch(target_bboxes, input_bboxes))

        if reduction == "mean":
            loss = loss.mean() if loss.numel() > 0 else 0.0 * loss.sum()
        elif reduction == "sum":
            loss = loss.sum()

        return loss
    # def __init__(self, epsilon: float = 0.5, gamma: float = 2, delta: float = 0.4, alpha: float = 1.0, eps: float = 1e-12):
    #     super(DIOULoss, self).__init__()
    #     self.epsilon = epsilon
    #     self.gamma = gamma
    #     self.delta = delta
    #     self.alpha = alpha
    #     self.eps = eps

    # def forward(self, inputs, targets):
    #     prob = inputs.sigmoid()
    #     ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    #     p_t = prob * targets + (1 - prob) * (1 - targets)
    #
    #     one_hot = targets > 0.5
    #     with torch.no_grad():
    #         p_sum = torch.sum(torch.where(one_hot, p_t, 0), dim=-1, keepdim=True)
    #         ps_sum = torch.sum(torch.where(one_hot, 1, 0), dim=-1, keepdim=True)
    #         gamma_adj = self.gamma + (1 - (p_sum / (ps_sum + self.eps)))
    #
    #     beta = (1 - p_t) ** gamma_adj
    #
    #     with torch.no_grad():
    #         sw_sum = torch.sum(torch.ones(p_t.shape, device=p_t.device), dim=-1, keepdim=True)
    #         beta_sum = (1 + self.delta * gamma_adj) * torch.sum(beta, dim=-1, keepdim=True) + self.eps
    #         mult = sw_sum / beta_sum
    #
    #     loss = mult * ce_loss * beta + self.alpha * (1 - p_t) ** (gamma_adj + 1)
    #
    #     if self.epsilon >= 0:
    #         epsilon_t = self.epsilon * targets + (1 - self.epsilon) * (1 - targets)
    #         loss = epsilon_t * loss
    #
    #     return loss.mean(1).sum()

@LOSSES.register_module()
class GIOULoss(nn.Module):
    def __init__(self):
        super(GIOULoss, self).__init__()

    def forward(
        self,
        input_bboxes: torch.Tensor,
        target_bboxes: torch.Tensor,
        reduction: str = "none",
        eps: float = 1e-8,
    ) -> torch.Tensor:
        loss = 1 - torch.diag(compute_giou_torch(target_bboxes, input_bboxes))

        if reduction == "mean":
            loss = loss.mean() if loss.numel() > 0 else 0.0 * loss.sum()
        elif reduction == "sum":
            loss = loss.sum()
        return loss
