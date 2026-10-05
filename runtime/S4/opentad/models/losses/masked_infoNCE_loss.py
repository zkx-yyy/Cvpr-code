import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    """Paper-aligned foreground contrastive loss for semantic anchors."""

    def __init__(self, dim, temperature=0.07, **kwargs):
        super().__init__()
        self.temperature = temperature
        self.last_infonce = None
        self.last_bg = None

    def forward(self, visual, text, class_ids, valid_mask=None, lambda_bg=0.0):
        """Align foreground visual tokens with same-class semantic anchors."""
        if visual.shape != text.shape or visual.dim() != 3:
            raise ValueError(f"visual/text must share [B, T, C], got {visual.shape} and {text.shape}")

        batch_size, num_steps, channels = visual.shape
        device = visual.device
        class_ids = class_ids.to(device=device, dtype=torch.long)
        if class_ids.shape != (batch_size, num_steps):
            raise ValueError(
                f"class_ids must be [B, T]={batch_size, num_steps}, got {tuple(class_ids.shape)}"
            )

        if valid_mask is None:
            valid_mask = torch.ones((batch_size, num_steps), device=device, dtype=torch.bool)
        else:
            valid_mask = valid_mask.to(device=device, dtype=torch.bool)
            if valid_mask.shape != (batch_size, num_steps):
                raise ValueError(
                    f"valid_mask must be [B, T]={batch_size, num_steps}, got {tuple(valid_mask.shape)}"
                )

        foreground = valid_mask & class_ids.gt(0)
        if lambda_bg > 0:
            background = valid_mask & ~foreground
            loss_bg = (text.square() * background.unsqueeze(-1)).sum()
            loss_bg = loss_bg / (background.sum() * channels).clamp_min(1)
        else:
            loss_bg = text.sum() * 0.0

        row_idx = foreground.reshape(-1).nonzero(as_tuple=True)[0]
        if row_idx.numel() == 0:
            self.last_infonce = visual.sum() * 0.0
            self.last_bg = loss_bg
            return lambda_bg * loss_bg

        visual = F.normalize(F.layer_norm(visual, (channels,), eps=1e-6), dim=-1, eps=1e-6)
        text = F.normalize(F.layer_norm(text, (channels,), eps=1e-6), dim=-1, eps=1e-6)
        visual = visual.reshape(batch_size * num_steps, channels)
        text = text.reshape(batch_size * num_steps, channels)
        flat_classes = class_ids.reshape(-1)
        flat_valid = valid_mask.reshape(-1)

        row_classes = flat_classes[row_idx]
        candidate_mask = flat_valid[None, :]
        positive_mask = candidate_mask & row_classes[:, None].eq(flat_classes[None, :])
        negative_exists = (candidate_mask & ~positive_mask).any(dim=1)

        if not negative_exists.any():
            self.last_infonce = visual.sum() * 0.0
            self.last_bg = loss_bg
            return lambda_bg * loss_bg

        logits = (visual[row_idx] @ text.t() / self.temperature).float()
        min_value = torch.finfo(logits.dtype).min
        log_denominator = torch.logsumexp(logits.masked_fill(~candidate_mask, min_value), dim=1)
        log_numerator = torch.logsumexp(logits.masked_fill(~positive_mask, min_value), dim=1)
        loss_infonce = -(log_numerator[negative_exists] - log_denominator[negative_exists]).mean()

        self.last_infonce = loss_infonce
        self.last_bg = loss_bg
        return loss_infonce + lambda_bg * loss_bg
