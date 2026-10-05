import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp


class SeqAttention(nn.Module):
    """ Sequence-level SE Attention (轻量版替换原自注意力) """
    def __init__(self,
                 dim=768,
                 mlp_ratio=4,
                 dropout=0.0,
                 vis_token=1182,
                 reduction=16):
        super().__init__()

        self.vis_token = vis_token

        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.dropout = nn.Dropout(dropout)
        self.drop_path = nn.Identity()

        # === SE 模块 ===
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),       # [B, C, L] -> [B, C, 1]
            nn.Conv1d(dim, dim // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(dim // reduction, dim, kernel_size=1),
            nn.Sigmoid()
        )

        # === MLP 残差层 ===
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, dim)
        )

    def forward(self, x):
        """
        x: [B, L, C]
        L = 1184 = 6*(197) + 2, 前1182为视觉，后2为文本
        """
        visual_tokens = x[:, :self.vis_token, :]
        text_tokens = x[:, self.vis_token:, :]

        # === 对视觉 token 做 SE 注意力 ===
        x_norm = self.norm(visual_tokens)
        x_perm = x_norm.permute(0, 2, 1).contiguous()          # [B, C, L]
        se_weight = self.se(x_perm)              # [B, C, 1]
        x_se = (x_perm * se_weight).permute(0, 2, 1).contiguous()   # [B, L, C]

        # === 残差 + MLP ===
        out = visual_tokens + self.dropout(x_se)
        out = out + self.mlp(out)

        return torch.cat([out, text_tokens], dim=1)


class SeqAttnModule(nn.Module):
    def __init__(self,
                 dim=768,
                 mlp_ratio=4,
                 num_layers=3,
                 vis_token=1182):
        super().__init__()
        self.layers = nn.ModuleList([
            SeqAttention(
                dim=dim,
                mlp_ratio=mlp_ratio,
                vis_token=vis_token
            )
            for _ in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class SSA(nn.Module):
    """
    Sequence Semantic Alignment (SSA)
    完全冻结 GIT，仅在 ADATAD 主干中插入三层通道注意力（由 SE 替代原自注意力）
    同时用 InfoNCE + 背景 L2 收缩约束训练 SCA。
    """
    def __init__(self,
                 dim=768,
                 num_heads=12,
                 mlp_ratio=4,
                 with_cp=False,
                 num_layers=3,
                 num_img_tokens=197,
                 num_frames=6,
                 tau=0.07,
                 lambda_bg=0.2):
        super().__init__()

        self.num_img_tokens = num_img_tokens
        self.num_frames = num_frames
        self.lambda_bg = lambda_bg
        self.vis_token = num_img_tokens * num_frames

        # SE
        self.SeqAttn_layer = SeqAttnModule(
            dim=dim,
            mlp_ratio=mlp_ratio,
            num_layers=num_layers,
            vis_token=self.vis_token
        )

        from opentad.models.losses.masked_infoNCE_loss2 import InfoNCELoss
        self.loss_infonce = InfoNCELoss(dim=dim, temperature=tau, bidirectional=False, MLP=True)
        # self.ln_vis = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, last_hidden_state, t2, gt_label_embed=None, frame_label_ids=None):
        """
        last_hidden_state: [B, 1184, 768] (冻结 GIT 输出)
        gt_label_embed: [B, L, 768]
        frame_label_ids: [B, T]，0 表示背景帧
        """

        B, L, C = last_hidden_state.shape

        # ===== 1. 三层通道注意力增强 =====
        SeqAttn = self.SeqAttn_layer(last_hidden_state)   # [B,1184,C] 可用于解码

        # ===== 2. 每帧池化取语义 =====
        vis = SeqAttn[:, :self.vis_token, :].view(B, self.num_frames, self.num_img_tokens, C)  # (B,6,197,C), 197是1+196 cls+visual patch
        cls = vis[:, :, 0, :]               # 每一帧的cls
        patch_mean = vis[:, :, 1:, :].mean(dim=2)  # 每一帧的196token的平均
        frame_level_mean = 0.5 * cls + 0.5 * patch_mean      # [B, T, C] (256,6,768)    可用于ADATAD。。。先修改batch
        frame_level_mean = frame_level_mean.reshape(B // int(t2), -1, C)  # BTC(2,768,768)

        # resume temporal sequence, prepare for mask
        # if self.training:
        #     gt_label_embed = gt_label_embed.reshape(-1, t2 * self.num_frames, C)   # BTC(2,768,768)
        #
        #     mask_act = (frame_label_ids != 0)  # [B, T] = [2, 768]
        #     mask_bg = ~mask_act
        #
        # # ===== 3. 动作帧 InfoNCE =====
        #     if mask_act.any():
        #         vis_act = frame_level_mean[mask_act]
        #         text_act = gt_label_embed[mask_act]
        #         vis_act = F.normalize(self.ln_vis(vis_act), dim=-1)
        #         text_act = F.normalize(
        #             F.layer_norm(text_act, text_act.shape[-1:], eps=1e-6),  # 无参固定LN
        #             dim=-1
        #         )
        #         # stat_info("vis_act", vis_act)   # 验证统计特征，for loss计算
        #         # stat_info("text_act", text_act)
        #         # vis_for_infonce = vis_act.clone().detach().requires_grad_(True)
        #         # text_for_infonce = text_act.clone().detach().requires_grad_(True)
        #         loss_infonce = self.loss_infonce(vis_act, text_act)
        #     else:
        #         loss_infonce = torch.tensor(0.0, device=frame_level_mean.device)
        #
        # # ===== 4. 背景帧 L2 收缩 =====
        #     if mask_bg.any():
        #         vis_bg = frame_level_mean[mask_bg]
        #         loss_bg = (vis_bg ** 2).mean()              # 向 0 收缩
        #     else:
        #         loss_bg = torch.tensor(0.0, device=frame_level_mean.device)
        #
        #     SSA_loss = loss_infonce + self.lambda_bg * loss_bg
        if self.training:
            gt_label_embed = gt_label_embed.reshape(-1, t2 * self.num_frames, C)   # BTC(2,768,768)
            SSA_loss = self.loss_infonce(
                frame_level_mean,  # [B,T,C]
                gt_label_embed,  # [B,T,C]
                mask=frame_label_ids,  # [B,T], 1=动作,0=背景
                lambda_bg=self.lambda_bg
            )

        frame_level_mean = frame_level_mean.permute(0, 2, 1)  # [B, C, T]

        if self.training:
            return frame_level_mean, SSA_loss   # (B,C768,T768), 是否detach关系检测损失是否影响SSA
        else:
            return frame_level_mean


def stat_info(name, x):
    mean = x.mean().item()
    std = x.std().item()
    minv = x.min().item()
    maxv = x.max().item()
    norm = x.norm(p=2, dim=-1).mean().item()
    print(f"[{name}] mean={mean:.4f}, std={std:.4f}, min={minv:.4f}, max={maxv:.4f}, ||x||_2={norm:.4f}")
