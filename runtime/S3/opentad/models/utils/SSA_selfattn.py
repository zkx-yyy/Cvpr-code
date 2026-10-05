import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp


class SeqAttention(nn.Module):
    """ Sequence-level Self-Attention along L dimension (for visual tokens only) """
    def __init__(self,
                 dim=768,
                 mlp_ratio=4,
                 with_cp=True,
                 dropout=0.0,
                 num_heads=12,
                 vis_token=1182):  # num_heads=8
        super().__init__()

        self.vis_token = vis_token
        self.with_cp = with_cp

        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # QKV projection
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.proj_out = nn.Linear(dim, dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, dim)
        )

        self.dropout = nn.Dropout(dropout)
        self.drop_path = nn.Identity()

        # self.gamma = nn.Parameter(torch.zeros(1))

    def _inner_forward(self, x):
        """
        x: [B, L, C]
        L = 1184 = 6*(197) + 2
        前1182为视觉，后2为文本
        """
        B, L_v, C = x.shape

        x_norm = self.norm(x)
        Q = self.q_proj(x_norm).reshape(B, L_v, self.num_heads, self.head_dim).transpose(1, 2)  # [B,H,Lv,D]
        K = self.k_proj(x_norm).reshape(B, L_v, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x_norm).reshape(B, L_v, self.num_heads, self.head_dim).transpose(1, 2)

        attn_output = F.scaled_dot_product_attention(
            Q, K, V,
            dropout_p=self.dropout.p,
            is_causal=False
        )
        attn_output = attn_output.transpose(1, 2).reshape(B, L_v, C)
        attn_output = self.proj_out(attn_output)
        attn_output = self.dropout(attn_output)

        attn_output = x + self.drop_path(attn_output)
        attn_output = attn_output + self.drop_path(self.mlp(attn_output))
        return attn_output


    def forward(self, x):
        """
        x: [B, L, C], where L = 1184 = 1182(visual) + 2(text)
        """
        visual_tokens = x[:, :self.vis_token, :]
        text_tokens = x[:, self.vis_token:, :]

        if self.with_cp:
            visual_tokens = cp.checkpoint(self._inner_forward, visual_tokens)
        else:
            visual_tokens = self._inner_forward(visual_tokens)

        out = torch.cat([visual_tokens, text_tokens], dim=1)
        return out


class SeqAttnModule(nn.Module):
    def __init__(self,
                 dim=768,
                 num_heads=12,
                 mlp_ratio=4,
                 with_cp=True,
                 num_layers=3,
                 vis_token=1182
                 ):
        super().__init__()
        self.layers = nn.ModuleList([
            SeqAttention(
                dim=dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                with_cp=with_cp,
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
    完全冻结 GIT，仅在 ADATAD 主干中插入三层通道注意力
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
                 lambda_bg=0.2):  # ！
        super().__init__()

        self.num_img_tokens = num_img_tokens
        self.num_frames = num_frames
        self.lambda_bg = lambda_bg

        self.vis_token = num_img_tokens * num_frames


        self.SeqAttn_layer = SeqAttnModule(
            dim=dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            with_cp=with_cp,
            num_layers=num_layers,
            vis_token=self.vis_token
        )

        from opentad.models.losses.masked_infoNCE_loss import InfoNCELoss
        self.loss_infonce = InfoNCELoss(dim=dim, temperature=tau, bidirectional=False, MLP=False)  # 你妈

        # self.ln_vis = nn.LayerNorm(dim, eps=1e-6)
        # self.ln_text = nn.LayerNorm(dim, eps=1e-6)
        self.linear_v = nn.Linear(dim, dim, bias=False)
        # self.linear_t = nn.Linear(dim, dim, bias=False)
        self.text_proj = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim),
        )
        # self.gamma = nn.Parameter(torch.tensor(0.1))

    # @torch.cuda.amp.autocast(enabled=False)
    def forward(self, last_hidden_state, t2, gt_label_embed=None, frame_label_ids=None):
        """
        last_hidden_state: [B, 1184, 768] (冻结 GIT 输出)
        gt_label_embed: [B, L, 768]
        frame_label_ids: [B, T]，0 表示背景帧
        """
        # print("[DEBUG] SSA input requires_grad =", last_hidden_state.requires_grad) #
        # print("[DEBUG] SSA input dtype/device =", last_hidden_state.dtype, last_hidden_state.device) #

        git_features = last_hidden_state.requires_grad_(True)
        B, L, C = git_features.shape

        #===== 1. 三层通道注意力增强 =====
        SeqAttn = self.SeqAttn_layer(git_features)   # [B,1184,C] 可用于解码

        # SeqAttn = SeqAttn + self.gamma * SeqAttn

        # ===== 2. 每帧池化取语义 =====
        vis = SeqAttn[:, :self.vis_token, :].view(B, self.num_frames, self.num_img_tokens, C)  # (B,6,197,C), 197是1+196 cls+visual patch
        cls = vis[:, :, 0, :]                # 每一帧的cls
        patch_mean = vis[:, :, 1:, :].mean(dim=2)  # 每一帧的196token的平均
        frame_level_mean = 0.5 * cls + 0.5 * patch_mean      # [B, T, C] (256,6,768)    可用于ADATAD。。。先修改batch
        frame_level_mean = frame_level_mean.reshape(B//int(t2), -1, C)  # BTC(2,768,768)

        frame_level_mean = self.linear_v(frame_level_mean)
        # print("linear_v weight norm:", self.linear_v.weight.norm().item())

        compute_loss = (gt_label_embed is not None) and (frame_label_ids is not None)  # 无这两项说明是test,则跳过loss
        if compute_loss:
            gt_label_embed = gt_label_embed.to(self.text_proj[-1].weight.dtype)  #
            gt_label_embed = gt_label_embed.to(self.text_proj[-1].weight.device) #

            gt_label_embed = gt_label_embed.reshape(-1, t2 * self.num_frames, C)  # BTC(2,768,768)
            gt_label_embed = self.text_proj(gt_label_embed)

            SSA_loss = self.loss_infonce(
                frame_level_mean,  # [B,T,C]
                gt_label_embed,  # [B,T,C]
                mask=frame_label_ids,  # [B,T], n=动作id,0=背景
                lambda_bg=self.lambda_bg
            )
            # SSA_loss = torch.tensor(0.1, device=frame_level_mean.device, requires_grad=True)

        frame_level_mean = frame_level_mean.permute(0, 2, 1)  # [B, C, T]

        if compute_loss:
            return frame_level_mean, SSA_loss   # (B,C768,T768), 是否detach关系检测损失是否影响SSA
        else:
            return frame_level_mean
        #     "out": SeqAttn,           # (B,1184,C)
        #     "SAC_loss": SAC_loss,
        #     # "loss_infonce": loss_infonce,
        #     # "loss_bg": loss_bg
        # }

def stat_info(name, x):
    mean = x.mean().item()
    std = x.std().item()
    minv = x.min().item()
    maxv = x.max().item()
    norm = x.norm(p=2, dim=-1).mean().item()
    print(f"[{name}] mean={mean:.4f}, std={std:.4f}, min={minv:.4f}, max={maxv:.4f}, ||x||_2={norm:.4f}")