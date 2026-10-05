import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from mmcv.cnn.bricks.transformer import FFN
from mmaction.utils import ConfigType
import math

"""
CRF/CMRF为初期尝试的融合方案,可删除
"""
# class CRF(nn.Module):
#     def __init__(self, dim: int, use_ln_attn: bool = False, print_stats: bool = False):
#         super().__init__()
#         self.print_stats = print_stats
#
#         self.ln_vis = nn.LayerNorm(dim)
#         self.ln_attn = nn.LayerNorm(dim) if use_ln_attn else nn.Identity()
#
#         self.fusion_fc = nn.Sequential(
#             nn.Linear(dim * 2, dim),
#             nn.ReLU(),
#             nn.LayerNorm(dim)
#         )
#
#         self.post_ln = nn.LayerNorm(dim)
#         # ReZero：初始0，逐步学到注入强度
#         self.alpha = nn.Parameter(torch.zeros(1))
#
    # def forward(self, x_visual: torch.Tensor, attn_output: torch.Tensor) -> torch.Tensor:
    #     """
    #     x_visual: [B, T, C]
    #     attn_output: [B, T, C]  (Q=visual, K/V=text 得到的“文本增强视觉token”)
    #     """
#         v = self.ln_vis(x_visual)
#         a = self.ln_attn(attn_output)
#
#         delta = self.fusion_fc(torch.cat([v, a], dim=-1))  # [B, T, C]
#         fused = x_visual + self.alpha * delta
#         fused = self.post_ln(fused)
#
#         if self.training and self.print_stats:
#             with torch.no_grad():
#                 cos = F.cosine_similarity(
#                     x_visual.reshape(x_visual.size(0), -1),
#                     fused.reshape(fused.size(0), -1),
#                     dim=1
#                 ).mean()
#                 delta_ratio = ((self.alpha * delta).norm(dim=(1, 2)) /
#                                (fused.norm(dim=(1, 2)) + 1e-12)).mean()
#                 print(f"[ConcatResidual] cos={cos:.4f}, delta_ratio={delta_ratio:.4f}, alpha={self.alpha.item():.4f}")
#         return fused
#
# class CMRF(nn.Module):
#     def __init__(self, dim: int, hidden_ratio: int = 4, drop: float = 0.1,
#                  use_ln_attn: bool = False, print_stats: bool = False):
#         super().__init__()
#         self.print_stats = print_stats
#
#         self.ln_vis = nn.LayerNorm(dim)
#         self.ln_attn = nn.LayerNorm(dim) if use_ln_attn else nn.Identity()
#
#         hidden = dim * hidden_ratio
#         self.fusion_fc = nn.Sequential(
#             nn.Linear(dim * 2, hidden),
#             nn.GELU(),
#             nn.Dropout(drop),
#             nn.Linear(hidden, dim)
#         )
#
#         # self.alpha_scale = 0.06
#         # self.raw_alpha = nn.Parameter(torch.zeros(1))
#
#         self.post_ln = nn.LayerNorm(dim)
#
#     def forward(self, x_visual: torch.Tensor, attn_output: torch.Tensor) -> torch.Tensor:
#         """
#         x_visual: [B, T, D]
#         attn_output: [B, T, D]
#         """
#         v = self.ln_vis(x_visual)
#         a = self.ln_attn(attn_output)
#
#         delta = self.fusion_fc(torch.cat([v, a], dim=-1))  # [B, T, D]
#         # alpha = self.alpha_scale * torch.sigmoid(self.raw_alpha)
#         fused = x_visual + 0.2 * delta
#         fused = self.post_ln(fused)
#
#         if self.training and self.print_stats:
#             with torch.no_grad():
#                 cos = F.cosine_similarity(
#                     x_visual.reshape(x_visual.size(0), -1),
#                     fused.reshape(fused.size(0), -1),
#                     dim=1
#                 ).mean()
#                 delta_ratio = ((delta).norm(dim=(1, 2)) /   # alpha * delta
#                                (fused.norm(dim=(1, 2)) + 1e-12)).mean()
#                 print(
#                     f"[ConcatMLPResidual] cos={cos:.4f}, delta_ratio={delta_ratio:.4f}") # , alpha={alpha.item():.4f}
#         return fused

class VerbGuidedCA(nn.Module):    # 视觉塔
    """
    Cross-Attention Module for Verb-guided Feature Modulation.

    Q = visual, K = text, V = text.
    """

    def __init__(self, dim, num_heads=8, dropout=0.0, use_gamma=False, fusion_init_scale=0.1):
        """
        Args:
            dim (int): embedding dimension
            num_heads (int): number of attention heads
            dropout (float): dropout for attention
            use_norm (bool): whether to use LayerNorm after attn
            use_gamma (bool): whether to use learnable scaling γ
        """
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        self.dropout = nn.Dropout(dropout)
        self.dropout_p = dropout

        self.use_gamma = use_gamma
        self.fusion_scale = nn.Parameter(torch.tensor(float(fusion_init_scale)))
        # if use_gamma:
        #     # self.gamma = nn.Parameter(torch.tensor(1.0)) # 标量
        #     self.gamma = nn.Parameter(torch.ones(1, 1, dim)*0.5) # 逐通道
        # self.gamma.register_hook(lambda g: print(f"[backward] gamma.grad_norm={g.norm().item():.6f}"))

        # gated
        self.gate_fc = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.Sigmoid()  # 输出 ∈ (0,1)
        )

        # self.fusion = CRF(dim,print_stats=True)
        # self.fusion = CMRF(dim, print_stats=True)

        def _debug_qkv_stats(      # debug打印，非必要
                Q, K, V,
                name="XATTN",
                heads_to_check=(0,),  # 只看哪些 head，避免输出太多
                max_tq=64,  # 限制参与统计的 query token 数（太大就慢）
                max_tk=256,  # 限制参与统计的 key token 数
                cos_pairs=512,  # cos 采样对数
                do_svd=True,  # 是否做 SVD（较慢）
                svd_center=True,  # SVD 前是否在时间维做去均值
                svd_topk=(1, 4, 8, 16),  # 输出前 k 个能量占比
                do_logits=True,  # 是否统计 logits / entropy
                do_cos_logits=True,  # 是否统计 normalize 后 logits_cos
                eps=1e-12,
                print_fn=print, ):
            """
            Q,K,V: [B, H, T, Dh]
            输出：幅值、角度、近低秩、logits尖锐度等统计。
            插在 q_proj/k_proj/v_proj reshape transpose 之后。
            统计QKV的幅值大小，以观察是否有效学习。
            """
            with torch.no_grad():
                assert Q.dim() == 4 and K.dim() == 4 and V.dim() == 4, "Q/K/V must be [B,H,T,Dh]"
                Bq, Hq, Tq, Dhq = Q.shape
                Bk, Hk, Tk, Dhk = K.shape
                assert Hq == Hk, f"head mismatch: Hq={Hq}, Hk={Hk}"
                assert Dhq == Dhk, f"head_dim mismatch: Dhq={Dhq}, Dhk={Dhk}"

                b = 0  # 只看第一个 batch
                tq = min(Tq, max_tq)
                tk = min(Tk, max_tk)

                print_fn(
                    f"\n[{name}] Q={tuple(Q.shape)} K={tuple(K.shape)} V={tuple(V.shape)}  (using b={b}, tq={tq}, tk={tk})")

                # helper: entropy
                def attn_entropy(p):
                    return -(p * (p.clamp_min(eps).log())).sum(dim=-1)

                for h in heads_to_check:
                    if h < 0 or h >= Hk:
                        continue

                    Qh = Q[b, h, :tq, :].float()  # [tq, Dh]
                    Kh = K[b, h, :tk, :].float()  # [tk, Dh]

                    # -------- 1) 幅值：K norm 分布 --------
                    kn = Kh.norm(dim=-1)  # [tk]
                    print_fn(
                        f"[{name}][h={h}] K-norm: mean={kn.mean().item():.4f} std={kn.std().item():.4f} "
                        f"min={kn.min().item():.4f} max={kn.max().item():.4f} range={(kn.max() - kn.min()).item():.4f}"
                    )

                    # -------- 2) 角度：K 时间 token 的 cos 相似度采样 --------
                    if tk >= 2 and cos_pairs > 0:
                        # 随机采样 i,j
                        i = torch.randint(0, tk, (cos_pairs,), device=Kh.device)
                        j = torch.randint(0, tk, (cos_pairs,), device=Kh.device)
                        mask = (i != j)
                        i = i[mask];
                        j = j[mask]
                        if i.numel() > 0:
                            cosv = F.cosine_similarity(Kh[i], Kh[j], dim=-1)  # [N]
                            q10, q50, q90 = torch.quantile(cosv, torch.tensor([0.1, 0.5, 0.9], device=cosv.device)).tolist()
                            print_fn(
                                f"[{name}][h={h}] K-cos(samp): mean={cosv.mean().item():.6f} std={cosv.std().item():.6f} "
                                f"q10={q10:.6f} q50={q50:.6f} q90={q90:.6f}"
                            )
                        else:
                            print_fn(f"[{name}][h={h}] K-cos(samp): skipped (no valid pairs)")

                    # -------- 3) 近低秩：SVD 能量占比 --------
                    if do_svd and tk >= 2:
                        X = Kh
                        if svd_center:
                            X = X - X.mean(dim=0, keepdim=True)  # [tk, Dh]
                        try:
                            # SVD: X = U S Vh
                            # full_matrices=False 更快
                            U, S, Vh = torch.linalg.svd(X, full_matrices=False)
                            energy = S.pow(2)
                            denom = energy.sum().clamp_min(eps)
                            ratios = {}
                            for k in svd_topk:
                                kk = min(k, energy.numel())
                                ratios[k] = (energy[:kk].sum() / denom).item()
                            ratio_str = " ".join([f"r{k}={ratios[k]:.4f}" for k in svd_topk])
                            print_fn(f"[{name}][h={h}] SVD-energy ({'centered' if svd_center else 'raw'}): {ratio_str}")
                        except RuntimeError as e:
                            print_fn(f"[{name}][h={h}] SVD failed: {str(e)[:120]}...")

                    # -------- 4) logits 尖锐度（最终关心） --------
                    if do_logits:
                        # scaled dot-product logits
                        logits = (Qh @ Kh.t()) / math.sqrt(Dhq)  # [tq, tk]
                        l_mean = logits.mean().item()
                        l_std = logits.std().item()
                        row_range = (logits.max(dim=1).values - logits.min(dim=1).values).mean().item()

                        attn = torch.softmax(logits, dim=-1)  # [tq,tk]
                        ent = attn_entropy(attn).mean().item()
                        ent_max = math.log(tk)

                        print_fn(
                            f"[{name}][h={h}] logits: mean={l_mean:.4f} std={l_std:.4f} row_range={row_range:.4f} "
                            f"entropy={ent:.4f} (max={ent_max:.4f})"
                        )

                    # -------- 5) 剥离幅值：normalize 后只看角度 logits_cos --------
                    if do_cos_logits and tk >= 2:
                        Qn = F.normalize(Qh, dim=-1)
                        Kn = F.normalize(Kh, dim=-1)
                        logits_cos = (Qn @ Kn.t())  # [tq, tk]
                        lc_std = logits_cos.std().item()
                        lc_range = (logits_cos.max(dim=1).values - logits_cos.min(dim=1).values).mean().item()
                        # 可选：同样算 entropy（这里不需要 /sqrt(D)）
                        attn_cos = torch.softmax(logits_cos, dim=-1)
                        ent_cos = attn_entropy(attn_cos).mean().item()
                        print_fn(
                            f"[{name}][h={h}] logits_cos(normed): std={lc_std:.4f} row_range={lc_range:.4f} "
                            f"entropy={ent_cos:.4f} (max={math.log(tk):.4f})"
                        )

        self._debug_qkv_stats = _debug_qkv_stats

    def forward(self, x_visual, x_text):
        """
        Args:
            x_visual: [B, T, C]
            x_text:   [B, N, C]

        Returns:
            fused: [B, T_v, C]
        """
        # BCT → BTC
        # x_visual = x_visual.permute(0, 2, 1)  # [B, T_v, C]
        # x_text = x_text.permute(0, 2, 1)      # [B, T_t, C] # 仅标签做效果验证的时候(lable_only) 不需要这个

        B_v, T_v, D = x_visual.size()
        # x_text = x_text.permute(0, 2, 1)  # [B, T_t, C]
        B_t, T_t, _= x_text.size()

        # print('xv', x_visual.shape, 'xt', x_text.shape)
        # print('qkv out_features', self.q_proj.out_features, self.k_proj.out_features, self.v_proj.out_features)
        # print('q_lin/k_lin/v_lin', self.q_proj.shape, self.k_proj.shape, self.v_proj.shape)
        # print('heads', self.num_heads, 'Dh', Dh)

        # Linear projection
        Q = self.q_proj(x_visual).reshape(B_v, T_v, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, T_v, D_head]
        K = self.k_proj(x_text).reshape(B_t, T_t, self.num_heads, self.head_dim).transpose(1, 2)    # [B, H, T_t, D_head]
        V = self.v_proj(x_text).reshape(B_t, T_t, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, T_t, D_head]

        # Attention
        # attn_scores = (Q @ K.transpose(-2, -1)) / (self.head_dim ** 0.5)  # [B, H, T_v, T_t]
        # attn_probs = F.softmax(attn_scores, dim=-1)
        # attn_probs = self.dropout(attn_probs)
        #
        # attn_output = attn_probs @ V  # [B, H, T_v, D_head]
        # attn_output = attn_output.transpose(1, 2).reshape(B, T_v, D)  # [B, T_v, D]
        # attn_output = self.out_proj(attn_output)

        # Scaled Dot Product Attention (PyTorch 2.x)
        attn_output = F.scaled_dot_product_attention(
            Q, K, V,
            dropout_p=self.dropout_p,
            is_causal=False
        )  # [B, H, T_v, D_head]
        attn_output = attn_output.transpose(1, 2).reshape(B_v, T_v, D)  # (BTC)
        attn_output = self.out_proj(attn_output)
        attn_output = self.dropout(attn_output)


        """
        不同的残差方式
        """
        # fused = attn_output  ####

        # fuse
        # fused = self.fusion(x_visual, attn_output)

        # gamma fuse
        # if self.use_gamma:
        #     # fused = x_visual + self.gamma * attn_output
        #     # a= torch.sigmoid(self.gamma)
        #     fused = (1 - self.gamma) * x_visual + self.gamma * attn_output
        # else:
        #     fused = x_visual + attn_output

        # # gated fuse
        fused = torch.tanh(self.fusion_scale) * attn_output

        # BTC → BCT
        # fused = fused.permute(0, 2, 1)  # [B, C, T_v]
        return fused # (BTC)


# class VisualGuidedCA(nn.Module):    # 文本塔
#     """
#     Cross-Attention Module for Verb-guided Feature Modulation.
#
#     Q = text, K = visual, V = visual.
#     """
#
#     def __init__(self, dim, num_heads=8, dropout=0.0, use_gamma=False):
#
#         super().__init__()
#         self.num_heads = num_heads
#         self.head_dim = dim // num_heads
#         assert dim % num_heads == 0, "dim must be divisible by num_heads"
#
#         self.q_proj = nn.Linear(dim, dim)
#         self.k_proj = nn.Linear(dim, dim)
#         self.v_proj = nn.Linear(dim, dim)
#         self.out_proj = nn.Linear(dim, dim)
#
#         self.dropout = nn.Dropout(dropout)
#         self.dropout_p = dropout
#
#         self.use_gamma = use_gamma
#
#     def forward(self, x_visual, x_text):
#         B_v, T_v, D = x_visual.size()
#         # x_text = x_text.permute(0, 2, 1)  # [B, T_t, C]
#         B_t, T_t, _= x_text.size()
#
#         # Linear projection
#         Q = self.q_proj(x_text).reshape(B_t, T_t, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, T_t, D_head]
#         K = self.k_proj(x_visual).reshape(B_v, T_v, self.num_heads, self.head_dim).transpose(1, 2)    # [B, H, T_v, D_head]
#         V = self.v_proj(x_visual).reshape(B_v, T_v, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, T_v, D_head]
#
#         # Attention
#         # attn_scores = (Q @ K.transpose(-2, -1)) / (self.head_dim ** 0.5)  # [B, H, T_v, T_t]
#         # attn_probs = F.softmax(attn_scores, dim=-1)
#         # attn_probs = self.dropout(attn_probs)
#         #
#         # attn_output = attn_probs @ V  # [B, H, T_v, D_head]
#         # attn_output = attn_output.transpose(1, 2).reshape(B, T_v, D)  # [B, T_v, D]
#         # attn_output = self.out_proj(attn_output)
#
#         # Scaled Dot Product Attention (PyTorch 2.x)
#         attn_output = F.scaled_dot_product_attention(
#             Q, K, V,
#             dropout_p=self.dropout_p,
#             is_causal=False
#         )  # [B, H, T_v, D_head]
#         attn_output = attn_output.transpose(1, 2).reshape(B_t, T_t, D)  # (BTC)
#         attn_output = self.out_proj(attn_output)
#         attn_output = self.dropout(attn_output)
#
#         # gated fuse
#         if self.use_gamma:
#             gate = self.gate_fc(torch.cat([x_visual, attn_output], dim=-1))
#             fused = (1 - gate) * x_visual + gate * attn_output
#         else:
#             fused = attn_output
#
#         # BTC → BCT
#         # fused = fused.permute(0, 2, 1)  # [B, C, T_v]
#         return fused # (BTC)

class VTBlock(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio,
                 with_cp=False,
                 dropout=0.0,
                 act_cfg: ConfigType = dict(type="GELU"),
                 fusion_init_scale=0.1):
        super().__init__()

        self.with_cp = with_cp

        self.norm_v1 = nn.LayerNorm(dim, eps=1e-6)
        self.norm_t1 = nn.LayerNorm(dim, eps=1e-6)

        self.v_attn = VerbGuidedCA(dim, num_heads, fusion_init_scale=fusion_init_scale)
        # self.t_attn = VisualGuidedCA(dim, num_heads)
        self.drop_path = nn.Identity() # 也可以self.dropout_v = nn.Dropout(dropout)  # dropout = 0.1

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.v_mlp = FFN(
            embed_dims=dim,
            feedforward_channels=mlp_hidden_dim,
            act_cfg=act_cfg,
            ffn_drop=0.0,
            add_identity=False,
        )
        self.t_mlp = FFN(
            embed_dims=dim,
            feedforward_channels=mlp_hidden_dim,
            act_cfg=act_cfg,
            ffn_drop=0.0,
            add_identity=False,
        )

    def forward(self, x_v, x_t):
        """
        args:
            x_v: [B, T_v, C]
            x_t: [B, N, C] 仅lable时确定是BTC。
        """
        def _inner_forward(x_v, x_t):
            # x = x_v  # [B, C, T]
            # x_v = x_v.permute(0, 2, 1)  # BCT -> BTC
            x_v = self.norm_v1(x_v)

            # visual
            # x_t = self.norm_t1(x_t)
            v_fused = x_v + self.drop_path(self.v_attn(x_v, x_t))  # 最新版本attn输出BTC
            # v_fused = v_fused.permute(0, 2, 1)  # BCT -> BTC
            v_fused = v_fused + self.drop_path(self.v_mlp(v_fused))
            # v_fused = v_fused.permute(0, 2, 1)  # BTC -> BCT

            # # text
            # t_fuesd = x_t + self.drop_path(self.t_attn(x_v, x_t))  # attn BTC
            # t_fuesd = t_fuesd + self.drop_path(self.t_mlp(t_fuesd))

            return v_fused     #t_fuesd   # BTC

        if self.with_cp:
            v_fused = cp.checkpoint(_inner_forward, x_v, x_t)
        else:
            v_fused = _inner_forward(x_v, x_t)
        return v_fused

class VTModuel(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio,
                 with_cp=False,
                 dropout=0.0,
                 num_layers=3,
                 hidden_ratio=4, # for MLP
                 act_cfg: ConfigType = dict(type="GELU"),
                 fusion_init_scale=0.1):
        super().__init__()

        # self.proj_visual =nn.LayerNorm(dim, eps=1e-6)

        self.proj_text = nn.Sequential(     # 1.8%是使用先Linear后LN
            nn.LayerNorm(dim, eps=1e-6),
            nn.Linear(dim, dim)
        )

        self.layers = nn.ModuleList([
            VTBlock(
                dim=dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                with_cp=with_cp,
                dropout=dropout,
                act_cfg=act_cfg,
                fusion_init_scale=fusion_init_scale,
            )
            for _ in range(num_layers)
        ])

        # # MLP方案
        # mlps = []
        # mlps.append(nn.Sequential(
        #     nn.Linear(dim * 2, dim * hidden_ratio),
        #     nn.GELU(),
        #     nn.Linear(dim * hidden_ratio, dim),
        #     nn.LayerNorm(dim, eps=1e-6)
        # ))
        #
        # for _ in range(num_layers - 1):
        #     mlps.append(nn.Sequential(
        #         nn.Linear(dim, dim * hidden_ratio),
        #         nn.GELU(),
        #         nn.Linear(dim * hidden_ratio, dim),
        #         nn.LayerNorm(dim, eps=1e-6)
        #     ))
        # self.MLP = nn.Sequential(*mlps)
        # self.gate = nn.Parameter(torch.ones(1, 1, dim))

    def forward(self, x_v, x_t, n_class):
        """
        args:
            x_v: [B, C, T_v]
            x_t: [B, C, T_t]
        """
        # 交叉注意力方案
        x_v = x_v.permute(0, 2, 1)  # BCT -> BTC
        x_t = x_t.permute(0, 2, 1)  # BCT -> BTC
        x_t = self.proj_text(x_t)
        # x_t = x_t.permute(0, 2, 1)  # BTC -> BCT

        for blk in self.layers:
            x_v = blk(x_v, x_t)

        x_v = x_v.permute(0, 2, 1)  # BTC -> BCT
        x_t = x_t.permute(0, 2, 1)  # BTC -> BCT

        # 多层MLP方案
        # x_v = x_v.permute(0, 2, 1)  # BCT -> BTC
        # x_t = x_t.permute(0, 2, 1)  # BCT -> BTC
        # x_v = self.proj_visual(x_v)
        # x_t = self.proj_text(x_t)
        #
        # fused = torch.cat([x_v, x_t], dim=-1)
        # fused = self.MLP(fused)
        # x_v = x_v + fused * self.gate
        # x_v = x_v.permute(0, 2, 1)  # BTC -> BCT

        # x_v = x_v + x_t   # 直接相加方案

        # 另一位同学的可学习嵌入
        # b, t, c = x_v.shape
        # class_embedding = nn.Embedding(n_class, c).cuda()
        # nn.init.normal_(class_embedding.weight, std=0.02)
        # class_embeds_expanded = class_embedding.unsqueeze(1).repeat(b, t, 1)
        return x_v
