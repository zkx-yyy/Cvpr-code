import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp


class SeqAttention(nn.Module):
    """
    Temporal-SE over T (explicit w_t) + residual + MLP.
    Input/Output: [B, T, C]
    """
    def __init__(self,
                 dim=768,
                 reduction=16,
                 k=7,
                 mlp_ratio=4,
                 dropout=0.0):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

        hidden = max(4, dim // reduction)
        pad = k // 2
        # Excite over temporal axis (on squeezed signal [B,1,T])
        self.t_se = nn.Sequential(
            nn.Conv1d(1, hidden, kernel_size=k, padding=pad),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, 1, kernel_size=k, padding=pad),
            nn.Sigmoid()
        )

        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, dim),
        )

    def forward(self, x, return_w=False):
        """
        x: [B, T, C]
        """
        # 时序SE
        x1 = self.norm1(x)                          # [B, T, C]
        s = x1.mean(dim=-1, keepdim=True)           # [B, T, 1]
        s = s.permute(0, 2, 1).contiguous()         # [B, 1, T]
        w = self.t_se(s).squeeze(1)                  # [B, T]
        x_se = x1 * w.unsqueeze(-1)                 # [B, T, C]
        x = x + self.dropout(x_se)                  # BTC

        # MLP
        x2 = self.norm2(x)
        x = x + self.dropout(self.mlp(x2))

        if return_w:
            return x, w
        return x


class SeqAttnModule(nn.Module):
    def __init__(self,
                 dim=768,
                 mlp_ratio=4,
                 num_layers=3,):
        super().__init__()
        self.layers = nn.ModuleList([
            SeqAttention(
                dim=dim,
                mlp_ratio=mlp_ratio,
            )
            for _ in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class SSA(nn.Module):
    def __init__(self,
                 dim=768,
                 num_heads=12,
                 mlp_ratio=4,
                 with_cp=False,
                 num_layers=3,
                 num_img_tokens=197,
                 num_frames=6,
                 tau=0.07,
                 lambda_bg=0.2,
                 raw_dim=3584,
                 max_T=4096,
                 pe_init_std=0.02,
                 pooled_dir=None
                 ):
        super().__init__()

        self.num_img_tokens = num_img_tokens
        self.num_frames = num_frames
        self.lambda_bg = lambda_bg
        self.dim = dim
        self.raw_dim = raw_dim
        self.max_T = max_T

        self.pooled_dir = pooled_dir or "data/thumos-14/qwen/frames_pooled"
        # A directory list selects S2; one directory retains S0/S1 behavior.
        self.layer_dirs = tuple(self.pooled_dir) if isinstance(self.pooled_dir, (list, tuple)) else ()
        if self.layer_dirs:
            if len(self.layer_dirs) != 4 or len(set(self.layer_dirs)) != 4:
                raise ValueError("S2 requires four distinct feature directories in H7/H14/H21/H28 order")
            for root in self.layer_dirs:
                if not os.path.isdir(root):
                    raise FileNotFoundError(root)
            self.layer_logits = nn.Parameter(torch.zeros(len(self.layer_dirs)))

        self.proj_in = nn.Linear(raw_dim, dim, bias=True)

        self.pe = nn.Parameter(torch.zeros(1, max_T, dim))
        nn.init.normal_(self.pe, mean=0.0, std=pe_init_std)
        # PE scale：可学习标量，初始很小，避免一上来用PE把语义锚点冲烂
        self.pe_scale = nn.Parameter(torch.tensor(5.0))   # 0.1太小,5.0?10.0?

        self.SeqAttn_layer = SeqAttnModule(
            dim=self.dim,
            mlp_ratio=mlp_ratio,
            num_layers=num_layers,
        )

        self._iter = 0
        self.iters_per_epoch = 107  # ceil(213/2)

    """
    读取不同语义的锚点
    """
    # 视频级
    # def build_BTD_from_metas(self, metas, T, device=None, dtype=None, repeat_mode: str = "expand"):
    #     feats = []
    #     names = []
    #     feat_dim = None
    #
    #     for meta in metas:
    #         name = meta["video_name"]
    #         feat = self._load_pooled_feat(name)  # [D], cpu
    #
    #         if feat_dim is None:
    #             feat_dim = feat.shape[0]
    #         else:
    #             assert feat.shape[0] == feat_dim, f"Dim mismatch: {name} has {feat.shape[0]}, expect {feat_dim}"
    #
    #         feats.append(feat)
    #         names.append(name)
    #
    #     video_BD = torch.stack(feats, dim=0)  # [B, D]
    #
    #     # move / cast once
    #     if dtype is not None:
    #         video_BD = video_BD.to(dtype=dtype)
    #     if device is not None:
    #         video_BD = video_BD.to(device=device, non_blocking=True)
    #
    #     video_B1D = video_BD.unsqueeze(1)  # [B,1,D]
    #     if repeat_mode == "repeat":
    #         video_BTD = video_B1D.repeat(1, T, 1)
    #     else:
    #         video_BTD = video_B1D.expand(-1, T, -1)
    #
    #     return video_BTD

    # 帧级
    def build_BTD_from_metas(self, metas, T, device=None, dtype=None, repeat_mode: str = "expand"):
        """
        metas: list[dict], each contains:
          - meta["video_name"]
          - meta["inds"] or meta["frame_inds"]  (length == T)
        return: video_BTD [B, T, raw_dim]  (raw_dim=D in packed.pt)
        """
        b_feats = []

        for meta in metas:
            video_name = meta["video_name"]
            inds = meta.get("frame_inds", None)

            if not torch.is_tensor(inds):    # inds: list/np/tensor -> tensor[int64]
                inds = torch.as_tensor(inds)
            inds = inds.to(torch.long)

            # 可选：去掉末尾 padding 的“连续重复”, 如想完全保留重复帧，把下面这行注释掉
            # inds = torch.unique_consecutive(inds)

            # 读取LLM pack
            packs = [self._load_video_pack(video_name, root) for root in self.layer_dirs] if self.layer_dirs else [self._load_video_pack(video_name)]
            pack = packs[0]
            frame_idx = pack["frame_idx"].to(torch.long)  # [N]
            pooled = pack["pooled"]  # [N, D]
            for other in packs[1:]:
                if (not torch.equal(pack["frame_idx"], other["frame_idx"])
                        or pooled.shape != other["pooled"].shape
                        or pooled.dtype != other["pooled"].dtype
                        or pack.get("prompt") != other.get("prompt")):
                    raise ValueError(f"{video_name}: inconsistent layer frame indices, shape, dtype or prompt")

            if inds.numel() != T:
                raise ValueError(f"{video_name}: expected {T} frame indices, got {inds.numel()}")
            inds = inds.reshape(-1)
            right = torch.searchsorted(frame_idx, inds).clamp(max=frame_idx.numel() - 1)
            left = (right - 1).clamp(min=0)
            left_distance = (inds - frame_idx[left]).abs()
            right_distance = (frame_idx[right] - inds).abs()
            pos = torch.where(left_distance <= right_distance, left, right)
            if self.layer_dirs:
                # Keep fusion in autograd; match the offline S1 mean's rounding.
                raw = torch.stack([p["pooled"][pos] for p in packs]).to(self.layer_logits.device)
                weights = self.layer_logits.float().softmax(dim=0)
                feat_TD = (raw.float() * weights[:, None, None]).sum(dim=0).to(pooled.dtype)
            else:
                feat_TD = pooled[pos]  # [T, D]
            b_feats.append(feat_TD)

        video_BTD = torch.stack(b_feats, dim=0)  # [B, T, D]

        if dtype is not None:
            video_BTD = video_BTD.to(dtype=dtype)
        if device is not None:
            video_BTD = video_BTD.to(device=device, non_blocking=True)

        return video_BTD

    def forward(self, metas, t, valid_mask=None):
        """
        metas: list[dict] each has "video_name"
        T: int
        """
        # 先从 pooled_dir 构造 BTD
        last_hidden_state = self.build_BTD_from_metas(
            metas=metas,
            T=t,
            device=self.proj_in.weight.device,      # e.g., torch.device("cuda")
            dtype=None,       # e.g., torch.bfloat16
            repeat_mode="expand",
        )  # last_hidden_state: [B, T, D]

        last_hidden_state = last_hidden_state.to(dtype=self.proj_in.weight.dtype)
        x = self.proj_in(last_hidden_state)   # BTC

        # ====== epoch-level debug (print once per epoch) ======
        self._iter += 1
        if self.layer_dirs and self.training and (self._iter == 1 or self._iter % 100 == 0):
            if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                print(f"[S2-LAYER-WEIGHTS] train_forward={self._iter} "
                      f"weights={self.layer_logits.detach().float().softmax(0).cpu().tolist()}", flush=True)
        epoch_idx = (self._iter - 1) // self.iters_per_epoch
        is_epoch_first_iter = ((self._iter - 1) % self.iters_per_epoch == 0)

        if t > self.max_T:
            raise ValueError(f"T={t} exceeds max_T={self.max_T}. Increase max_T in SSA.__init__.")

        # slice PE once
        pe_slice = self.pe[:, :t, :]  # [1, T, dim]

        # 打印统计信息,需要时开启
        # if self.training and is_epoch_first_iter:
        #     with torch.no_grad():
        #         # (1) 原始 x 分布（before PE）
        #         x_mean = x.mean().item()
        #         x_std = x.std().item()
        #         x_abs = x.abs().mean().item()
        #
        #         # (2) PE 分布（raw）
        #         pe_mean = pe_slice.mean().item()
        #         pe_std = pe_slice.std().item()
        #         pe_abs = pe_slice.abs().mean().item()
        #
        #         print(
        #             f"[SSA][epoch {epoch_idx}][before PE] "
        #             f"mean={x_mean:.4e}, std={x_std:.4e}, |x|={x_abs:.4e}, "
        #             f"pe_scale={self.pe_scale.item():.4e}"
        #         )
        #         print(
        #             f"[SSA][epoch {epoch_idx}][PE raw] "
        #             f"mean={pe_mean:.4e}, std={pe_std:.4e}, |pe|={pe_abs:.4e}"
        #         )

        # (3) 加 PE
        x_pe = x + self.pe_scale * pe_slice  # [B, T, C]  self.pe_scale * pe_slice

        # 打印统计信息,需要时可开启
        # if self.training and is_epoch_first_iter:
        #     with torch.no_grad():
        #         # 看“不同时间步是否真的分化”：||x0-xT|| / ||x0||
        #         diff = (x_pe[:, 0, :] - x_pe[:, -1, :]).norm(dim=-1).mean().item()
        #         base = x_pe[:, 0, :].norm(dim=-1).mean().item()
        #         ratio = diff / (base + 1e-6)
        #         print(
        #             f"[SSA][epoch {epoch_idx}][after PE] "
        #             f"||x0-xT||={diff:.4e}, ratio={ratio:.4e}"
        #         )

        if valid_mask is not None:
            valid_mask = valid_mask.to(device=x_pe.device, dtype=x_pe.dtype)
            if valid_mask.shape != x_pe.shape[:2]:
                raise ValueError(f"valid_mask must be {tuple(x_pe.shape[:2])}, got {tuple(valid_mask.shape)}")
            x_pe = x_pe * valid_mask.unsqueeze(-1)

        SeqAttn = self.SeqAttn_layer(x_pe)
        if valid_mask is not None:
            SeqAttn = SeqAttn * valid_mask.unsqueeze(-1)
        return SeqAttn


    def _load_pooled_feat(self, video_name: str) -> torch.Tensor:
        """
        Load pooled_last_hidden from .pt file.
        Expect: dict with key 'pooled_last_hidden'
        Return: Tensor [D]
        """
        p = os.path.join(self.pooled_dir, f"{video_name}.pt")
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing pooled feature: {p}")

        obj = torch.load(p, map_location="cpu")

        assert isinstance(obj, dict), \
            f"{video_name}: expect dict in pt file, got {type(obj)}"
        assert "pooled_last_hidden" in obj, \
            f"{video_name}: 'pooled_last_hidden' not found in {p}"

        feat = obj["pooled_last_hidden"]  # Tensor

        # (1, D) -> (D)
        if feat.ndim == 2 and feat.shape[0] == 1:
            feat = feat.squeeze(0)

        assert feat.ndim == 1, \
            f"{video_name}: expect [D], got {tuple(feat.shape)}"

        return feat

    def _load_video_pack(self, video_name: str, root=None):
        # 上游保存路径是: save_root / stem / "packed.pt"  :contentReference[oaicite:2]{index=2}
        p = os.path.join(self.pooled_dir if root is None else root, video_name, "packed.pt")
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing packed.pt: {p}")
        pack = torch.load(p, map_location="cpu")
        assert "frame_idx" in pack and "pooled" in pack, f"{p} missing keys, got {list(pack.keys())}"
        return pack  # {"frame_idx": [N], "pooled": [N,D], ...}
