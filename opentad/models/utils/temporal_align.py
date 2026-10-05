import torch
import numpy as np

@torch.no_grad()
def label2frames(
    x,                       # Tensor [B,1,C,T,H,W]
    gt_segments,             # list[B]{Tensor [N_b,2]}
    gt_ids,                  # list[B]{Tensor [N_b]}
    gt_labels_text,          # list[B]{list[str]}，len=N_b
    null_token="[NULL]",
    strict_inside=True,      # True: (t-s>0 & e-t>0)，False: 含边界
):
    assert x.dim() == 6, f"Unexpected x.shape: {tuple(x.shape)}"
    T = x.size(3)
    device = x.device

    t = torch.arange(T, device=device, dtype=torch.float32)  # [0..T-1]

    aligned_texts, aligned_idx = [], []

    for segs, texts in zip(gt_segments, gt_labels_text):
        # --- 允许空窗：N==0 直接填占位并继续 ---
        if segs is None:
            N = 0
        else:
            # 支持 numpy / tensor
            if isinstance(segs, np.ndarray):
                N = segs.shape[0]
            elif torch.is_tensor(segs):
                N = segs.size(0)
            else:
                N = len(segs)

        if N == 0:
            aligned_texts.append([null_token] * T)
            aligned_idx.append(torch.full((T,), -1, dtype=torch.long, device=device))
            continue
        # --- 正常分支 ---

        if isinstance(segs, np.ndarray):
            segs = torch.from_numpy(segs)
        segs = segs.to(device=device, dtype=torch.float32)

        # clamp 到合法范围
        segs = torch.stack([
            segs[:, 0].clamp(0, float(T)),
            segs[:, 1].clamp(0, float(T))
        ], dim=1)

        # 逐帧到段边界的距离与内点判断
        left = t[:, None] - segs[None, :, 0]  # [T,N]
        right = segs[None, :, 1] - t[:, None]  # [T,N]
        inside = (left > 0) & (right > 0) if strict_inside else (left >= 0) & (right >= 0)

        # “最短时长优先”选择段
        lens = (segs[:, 1] - segs[:, 0])[None, :].repeat(T, 1)  # [T,N]
        lens.masked_fill_(~inside, float("inf"))

        min_len, min_idx = lens.min(dim=1)  # [T], [T]
        pos = torch.isfinite(min_len)  # 哪些帧命中了某个段

        # 初始化逐帧文本与索引
        per_t_text = [null_token] * T
        per_t_idx = torch.full((T,), -1, dtype=torch.long, device=device)

        # 映射命中的帧 → 文本
        if pos.any():
            ts = pos.nonzero(as_tuple=False).squeeze(1).tolist()
            idx = min_idx[pos].tolist()
            # 防御：texts 可能比 N 少（极少见），做边界判断
            for tt, gi in zip(ts, idx):
                if 0 <= gi < len(texts):
                    per_t_text[tt] = texts[gi]
                    per_t_idx[tt] = gi

        aligned_texts.append(per_t_text)
        aligned_idx.append(per_t_idx)   # aligned_idx表示长视频里的第几段动作

    # # frame_label_ids for infoNCE_L2
    # frame_label_ids = []
    # for idx in aligned_idx:
    #     frame_label_ids.append((idx != -1).long())  # 每个样本 [T]
    #
    # frame_label_ids = torch.stack(frame_label_ids, dim=0)  # [B, T] 区分动作与背景

    # Foreground frames retain their action category; zero remains background.
    frame_class_ids = []
    for idx, ids in zip(aligned_idx, gt_ids):
        per_t_ids = torch.zeros_like(idx, dtype=torch.long)
        for i, class_id in enumerate(ids):
            per_t_ids[idx == i] = class_id.long() + 1
        frame_class_ids.append(per_t_ids)

    frame_class_ids = torch.stack(frame_class_ids, dim=0)

    return aligned_texts, frame_class_ids
