import torch
import torch.nn as nn
import os
from contextlib import nullcontext
from ..utils.temporal_align import label2frames

from ..builder import DETECTORS
from .single_stage import SingleStageDetector
from ..bricks import Scale, AffineDropPath
from ..necks.tirm import TIR_Block


@DETECTORS.register_module()
class ActionFormer(SingleStageDetector):
    def __init__(
        self,
        projection,
        rpn_head,
        neck=None,
        backbone=None,
        task_semantic_mode=None,
        # git_backbone=None,
    ):
        super().__init__(
            backbone=backbone,
            neck=neck,
            projection=projection,
            rpn_head=rpn_head,
        )

        # channels = projection.out_channels    # 颈部改进,非当前重点,先搞定多模态/融合,再做颈部凑数创新
        # self.tirm = TIR_Block(channels)

        n_mha_win_size = self.projection.n_mha_win_size
        if isinstance(n_mha_win_size, int):
            self.mha_win_size = [n_mha_win_size] * (1 + projection.arch[-1])
        else:
            assert len(n_mha_win_size) == (1 + projection.arch[-1])
            self.mha_win_size = n_mha_win_size
        self.max_seq_len = self.projection.max_seq_len

        max_div_factor = 1
        for s, w in zip(rpn_head.prior_generator.strides, self.mha_win_size):
            stride = s * (w // 2) * 2 if w > 1 else s
            assert (
                self.max_seq_len % stride == 0
            ), f"max_seq_len {self.max_seq_len} must be divisible by fpn stride and window size {stride}"
            if max_div_factor < stride:
                max_div_factor = stride
        self.max_div_factor = max_div_factor
        self.task_semantic_mode = task_semantic_mode
        if task_semantic_mode is not None:
            if task_semantic_mode not in ('hard', 'shared'):
                raise ValueError(f'Unsupported task semantic mode: {task_semantic_mode}')
            ssa = self.backbone.SSA
            if [os.path.basename(p.rstrip('/')) for p in ssa.layer_dirs] != ['layer_7', 'layer_14', 'layer_21', 'layer_28']:
                raise ValueError('Feature directory order must be H7/H14/H21/H28')
            if task_semantic_mode == 'hard':
                del ssa.layer_logits
            self._last_task_epoch = None

    def _task_weights(self):
        ssa = self.backbone.SSA
        if self.task_semantic_mode == 'hard':
            return ssa.proj_in.weight.new_tensor([[0., 0., 0., 1.], [0., 0., 1., 0.]])
        return ssa.layer_logits.float().softmax(0).expand(2, -1)

    def _task_raw(self, metas, length):
        ssa = self.backbone.SSA
        samples = []
        for meta in metas:
            packs = [ssa._load_video_pack(meta['video_name'], root) for root in ssa.layer_dirs]
            first = packs[0]
            for pack in packs[1:]:
                if (not torch.equal(first['frame_idx'], pack['frame_idx'])
                        or first['pooled'].shape != pack['pooled'].shape
                        or first['pooled'].dtype != pack['pooled'].dtype
                        or first.get('prompt') != pack.get('prompt')):
                    raise ValueError(f"Unaligned semantic layers: {meta['video_name']}")
            idx = first['frame_idx'].long()
            inds = torch.as_tensor(meta['frame_inds'], dtype=torch.long, device='cpu').reshape(-1)
            if inds.numel() != length:
                raise ValueError('Semantic and visual sequence lengths differ')
            right = torch.searchsorted(idx, inds).clamp(max=idx.numel() - 1)
            left = (right - 1).clamp(min=0)
            pos = torch.where((inds - idx[left]).abs() <= (idx[right] - inds).abs(), left, right)
            samples.append(torch.stack([p['pooled'][pos] for p in packs]))
        raw = torch.stack(samples).to(ssa.proj_in.weight.device)  # B,L,T,D
        return [(raw.float() * w[None, :, None, None]).sum(1).to(raw.dtype) for w in self._task_weights()]

    def _task_visual(self, inputs):
        b = self.backbone
        b.set_norm_layer()
        frames, _ = b.model.data_preprocessor.preprocess(b.tensor_to_list(inputs), data_samples=None, training=False)
        if b.pre_processing_pipeline is not None:
            frames = b.pre_processing_pipeline(dict(frames=frames))['frames']
        batches, num_segs = frames.shape[:2]
        frames = frames.flatten(0, 1).contiguous()
        with torch.no_grad() if b.freeze_backbone else nullcontext():
            if b.use_temporal_checkpointing:
                features = b.temporal_checkpointing(frames, b.temporal_checkpointing_chunk_num, b.temporal_checkpointing_chunk_dim)
            else:
                features = b.model.backbone(frames)
        if isinstance(features, (tuple, list)):
            return torch.cat([b.unflatten_and_pool_features(f, batches, num_segs) for f in features], dim=1)
        return b.unflatten_and_pool_features(features, batches, num_segs)

    def _task_path(self, visual, raw, masks, n_class):
        b = self.backbone
        ssa = b.SSA
        length = raw.shape[1]
        if length > ssa.max_T:
            raise ValueError('Semantic sequence exceeds positional embedding length')
        caption = ssa.proj_in(raw.to(ssa.proj_in.weight.dtype))
        caption = caption + ssa.pe_scale * ssa.pe[:, :length]
        # Preserve S0/S1/S2's unmasked backbone semantic processing.
        caption = ssa.SeqAttn_layer(caption)
        fused = b.verb_guided_cross_attention(visual, caption.permute(0, 2, 1), n_class).float()
        x, out_masks = self.pad_data(fused, masks)
        if self.with_projection:
            x, out_masks = self.projection(x, out_masks)
        if self.with_neck:
            x, out_masks = self.neck(x, out_masks)
        return x, out_masks, caption

    def _task_features(self, inputs, masks, metas, **kwargs):
        visual = self._task_visual(inputs)
        raw_cls, raw_loc = self._task_raw(metas, visual.shape[-1])
        # Both paths share stochastic masks; advance RNG only once.
        cpu_state = torch.get_rng_state()
        device = visual.device
        cuda_state = torch.cuda.get_rng_state(device) if device.type == 'cuda' else None
        cls = self._task_path(visual, raw_cls, masks, kwargs.get('n_class'))
        with torch.random.fork_rng(devices=[device.index] if cuda_state is not None else []):
            torch.set_rng_state(cpu_state)
            if cuda_state is not None:
                torch.cuda.set_rng_state(cuda_state, device)
            loc = self._task_path(visual, raw_loc, masks, kwargs.get('n_class'))
        if any(not torch.equal(a, b) for a, b in zip(cls[1], loc[1])):
            raise RuntimeError('Classification and regression masks differ')
        return visual, cls, loc

    def _task_predictions(self, cls_features, loc_features, masks):
        head = self.rpn_head
        cls_pred, reg_pred = [], []
        for level, (cls, loc, mask) in enumerate(zip(cls_features, loc_features, masks)):
            cm, lm = mask, mask
            for i in range(head.num_convs):
                cls, cm = head.cls_convs[i](cls, cm)
                loc, lm = head.reg_convs[i](loc, lm)
            cls_pred.append(head.cls_head(cls))
            reg_pred.append(torch.nn.functional.relu(head.scale[level](head.reg_head(loc))))
        return cls_pred, reg_pred

    def _task_train(self, inputs, masks, metas, gt_segments, gt_labels, gt_labels_text, **kwargs):
        visual, cls, loc = self._task_features(inputs, masks, metas, **kwargs)
        cls_pred, reg_pred = self._task_predictions(cls[0], loc[0], cls[1])
        points = self.rpn_head.prior_generator(cls[0])
        losses = self.rpn_head.losses(cls_pred, reg_pred, cls[1], points, gt_segments, gt_labels)
        b = self.backbone
        epoch = kwargs.get('curr_epoch', 0)
        if b.use_infonce and b.infonce_loss_weight > 0:
            _, ids = label2frames(inputs, gt_segments, gt_labels, gt_labels_text)
            warmup = min(1., float(epoch + 1) / b.infonce_warmup_epochs) if b.infonce_warmup_epochs > 0 else 1.
            aux = [b.loss_infonce(visual.permute(0, 2, 1).float(), branch[2].float(),
                                class_ids=ids.to(visual.device), valid_mask=None,
                                lambda_bg=b.SSA.lambda_bg) for branch in (cls, loc)]
            losses['infonce_loss'] = (aux[0] + aux[1]) * (0.5 * b.infonce_loss_weight * warmup)
        else:
            losses['infonce_loss'] = visual.sum() * 0.
        if self._last_task_epoch != epoch:
            if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                print(f'[TASK-ROUTING] mode={self.task_semantic_mode} epoch={epoch} cls_loc_weights={self._task_weights().detach().cpu().tolist()}', flush=True)
            self._last_task_epoch = epoch
        losses['cost'] = sum(losses.values())
        return losses

    def pad_data(self, inputs, masks):
        feat_len = inputs.shape[-1]
        if feat_len == self.max_seq_len:
            return inputs, masks
        elif feat_len < self.max_seq_len:
            max_len = self.max_seq_len
        else:  # feat_len > self.max_seq_len
            max_len = feat_len
            # pad the input to the next divisible size
            stride = self.max_div_factor
            max_len = (max_len + (stride - 1)) // stride * stride

        padding_size = [0, max_len - feat_len]
        inputs = torch.nn.functional.pad(inputs, padding_size, value=0)
        pad_masks = torch.zeros((inputs.shape[0], max_len), device=masks.device).bool()
        pad_masks[:, :feat_len] = masks
        return inputs, pad_masks

    def forward_train(self, inputs, masks, metas, gt_segments, gt_labels, gt_labels_text, **kwargs): # for txt : gt_labels_text
        if self.task_semantic_mode is not None:
            return self._task_train(inputs, masks, metas, gt_segments, gt_labels, gt_labels_text, **kwargs)
        losses = dict()
        if self.with_backbone:
            # out = self.backbone(inputs,  gt_segments, gt_labels, gt_labels_text, metas, **kwargs)       # 调取的是backbone_wrapper，封装调取的是vit_adapter, return BCT  # for txt, kwargs生成类别矩阵
            # x = out["features"]    # GIT版本用此上两行
            x = self.backbone(inputs, gt_segments, gt_labels, gt_labels_text, metas, **kwargs)
        else:
            x = inputs

        if isinstance(x, dict):
            if "infonce_loss" in x:
                losses["infonce_loss"] = x["infonce_loss"]
            x = x["features"]

        # pad the features and unsqueeze the mask for actionformer
        x, masks = self.pad_data(x, masks)           # 检查序列长度是否符合要求，否则进行填充

        if self.with_projection:
            x, masks = self.projection(x, masks)   # BCT

        if self.with_neck:
            x, masks = self.neck(x, masks) # 原始颈部
            # x = self.tirm(x)     # 颈部改进,非当前重点,先搞定多模态/融合,再做颈部凑数创新

        loc_losses = self.rpn_head.forward_train(
            x,
            masks,
            gt_segments=gt_segments,
            gt_labels=gt_labels,
            gt_labels_text=gt_labels_text, # for txt
            **kwargs,
        )
        losses.update(loc_losses)

        # only key has loss will be record
        losses["cost"] = sum(_value for _key, _value in losses.items())  # 关系检测loss是否影响SSA, _value for _key, _value in losses.items() if "infonce" not in _key
        return losses

    def forward_test(self, inputs, masks, metas=None, infer_cfg=None, gt_segments=None, gt_labels=None, gt_labels_text=None, **kwargs): # for txt : gt_segments, gt_labels, gt_labels_text,
        if self.task_semantic_mode is not None:
            _, cls, loc = self._task_features(inputs, masks, metas, **kwargs)
            cls_pred, reg_pred = self._task_predictions(cls[0], loc[0], cls[1])
            points = self.rpn_head.prior_generator(cls[0])
            return self.rpn_head.get_valid_proposals_scores(points, reg_pred, cls_pred, cls[1])
        if self.with_backbone:
            x = self.backbone(inputs, gt_segments, gt_labels, gt_labels_text, metas, **kwargs)  # for txt : gt_segments, gt_labels_text, kwargs生成类别矩阵, metas for qwen
        else:
            x = inputs

        x, masks = self.pad_data(x, masks)

        if self.with_projection:
            x, masks = self.projection(x, masks)

        if self.with_neck:
            x, masks = self.neck(x, masks)

        rpn_proposals, rpn_scores = self.rpn_head.forward_test(x, masks, **kwargs)
        predictions = rpn_proposals, rpn_scores
        return predictions

    def get_optim_groups(self, cfg):
        # separate out all parameters that with / without weight decay
        # see https://github.com/karpathy/minGPT/blob/master/mingpt/model.py#L134
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (nn.Linear, nn.Conv1d)
        blacklist_weight_modules = (nn.LayerNorm, nn.GroupNorm,  nn.BatchNorm1d) # !

        # loop over all modules / params
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name

                # exclude the backbone parameters
                if fpn.startswith("backbone"):
                    continue

                if pn.endswith("bias"):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)
                elif pn.endswith("scale") and isinstance(m, (Scale, AffineDropPath)):
                    # corner case of our scale layer
                    no_decay.add(fpn)
                elif pn.endswith("rel_pe"):
                    # corner case for relative position encoding
                    no_decay.add(fpn)

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters() if not pn.startswith("backbone")}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert (
            len(param_dict.keys() - union_params) == 0
        ), "parameters %s were not separated into either decay/no_decay set!" % (str(param_dict.keys() - union_params),)

        # create the pytorch optimizer object
        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": cfg["weight_decay"],
                "lr": cfg["lr"],
            },
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0, "lr": cfg["lr"]},
        ]
        return optim_groups


"""
打印lora梯度,监控训练信息，现已弃用微lora调方案
"""
# def monitor_lora_gradients(model, keyword="lora"):
#     """
#     检查 LoRA 参数的梯度流动情况。
#     Args:
#         model: 含 LoRA 模块的模型（如 self.git_backbone）
#         keyword: 匹配 LoRA 参数名的关键词（通常为 "lora"）
#     """
#     grad_norms = []
#     nonzero_count = 0
#     total_count = 0
#
#     for name, param in model.named_parameters():
#         if keyword in name and param.requires_grad:
#             total_count += 1
#             if param.grad is not None:
#                 norm = param.grad.data.norm().item()
#                 grad_norms.append(norm)
#                 if norm > 1e-9:
#                     nonzero_count += 1
#
#     if len(grad_norms) == 0:
#         print("[LoRA Grad Monitor] ❌ No LoRA params found or requires_grad=False")
#     else:
#         mean_norm = sum(grad_norms) / len(grad_norms)
#         ratio = nonzero_count / total_count * 100
#         print(f"[LoRA Grad Monitor] mean_grad_norm={mean_norm:.6e}, nonzero_ratio={ratio:.1f}% ({nonzero_count}/{total_count})")
