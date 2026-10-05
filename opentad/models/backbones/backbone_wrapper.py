import copy
import importlib.util
import os
import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm
import torch.utils.checkpoint as cp

from mmengine.dataset import Compose
from mmengine.registry import MODELS as MM_BACKBONES
from mmengine.runner import load_checkpoint
from opentad.models.utils.temporal_align import label2frames    # for txt
from opentad.models.backbones.textguid_attn import VTModuel
from opentad.models.utils.SSA_qwen import SSA


def _load_info_nce_loss():
    loss_path = os.path.join(os.path.dirname(__file__), "..", "losses", "masked_infoNCE_loss.py")
    spec = importlib.util.spec_from_file_location("opentad_masked_infoNCE_loss", loss_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.InfoNCELoss


InfoNCELoss = _load_info_nce_loss()

BACKBONES = MM_BACKBONES

"""
VLM/MLLM部分，git模型为初期尝试,涉及git的均弃用,当前版本使用qwen-2.5VL-7B预提取的帧级/视频级last hidden state
"""

class BackboneWrapper(nn.Module):    # wrapper封装
    def __init__(self, cfg):
        super(BackboneWrapper, self).__init__()
        custom_cfg = cfg.custom
        # git_cfg = cfg.git_backbone
        model_cfg = copy.deepcopy(cfg)
        model_cfg.pop("custom")
        model_cfg.pop("git_backbone", None)

        # build the backbone
        self.model = BACKBONES.build(model_cfg)
        # self.git_backbone = GITFeatureExtractor(git_cfg)

        # custom settings: pretrained checkpoint, post_processing_pipeline, norm_eval, freeze_backbone
        # 1. load the pretrained model
        if hasattr(custom_cfg, "pretrain") and custom_cfg.pretrain is not None:
            load_checkpoint(self.model, custom_cfg.pretrain, map_location="cpu")
        else:
            print(
                "Warning: no pretrain path is provided, the backbone will be randomly initialized, "
                "unless you have initialized the weights in the model.py."
            )

        # 2. pre_processing_pipeline
        if hasattr(custom_cfg, "pre_processing_pipeline"):
            self.pre_processing_pipeline = Compose(custom_cfg.pre_processing_pipeline)
            # self.pre_processing_pipeline_text = Compose(custom_cfg.pre_processing_pipeline_text)
        else:
            self.pre_processing_pipeline = None

        # 3. post_processing_pipeline for pooling and other operations
        if hasattr(custom_cfg, "post_processing_pipeline"):
            self.post_processing_pipeline = Compose(custom_cfg.post_processing_pipeline)
            # self.post_processing_pipeline_text = Compose(custom_cfg.post_processing_pipeline_text)
        else:
            self.post_processing_pipeline = None

        # 4. norm_eval: set all norm layers to eval mode
        self.norm_eval = getattr(custom_cfg, "norm_eval", True)

        # 5. freeze_backbone: whether to freeze the backbone, default is False
        self.freeze_backbone = getattr(custom_cfg, "freeze_backbone", False)

        print("freeze_backbone: {}, norm_eval: {}".format(self.freeze_backbone, self.norm_eval))

        # 6. whether to use temporal activation checkpointing
        self.use_temporal_checkpointing = getattr(custom_cfg, "temporal_checkpointing", False)
        if self.use_temporal_checkpointing:
            assert hasattr(
                custom_cfg, "temporal_checkpointing_chunk_num"
            ), "temporal_checkpointing_chunk_num should be provided when using temporal checkpointing"
            assert hasattr(
                custom_cfg, "temporal_checkpointing_chunk_dim"
            ), "temporal_checkpointing_chunk_dim should be provided when using temporal checkpointing"
            self.temporal_checkpointing_chunk_num = custom_cfg.temporal_checkpointing_chunk_num
            self.temporal_checkpointing_chunk_dim = custom_cfg.temporal_checkpointing_chunk_dim

        # 7. verb guided cross attention
        self.verb_guided_cross_attention = VTModuel(
            dim=model_cfg["backbone"]["embed_dims"],
            num_heads=model_cfg["backbone"]["num_heads"],
            mlp_ratio=model_cfg["backbone"]["mlp_ratio"],
            with_cp=model_cfg["backbone"]["with_cp"],
            fusion_init_scale=getattr(custom_cfg, "fusion_init_scale", 0.1),
            # dropout=model_cfg["backbone"]["drop_path_rate"],  ### !
        )

        self.infonce_loss_weight = getattr(custom_cfg, "infonce_loss_weight", 0.5)
        self.infonce_warmup_epochs = getattr(custom_cfg, "infonce_warmup_epochs", 0)
        self.use_infonce = getattr(custom_cfg, "use_infonce", True)
        self.loss_infonce = InfoNCELoss(
            dim=model_cfg["backbone"]["embed_dims"],
            temperature=getattr(custom_cfg, "infonce_temperature", 0.07),
        )
        self.SSA = SSA(
            dim=model_cfg["backbone"]["embed_dims"],
            num_heads=model_cfg["backbone"]["num_heads"],
            mlp_ratio=model_cfg["backbone"]["mlp_ratio"],
            with_cp=model_cfg["backbone"]["with_cp"],
            pooled_dir=getattr(custom_cfg, "qwen_feature_dir", None),
            lambda_bg=getattr(custom_cfg, "infonce_lambda_bg", 0.2),
            raw_dim=getattr(custom_cfg, "qwen_raw_dim", 3584),
        )

    def forward(self, frames, segments, gt_ids, text, metas=None, masks=None, **kwargs):  # "segments, text" for txt, kwargs生成类别矩阵, metas读取视频ID
        # two types: snippet or frame

        # snippet: 3D backbone, [bs, T, 3, clip_len, H, W]
        # frame: 3D backbone, [bs, 1, 3, T, H, W]

        # set all normalization layers
        self.set_norm_layer()
                                                                                  # 768=48*16,将一个视频分为48个chunk，每个chunk16帧
        # data preprocessing: normalize mean and std                 # frame（2, 1, 3, 768, 160, 160）
        frames, _ = self.model.data_preprocessor.preprocess(         # 这一步做了Normalization
            self.tensor_to_list(frames),  # need list input
            data_samples=None,
            training=False,  # for blending, which is not used in openTAD
        )
        # text_frames = copy.deepcopy(frames)
        _, frame_class_ids = label2frames(frames, segments, gt_ids, text)

        # pre_processing_pipeline:                                                   # （2, 1, 3, 768, 160, 160）
        if self.pre_processing_pipeline is not None:                                 # b n c (t1 t) h w -> (b t1) n c t h w', t1=48
            frames = self.pre_processing_pipeline(dict(frames=frames))["frames"]     # frame（96, 1, 3, 16, 160, 160）
        # # if self.pre_processing_pipeline_text is not None:                          # for VLM
        #     text_frames = self.pre_processing_pipeline_text(dict(text_frames=text_frames))["text_frames"]

        # flatten the batch dimension and num_segs dimension
        batches, num_segs = frames.shape[0:2]      # batch_size=96, num_seg=1
        frames = frames.flatten(0, 1).contiguous()  # [bs*num_seg, ...]  (96*1, 3, 16, 160, 160)

        # # # go through the git backbone
        # # text_batches, _, _, t = text_frames.shape[0:4]
        # # label_features = self.git_backbone(text_frames)    # text_features: (256, # “prefixLM=6”,1182=197*6,768)   caption_features, label_features
        # B_t = len(aligned_texts)
        # T_t = len(aligned_texts[0])
        # flat_texts = [s for seq in aligned_texts for s in seq]  # 为了将标签送tokenizer处理，把两个batch的label拼成2*768=1536的序列
        # if self.training:
        #     lss, label_features = self.git_backbone(text_frames, flat_texts)   #  last hidden state: t'6生了1184长度的句级嵌入
        #     caption = lss.detach().requires_grad_()
        #     label_features = label_features.detach()
        #     # fake inputs for debug, unable git_backbone
        #     # caption = torch.randn(256, 1184, 768, device="cuda")
        #     # label_features = torch.randn(1536, 768, device="cuda")
        # else:
        #     caption = self.git_backbone(text_frames, flat_texts)
        # # label_features = label_features.reshape(B_t, T_t, -1) # 纯净标签需要这行
        # # gtf = label_features.permute(0, 2, 1)  # [B, C, T]


        # go through the video backbone
        if self.freeze_backbone:  # freeze everything even in training
            with torch.no_grad():
                if self.use_temporal_checkpointing:
                    features = self.temporal_checkpointing(
                        frames,
                        self.temporal_checkpointing_chunk_num,
                        self.temporal_checkpointing_chunk_dim,
                    )
                else:
                    features = self.model.backbone(frames)

        else:  # let the model.train() or model.eval() decide whether to freeze
            if self.use_temporal_checkpointing:
                features = self.temporal_checkpointing(
                    frames,
                    self.temporal_checkpointing_chunk_num,
                    self.temporal_checkpointing_chunk_dim,
                )
            else:
                features = self.model.backbone(frames)       # features: s(96,384,8,10,10) b(96,768,8,10,10)  # , text_frames

        # unflatten and pool the features
        if isinstance(features, (tuple, list)):
            features = torch.cat([self.unflatten_and_pool_features(f, batches, num_segs) for f in features], dim=1)
        else:
            features = self.unflatten_and_pool_features(features, batches, num_segs)   # (B,C,T) t维度48*8=384然后差值到768

        # # unflatten the pool the text features,GIT方法下使用,对齐标签暂时注释
        B, T, C = features.shape
        # t2 = self.pre_processing_pipeline_text.transforms[0].kwargs['t2']   # 768和6怎么时间步对齐

        # gtf = label_features
        # global_text_features = caption   # (256,1177,768)

        # # infoNCE between caption and label_gt
        # caption = caption.mean(dim=1)
        # # 取6帧生成的句级平均
        # if self.training:
        #     label_gt = label_features.reshape(B*t2, -1, C) #.mean(dim=1)   # 768t也整合到batch维度，t维度变为6，6帧的label再取平均
        #     infonce_loss = self.loss_infonce(caption, label_gt)


        # GIT版本
        # if self.training:
        #     # # 缓存供第二次独立forward使用
        #     # self.cached_caption = caption.detach()
        #     # self.cached_t2 = t2
        #     # self.cached_label_features = label_features.detach()
        #     # self.cached_frame_label_ids = frame_label_ids
        #
        #     with torch.cuda.amp.autocast(enabled=False):
        #         caption, SSA_loss = self.SSA(caption, t2, label_features, frame_label_ids)
        #         caption_detached = caption.detach()
        # else:
        #     caption = self.SSA(caption, t2)

        # qwen版本
        caption = self.SSA(metas, T, valid_mask=masks)
        if self.training and self.use_infonce and self.infonce_loss_weight > 0:
            visual_tokens = features.permute(0, 2, 1).contiguous()
            frame_class_ids = frame_class_ids.to(device=visual_tokens.device)
            current_epoch = kwargs.get("curr_epoch", 0)
            warmup_scale = 1.0
            if self.infonce_warmup_epochs > 0:
                warmup_scale = min(1.0, float(current_epoch + 1) / self.infonce_warmup_epochs)
            SSA_loss = self.loss_infonce(
                visual_tokens.float(),
                caption.float(),
                class_ids=frame_class_ids,
                valid_mask=masks,
                lambda_bg=self.SSA.lambda_bg,
            ) * (self.infonce_loss_weight * warmup_scale)
        elif self.training:
            SSA_loss = features.sum() * 0.0
        caption = caption.permute(0, 2, 1)  # BCT

        # caption = caption.reshape(caption.shape[0]//t2, t2, -1).permute(0, 2, 1)   # BCT

        # 测试是否显示时序建模先关
        # B, C, T = gtf.shape
        # # perm = torch.randperm(T)
        # # gtf = gtf[:, :, perm]     # 或下面两行
        # gtf_mean = gtf.mean(dim=2, keepdim=True)  # [B, C, 1]
        # gtf = gtf_mean.expand(-1, -1, T)  # [B, C, T] 静态重复

        # apply mask
        if masks is not None and features.dim() == 3:
            features = features * masks.unsqueeze(1).detach().float()

        # mutimodal fusion
        # if hasattr(self.model, "fusing_module"):
        #     features = self.model.fusing_module(features, text_features)  如果有机会，设计一个对齐模块

        # def print_tensor_stats(tensor, name):
        #     print(f"{name}: shape={tensor.shape}")
        #     print(f"  min: {tensor.min().item():.6f}")
        #     print(f"  max: {tensor.max().item():.6f}")
        #     print(f"  mean: {tensor.mean().item():.6f}")
        #     print(f"  std: {tensor.std().item():.6f}")
        #     print()
        #
        # print_tensor_stats(features, "features")
        # print_tensor_stats(gtf, "gtf")

        # GIT 版本
        # n_class = kwargs.get("n_class")   # 用于类别矩阵
        # if self.training:
        #     features_fused = self.verb_guided_cross_attention(features, caption_detached, n_class)  # v(B,C,T), t_label(B,N,C)?, 纯净标签时为BCT,return (B,C,T)
        #                                                                            # 若换回GIT，必须注意gtf的维度变换！进去DEBUG！ GIT时输入是BCT
        # else:
        #     features_fused = self.verb_guided_cross_attention(features, caption, n_class)

        #qwen版本
        n_class = kwargs.get("n_class")  # 用于类别矩阵
        features_fused = self.verb_guided_cross_attention(features, caption, n_class)

        # 判断扰动是否有效
        # cos
        # cos = torch.nn.functional.cosine_similarity(features, features_fused, dim=-1)
        # print("Cosine similarity before/after fusion:", cos.mean().item())

        # make sure detector has the float32 input
        features_fused = features_fused.to(torch.float32)
        # return features_fused
        if self.training:
            return dict(features=features_fused, infonce_loss=SSA_loss)
            # return features_fused
        else:
            return features_fused

    def tensor_to_list(self, tensor):
        return [t for t in tensor]

    def unflatten_and_pool_features(self, features, batches, num_segs):     # batches=96, num_seg=1
        # unflatten the batch dimension and num_segs dimension
        features = features.unflatten(dim=0, sizes=(batches, num_segs))  # [bs, num_seg, ...]     ### (96,384,8,10,10)→(96,1,384,8,10,10)

        # convert the feature to [B,C,T]: pooling and other operations
        if self.post_processing_pipeline is not None:
            features = self.post_processing_pipeline(dict(feats=features))["feats"]            # (2,384,768)BCT    T=768是由384差值而来
        return features

    def unflatten_and_pool_text(self, gtf, batches, num_segs):
        # unflatten the batch dimension and num_segs dimension
        # gtf_feats = gtf.unflatten(dim=0, sizes=(batches, num_segs))
        gtf_feats = gtf.unsqueeze(1)

        # convert the feature to [B,C,T]: pooling and other operations
        if self.post_processing_pipeline_text is not None:
            gtf_feats = self.post_processing_pipeline_text(dict(feats=gtf_feats))["feats"]  # BCT
        return gtf_feats

    def set_norm_layer(self):
        if self.norm_eval:
            for m in self.modules():
                if isinstance(m, (nn.LayerNorm, nn.GroupNorm, _BatchNorm)):
                    m.eval()

                    for param in m.parameters():
                        param.requires_grad = False

    def temporal_checkpointing(self, frames, chunk_num, chunk_dim):
        """Temporal Checkpointing for Video Backbone.

        Temporal checkpointing will 1) split the video frames along the temporal dimension and sequentially forward each chunk with
        no gradients. 2) The backward pass will recompute the intermediate activations and compute each chunk's gradient. 3) Backbone's
        gradients will be accumulated along different chunks.

        Args:
            frames (Tensor): input frames, [B*N,3,T,H,W]
            chunk_num (int): number of chunks to split the temporal dimension
            chunk_dim (int): input shape is [B*N,3,T,H,W], so either dim=0 or 2 is fine
        """

        def _inner_forward(frames):
            return self.model.backbone(frames)

        video_feat = []
        for mini_frames in torch.chunk(frames, chunk_num, dim=chunk_dim):  # B*N is chunked
            # we can use torch.cp.checkpoint to implement an efficient temporal checkpointing mechanism
            mini_feat = cp.checkpoint(
                _inner_forward,
                mini_frames,
                use_reentrant=False,
            )
            video_feat.append(mini_feat)

        if isinstance(video_feat[0], (tuple, list)):
            video_feat = [torch.cat([f[idx] for f in video_feat], dim=chunk_dim) for idx in range(len(video_feat[0]))]
        else:
            video_feat = torch.cat(video_feat, dim=chunk_dim)
        return video_feat
