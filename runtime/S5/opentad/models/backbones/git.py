import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional
from transformers import AutoConfig,AutoProcessor, AutoModelForCausalLM
from transformers.models.git.modeling_git import GitSelfAttention, GitEmbeddings
from peft import get_peft_model, LoraConfig, TaskType
"""
git版本,已弃用,现使用qwen
"""
class PatchedGitSelfAttention(GitSelfAttention):
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        past_key_value=None,
        output_attentions=False,
        pixel_values_present=False,
    ):
        mixed_query_layer = self.query(hidden_states)

        cutoff = self.image_patch_tokens if pixel_values_present else 0
        if past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([key_layer[:, :, :cutoff, :], past_key_value[0], key_layer[:, :, -1:, :]], dim=2)
            value_layer = torch.cat(
                [value_layer[:, :, :cutoff, :], past_key_value[1], value_layer[:, :, -1:, :]], dim=2
            )
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        use_cache = past_key_value is not None
        past_key_value = (
            key_layer[:, :, cutoff:, :],
            value_layer[:, :, cutoff:, :],
        )

        q = query_layer  # (B, H, L, D)
        k = key_layer    # (B, H, L_k, D)
        v = value_layer  # (B, H, L_v, D)
        attn_mask = None
        head_dim = q.shape[-1]

        try:
            with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
                context_layer = F.scaled_dot_product_attention(
                    q, k, v, attn_mask=attn_mask,
                    dropout_p=self.dropout.p if self.training else 0.0
                )
        except Exception as e:
            if self.training:
                print(f"[Warning] FlashAttention fallback: {e}")
            # fallback to manual attention
            attn_scores = torch.matmul(q, k.transpose(-1, -2)) / (head_dim ** 0.5)
            if attn_mask is not None:
                attn_scores = attn_scores.masked_fill(attn_mask, float("-inf"))
            attn_weights = attn_scores.softmax(dim=-1)
            attn_weights = F.dropout(attn_weights, p=self.dropout.p if self.training else 0.0, training=self.training)
            context_layer = torch.matmul(attn_weights, v)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()  # (B, L, H, D)
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)  # (B, L, hidden_size)

        outputs = (context_layer,)
        if output_attentions:
            outputs = outputs + (None,)
        outputs = outputs + (past_key_value,)

        return outputs

class GITFeatureExtractor(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # self._patch_gitmodel_forward()
        self.lora = cfg.lora

        self.processor = AutoProcessor.from_pretrained(cfg.model_path, local_files_only=True)     # attn_implementation=flash_attention_2"
        self.tokenizer = self.processor.tokenizer

        # config = AutoConfig.from_pretrained(model_path, local_files_only=True)
        # config.num_image_with_embedding = 768

        self.model = AutoModelForCausalLM.from_pretrained(cfg.model_path, local_files_only=True).cuda()  # config=config,
        self.model.config.use_cache = False
        self.model.gradient_checkpointing_enable()
        self._patch_embedding()
        self._patch_attention()

        if self.lora:
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,   # FEATURE_EXTRACTION
                r=cfg.r,                            # 4 for lite and 8 for standrad
                lora_alpha=cfg.lora_alpha,          # 16/32
                lora_dropout=cfg.lora_dropout,      # 0.05/0.1
                bias="none",
                target_modules=["query", "value"],  # "dense"你可以根据实际模型结构调整
            )
            self.model = get_peft_model(self.model, peft_config)
            print("[LoRA] Enabled. Trainable params:")
            self.model.print_trainable_parameters()
        else:
            for param in self.model.parameters():
                param.requires_grad = False

        # self._init_img_temperal_embedding()
        # self._freeze_except_img_temperal_embedding()
        # self.model.eval()
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Trainable params: {trainable} / {total}")
        print("GIT_Checkpointing enabled:", self.model.is_gradient_checkpointing)

        self.text_embeddings = self.model.get_input_embeddings()
        # self.class_name = self._load_class("/nvme3/stu04/OpenTAD-main/data/thumos-14/annotations/category_idx.txt")

    def forward(self, frames, flat_texts):    # 原始输入frames # frames: [256=2*128, 1, 3, 6(git), 160, 160]
        B, _, C, T, H, W = frames.shape   # 测试逐帧标签时暂时不需要

        # interpolate for git
        frames_reshaped = frames.permute(0, 1, 3, 2, 4, 5).reshape(B * 1 * T, C, H, W)
        frames_resized = F.interpolate(frames_reshaped, size=(224, 224), mode='bilinear', align_corners=False)
        frames_resized = frames_resized.reshape(B, 1, T, C, 224, 224).permute(0, 1, 3, 2, 4, 5)   # (B*t1, 1, C, T, H, W)

        # 测试逐帧标签时暂时不需要
        frames = frames.squeeze(1)
        frames = frames.permute(0, 2, 1, 3, 4)
        frames_reshaped = frames.reshape(B * T, C, H, W)
        frames_resized = F.interpolate(frames_reshaped, size=(224, 224), mode='bilinear', align_corners=False)
        frames_resized = frames_resized.reshape(B, T, C, 224, 224)
        pixel_values = frames_resized

        # lightweight caption, 测试逐帧标签时暂时不需要
        inputs = self.processor(
            text=[""] * B,
            return_tensors="pt",
            padding=True
        ).to("cuda")

        if self.training and self.lora:
            outputs = self.model(
                **inputs,
                pixel_values=pixel_values,
                output_hidden_states=True,
                return_dict=True
            )
        else:
            with torch.no_grad():
                outputs = self.model(
                    **inputs,
                    pixel_values=pixel_values,
                    output_hidden_states=True,
                    return_dict=True
                )

        # for debug
        # fake_hidden = torch.randn(B, 1184, 768, device="cuda")
        # class FakeOutput:
        #     def __init__(self, h):
        #         self.hidden_states = [h]
        # outputs = FakeOutput(fake_hidden)

        sequence_output = outputs.hidden_states[-1]  # [B, L, D] (256=2*128, 1184=6*(1+196)+2, 768)其中128是为了拆为6帧小片768÷128=6,1184是6乘（cls+196个img token）+2词表ids token
        num_img_tokens = getattr(self.model.git.config, "num_image_with_embedding", 0)
        # num_img_tokens = (self.model.git.config.image_size //self.model.git.config.patch_size)**2
        # caption_tokens = sequence_output[:, num_img_tokens + 1:, :]  # remove <bos><img> tokens [T, N-n, D]
        # caption_tokens = sequence_output[:, 2:, :]  # remove <bos> <img> tokens(非语义token) [T, N-2, D]  语义不正确！

        if self.training:
            token_emb, label_emb = self.labels_to_embeddings(flat_texts)    # embedding lookup, label_emb(1536=2*768, 768)
        # if len(label_emb.size()) == 2:
        #     label_emb = label_emb.expand(B//128, -1, -1)   # for thumos

        if self.training:
            return sequence_output, label_emb   # caption_tokens
        else:
            return sequence_output

    def _load_class(self,file_path):
        with open(file_path, 'r') as f:
            cls_name = [line.strip() for line in f.readlines()]
        return cls_name

    def labels_to_embeddings(self, class_names):
        device = next(self.text_embeddings.parameters()).device
        B = len(class_names)

        # 标记 [NULL] 文本
        is_null = [t.strip() == '[NULL]' for t in class_names]

        enc = self.tokenizer(
            class_names,
            add_special_tokens=False,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = enc.input_ids.to(device)
        attn_mask = enc.attention_mask.to(device)

        with torch.no_grad():
            token_emb = self.text_embeddings(input_ids)             # [B, L, D_t] D_t即text支路所要求的维度
            mask = attn_mask.unsqueeze(-1).to(token_emb.dtype)      # [B, L, 1]
            denom = mask.sum(dim=1).clamp_min(1.0)
            label_emb = (token_emb * mask).sum(dim=1) / denom       # [B, D_t] （1536=2*768, 768）(BT, C)

            for i, null_flag in enumerate(is_null):
                if null_flag:
                    label_emb[i] = 0.0

            label_emb = F.normalize(label_emb, dim=-1)

        return token_emb, label_emb  # label（1536=2*768, 768）(BT, C)
    def _patch_attention(self):
        """Patch attention layer to use Flash attention."""
        for i, layer in enumerate(self.model.git.encoder.layer):
            layer.attention.self = PatchedGitSelfAttention(self.model.config)

    def _patch_embedding(self):
        from transformers.models.git.modeling_git import GitEmbeddings

        def safe_forward(
                self,
                input_ids: Optional[torch.LongTensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None,
                past_key_values_length: int = 0,
        ) -> torch.Tensor:
            if input_ids is not None:
                input_shape = input_ids.size()
            else:
                input_shape = inputs_embeds.size()[:-1]

            seq_length = input_shape[1]

            if position_ids is None:
                # 兼容长序列（即便 position_ids 超过 max_position_embeddings）
                position_ids = torch.arange(
                    past_key_values_length, seq_length + past_key_values_length,
                    dtype=torch.long,
                    device=self.position_ids.device
                ).unsqueeze(0)

            if inputs_embeds is None:
                embeddings = self.word_embeddings(input_ids)
            else:
                embeddings = inputs_embeds

            if self.position_embedding_type == "absolute":
                position_embeddings = self.position_embeddings(position_ids)
                embeddings = embeddings.add(position_embeddings)  # 非 inplace，加法安全

            embeddings = self.LayerNorm(embeddings)
            embeddings = self.dropout(embeddings)
            return embeddings

        GitEmbeddings.forward = safe_forward
        print("[Patch] GitEmbeddings.forward safely patched to avoid inplace addition.")