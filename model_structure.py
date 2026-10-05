# DistributedDataParallel
# └── module: ActionFormer
#     ├── backbone: BackboneWrapper
#     │   └── model: Recognizer3D
#     │       ├── data_preprocessor: ActionDataPreprocessor()
#     │       └── backbone: VisionTransformerAdapter
#     │           ├── patch_embed: PatchEmbed
#     │           │   └── projection: Conv3d(3→768, kernel=(2,16,16), stride=(2,16,16))
#     │           ├── pos_drop: Dropout(p=0.0)
#     │           ├── blocks: ModuleList (12 blocks)
#     │           │   ├── Block 0:
#     │           │   │   ├── norm1: LayerNorm(768)
#     │           │   │   ├── attn: Attention
#     │           │   │   │   ├── qkv: Linear(768→2304, bias=False)
#     │           │   │   │   ├── attn_drop: Dropout(p=0.0)
#     │           │   │   │   ├── proj: Linear(768→768, bias=True)
#     │           │   │   │   └── proj_drop: Dropout(p=0.0)
#     │           │   │   ├── drop_path: Identity()
#     │           │   │   ├── norm2: LayerNorm(768)
#     │           │   │   ├── mlp: FFN
#     │           │   │   │   ├── layers: Sequential
#     │           │   │   │   │   ├── [0]: Linear(768→3072) + GELU + Dropout(p=0.0)
#     │           │   │   │   │   ├── [1]: Linear(3072→768, bias=True) + Dropout(p=0.0)
#     │           │   │   │   ├── dropout_layer: Identity()
#     │           │   │   │   └── gamma2: Identity()
#     │           │   │   └── adapter: Adapter
#     │           │   │       ├── down_proj: Linear(768→192, bias=True)
#     │           │   │       ├── act: GELU
#     │           │   │       ├── conv: Conv1d(192→192, kernel=1, stride=1)
#     │           │   │       ├── dwconv: Conv1d(192→192, kernel=3, pad=1, groups=192)
#     │           │   │       └── up_proj: Linear(192→768, bias=True)
#     │           │   │
#     │           │   └── Blocks 1-11 (11x Block): 结构与 Block 0 相同
#     │           │       └── 区别: drop_path 用 DropPath() 而非 Identity()
#     │           │
#     │           ├── norm: Identity()
#     │           └── fc_norm: LayerNorm(768)
#     │
#     │       ├── git_backbone: GITFeatureExtractor
#     │       │   └── model: PeftModelForCausalLM → LoraModel
#     │       │       └── model: GitForCausalLM
#     │       │           └── git: GitModel
#     │       │               ├── embeddings: GitEmbeddings
#     │       │               │   ├── word_embeddings: Embedding(30522→768, pad_idx=0)
#     │       │               │   ├── position_embeddings: Embedding(1024→768)
#     │       │               │   ├── LayerNorm: LayerNorm(768)
#     │       │               │   └── dropout: Dropout(p=0.1)
#     │       │               │
#     │       │               ├── image_encoder: GitVisionModel
#     │       │               │   └── vision_model: GitVisionTransformer
#     │       │               │       ├── patch_embedding: Conv2d(3→768, kernel=16x16, stride=16)
#     │       │               │       ├── position_embedding: Embedding(197→768)
#     │       │               │       ├── pre_layrnorm: LayerNorm(768)
#     │       │               │       ├── encoder (12x GitVisionEncoderLayer):
#     │       │               │       │   ├── self_attn: q/k/v/out_proj Linear(768→768)
#     │       │               │       │   ├── layer_norm1: LayerNorm(768)
#     │       │               │       │   ├── mlp: fc1(768→3072) + QuickGELU + fc2(3072→768)
#     │       │               │       │   └── layer_norm2: LayerNorm(768)
#     │       │               │       └── post_layernorm: LayerNorm(768)
#     │       │               │
#     │       │               ├── encoder (6x GitLayer):
#     │       │               │   ├── self_attn: query(768→768, LoRA) / key(Linear) / value(768→768, LoRA)
#     │       │               │   │   └── LoRA: A(768→8) + B(8→768), dropout=0.1
#     │       │               │   ├── output: dense(768→768) + LayerNorm + Dropout(0.1)
#     │       │               │   ├── intermediate: dense(768→3072) + GELU
#     │       │               │   └── output: dense(3072→768) + LayerNorm + Dropout(0.1)
#     │       │               │
#     │       │               ├── visual_projection: Linear(768→768) + LayerNorm(768)
#     │       │               │
#     │       │               └── img_temperal_embedding: ParameterList(6 params, each 1x1x768)
#     │       │
#     │       └── output: Linear(768→30522, bias=True)
#     │
#     │   └── verb_guided_cross_attention: VerbGuidedCrossAttention
#     │       ├── q_proj/k_proj/v_proj/out_proj: Linear(768→768, bias=True)
#     │       ├── dropout: Dropout(p=0.1)
#     │       └── ln: LayerNorm(768)
#     │
#     ├── projection: Conv1DTransformerProj
#     │   ├── embed (2 x ConvModule):
#     │   │   └── Conv1d(k=3,s=1,p=1) + LayerNorm + ReLU
#     │   │       ├── [0]: 768→512
#     │   │       └── [1]: 512→512
#     │   │
#     │   ├── stem (2x TransformerBlock):
#     │   │   ├── ln1/ln2: LayerNorm(512)
#     │   │   ├── attn: MaskedMHCA
#     │   │   │   ├── query/key/value_conv: Conv1d(512→512, k=3,s=1,p=1,groups=512)
#     │   │   │   ├── query/key/value_norm: LayerNorm(512)
#     │   │   │   ├── query/key/value: Conv1d(512→512, k=1,s=1)
#     │   │   │   ├── attn/proj_drop: Dropout(p=0.0)
#     │   │   │   └── proj: Conv1d(512→512, k=1,s=1)
#     │   │   ├── pool_skip: Identity()
#     │   │   ├── mlp: Conv1d(512→2048) + GELU + Drop → Conv1d(2048→512) + Drop
#     │   │   ├── drop_path_attn: AffineDropPath(0.1)
#     │   │   └── drop_path_mlp: AffineDropPath(0.1)
#     │   │
#     │   └── branch (5x TransformerBlock): 与 stem 结构类似
#     │       └── 区别: query/key/value_conv stride=2, pool_skip=MaxPool1d(k=3,s=2,p=1)
#     │
#     ├── neck: FPNIdentity
#     │   └── fpn_norms: 6x LayerNorm(512)
#     │
#     └── rpn_head: ActionFormerHead
#         ├── cls_convs (2x ConvModule): Conv1d(512→512,k=3,p=1) + LayerNorm + ReLU
#         ├── reg_convs (2x ConvModule): Conv1d(512→512,k=3,p=1) + LayerNorm + ReLU
#         ├── cls_head: Conv1d(512→20, k=3, p=1)
#         ├── reg_head: Conv1d(512→2, k=3, p=1)
#         ├── scale: 6x Scale()
#         ├── cls_loss: FocalLoss(alpha=0.25, gamma=2.0)
#         └── reg_loss: DIOULoss()
