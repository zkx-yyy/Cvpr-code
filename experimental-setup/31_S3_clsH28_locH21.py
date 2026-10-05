# 实验：31_s3_hard_cls28_loc21_seed42
# 完整生效配置来源：exps/thumos/open-git/31_s3_hard_cls28_loc21_seed42/gpu2_id0/log.json:3
# 原始配置记录时间：2026-09-13 11:20:03
# 训练入口：runtime/S3/tools/train.py
# 实验命名记录 seed=42；启动时仍需显式传入 --seed 42。
# 原运行使用 2 张 GPU、--id 0；本文件不自动设置这些命令行参数。
# 所有训练/模型/数据参数保持日志中的值；唯一调整是去除 work_dir 的 gpuN_idM 后缀，
# 因为训练入口会自动追加该后缀。开始新实验时请显式指定新的 work_dir。

annotation_path = '/mnt/nvme1n1/zhouzhewen/OpenTAD-main/data/thumos-14/annotations/thumos_14_anno.json'
block_list = None
chunk_num = 48
class_map = '/mnt/nvme1n1/zhouzhewen/OpenTAD-main/data/thumos-14/annotations/category_idx.txt'
data_path = '/mnt/nvme1n1/zhouzhewen/OpenTAD-main/data/thumos-14/raw_data/video'
data_root = '/mnt/nvme1n1/zhouzhewen/OpenTAD-main/data/thumos-14'
dataset = dict(
    test=dict(
        ann_file=
        '/mnt/nvme1n1/zhouzhewen/OpenTAD-main/data/thumos-14/annotations/thumos_14_anno.json',
        block_list=None,
        class_map=
        '/mnt/nvme1n1/zhouzhewen/OpenTAD-main/data/thumos-14/annotations/category_idx.txt',
        data_path=
        '/mnt/nvme1n1/zhouzhewen/OpenTAD-main/data/thumos-14/raw_data/video',
        feature_stride=4,
        filter_gt=False,
        pipeline=[
            dict(format='mp4', type='PrepareVideoInfo'),
            dict(num_threads=4, type='mmaction.DecordInit'),
            dict(
                method='sliding_window',
                num_clips=1,
                scale_factor=1,
                type='LoadFrames'),
            dict(type='mmaction.DecordDecode'),
            dict(scale=(
                -1,
                160,
            ), type='mmaction.Resize'),
            dict(crop_size=160, type='mmaction.CenterCrop'),
            dict(input_format='NCTHW', type='mmaction.FormatShape'),
            dict(
                keys=[
                    'imgs',
                    'gt_segments',
                    'gt_labels',
                ],
                type='ConvertToTensor'),
            dict(
                inputs='imgs',
                keys=[
                    'masks',
                    'gt_segments',
                    'gt_labels',
                    'gt_labels_text',
                ],
                type='Collect'),
        ],
        sample_stride=1,
        subset_name='validation',
        test_mode=True,
        type='ThumosSlidingDataset',
        window_overlap_ratio=0.5,
        window_size=768),
    train=dict(
        ann_file=
        '/mnt/nvme1n1/zhouzhewen/OpenTAD-main/data/thumos-14/annotations/thumos_14_anno.json',
        block_list=None,
        class_map=
        '/mnt/nvme1n1/zhouzhewen/OpenTAD-main/data/thumos-14/annotations/category_idx.txt',
        data_path=
        '/mnt/nvme1n1/zhouzhewen/OpenTAD-main/data/thumos-14/raw_data/video',
        feature_stride=4,
        filter_gt=False,
        pipeline=[
            dict(format='mp4', type='PrepareVideoInfo'),
            dict(num_threads=4, type='mmaction.DecordInit'),
            dict(
                crop_ratio=[
                    0.9,
                    1.0,
                ],
                method='random_trunc',
                num_clips=1,
                scale_factor=1,
                trunc_len=768,
                trunc_thresh=0.75,
                type='LoadFrames'),
            dict(type='mmaction.DecordDecode'),
            dict(scale=(
                -1,
                182,
            ), type='mmaction.Resize'),
            dict(type='mmaction.RandomResizedCrop'),
            dict(keep_ratio=False, scale=(
                160,
                160,
            ), type='mmaction.Resize'),
            dict(flip_ratio=0.5, type='mmaction.Flip'),
            dict(transforms='default', type='mmaction.ImgAug'),
            dict(type='mmaction.ColorJitter'),
            dict(input_format='NCTHW', type='mmaction.FormatShape'),
            dict(
                keys=[
                    'imgs',
                    'gt_segments',
                    'gt_labels',
                ],
                type='ConvertToTensor'),
            dict(
                inputs='imgs',
                keys=[
                    'masks',
                    'gt_segments',
                    'gt_labels',
                    'gt_labels_text',
                ],
                type='Collect'),
        ],
        sample_stride=1,
        subset_name='training',
        type='ThumosPaddingDataset'),
    val=dict(
        ann_file=
        '/mnt/nvme1n1/zhouzhewen/OpenTAD-main/data/thumos-14/annotations/thumos_14_anno.json',
        block_list=None,
        class_map=
        '/mnt/nvme1n1/zhouzhewen/OpenTAD-main/data/thumos-14/annotations/category_idx.txt',
        data_path=
        '/mnt/nvme1n1/zhouzhewen/OpenTAD-main/data/thumos-14/raw_data/video',
        feature_stride=4,
        filter_gt=False,
        pipeline=[
            dict(format='mp4', type='PrepareVideoInfo'),
            dict(num_threads=4, type='mmaction.DecordInit'),
            dict(
                method='sliding_window',
                num_clips=1,
                scale_factor=1,
                type='LoadFrames'),
            dict(type='mmaction.DecordDecode'),
            dict(scale=(
                -1,
                160,
            ), type='mmaction.Resize'),
            dict(crop_size=160, type='mmaction.CenterCrop'),
            dict(input_format='NCTHW', type='mmaction.FormatShape'),
            dict(
                keys=[
                    'imgs',
                    'gt_segments',
                    'gt_labels',
                ],
                type='ConvertToTensor'),
            dict(
                inputs='imgs',
                keys=[
                    'masks',
                    'gt_segments',
                    'gt_labels',
                ],
                type='Collect'),
        ],
        sample_stride=1,
        subset_name='validation',
        type='ThumosSlidingDataset',
        window_overlap_ratio=0.25,
        window_size=768))
evaluation = dict(
    ground_truth_filename=
    '/mnt/nvme1n1/zhouzhewen/OpenTAD-main/data/thumos-14/annotations/thumos_14_anno.json',
    subset='validation',
    tiou_thresholds=[
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
    ],
    type='mAP')
inference = dict(load_from_raw_predictions=False, save_raw_prediction=False)
model = dict(
    backbone=dict(
        backbone=dict(
            adapter_index=[
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
            ],
            depth=12,
            drop_path_rate=0.1,
            embed_dims=768,
            img_size=224,
            mlp_ratio=4,
            norm_cfg=dict(eps=1e-06, type='LN'),
            num_frames=16,
            num_heads=12,
            patch_size=16,
            qkv_bias=True,
            return_feat_map=True,
            total_frames=768,
            type='VisionTransformerAdapter',
            with_cp=True),
        custom=dict(
            freeze_backbone=False,
            fusion_init_scale=0.075,
            infonce_lambda_bg=0.25,
            infonce_loss_weight=0.1,
            infonce_temperature=0.07,
            infonce_warmup_epochs=10,
            norm_eval=False,
            post_processing_pipeline=[
                dict(
                    keys=[
                        'feats',
                    ],
                    ops='b n c t h w -> b c t',
                    reduction='mean',
                    type='Reduce'),
                dict(
                    keys=[
                        'feats',
                    ],
                    ops='(b t1) c t -> b c (t1 t)',
                    t1=48,
                    type='Rearrange'),
                dict(keys=[
                    'feats',
                ], size=768, type='Interpolate'),
            ],
            pre_processing_pipeline=[
                dict(
                    keys=[
                        'frames',
                    ],
                    ops='b n c (t1 t) h w -> (b t1) n c t h w',
                    t1=48,
                    type='Rearrange'),
            ],
            pretrain=
            '/mnt/nvme1n1/zhouzhewen/OpenTAD-main/configs/adatad/pretrained/vit-base-p16_videomae-k400-pre_16x4x1_kinetics-400_20221013-860a3cd3.pth',
            qwen_feature_dir=[
                '/mnt/nvme1n1/zhouzhewen/actionteller/qwen/data/thumos_hidden/frames_llm_image_tokens_single_layers_7_14_21_28/layer_7',
                '/mnt/nvme1n1/zhouzhewen/actionteller/qwen/data/thumos_hidden/frames_llm_image_tokens_single_layers_7_14_21_28/layer_14',
                '/mnt/nvme1n1/zhouzhewen/actionteller/qwen/data/thumos_hidden/frames_llm_image_tokens_single_layers_7_14_21_28/layer_21',
                '/mnt/nvme1n1/zhouzhewen/actionteller/qwen/data/thumos_hidden/frames_llm_image_tokens_single_layers_7_14_21_28/layer_28',
            ],
            qwen_raw_dim=3584,
            use_infonce=True),
        data_preprocessor=dict(
            format_shape='NCTHW',
            mean=[
                123.675,
                116.28,
                103.53,
            ],
            std=[
                58.395,
                57.12,
                57.375,
            ],
            type='mmaction.ActionDataPreprocessor'),
        type='mmaction.Recognizer3D'),
    neck=dict(
        in_channels=512, num_levels=6, out_channels=512, type='FPNIdentity'),
    projection=dict(
        arch=(
            2,
            2,
            5,
        ),
        attn_cfg=dict(n_head=4, n_mha_win_size=-1),
        conv_cfg=dict(kernel_size=3, proj_pdrop=0.0),
        in_channels=768,
        max_seq_len=768,
        norm_cfg=dict(type='LN'),
        out_channels=512,
        path_pdrop=0.1,
        type='Conv1DTransformerProj',
        use_abs_pe=False),
    rpn_head=dict(
        center_sample='radius',
        center_sample_radius=1.5,
        cls_prior_prob=0.01,
        feat_channels=512,
        in_channels=512,
        label_smoothing=0.0,
        loss=dict(
            cls_loss=dict(type='FocalLoss'), reg_loss=dict(type='DIOULoss')),
        loss_normalizer=100,
        loss_normalizer_momentum=0.9,
        num_classes=20,
        num_convs=2,
        prior_generator=dict(
            regression_range=[
                (
                    0,
                    4,
                ),
                (
                    4,
                    8,
                ),
                (
                    8,
                    16,
                ),
                (
                    16,
                    32,
                ),
                (
                    32,
                    64,
                ),
                (
                    64,
                    10000,
                ),
            ],
            strides=[
                1,
                2,
                4,
                8,
                16,
                32,
            ],
            type='PointGenerator'),
        type='ActionFormerHead'),
    task_semantic_mode='hard',
    type='ActionFormer')
optimizer = dict(
    backbone=dict(
        custom=[
            dict(lr=0.0001, name='adapter', weight_decay=0.05),
            dict(lr=0.0001, name='SSA', weight_decay=0.05),
            dict(
                lr=0.0001,
                name='verb_guided_cross_attention',
                weight_decay=0.05),
        ],
        exclude=[
            'backbone',
        ],
        lr=0,
        weight_decay=0),
    lr=0.0001,
    paramwise=True,
    type='AdamW',
    weight_decay=0.05)
post_processing = dict(
    nms=dict(
        max_seg_num=2000,
        multiclass=True,
        sigma=0.7,
        use_soft_nms=True,
        voting_thresh=0.7),
    save_dict=False)
pretrain_path = '/mnt/nvme1n1/zhouzhewen/OpenTAD-main/configs/adatad/pretrained/vit-base-p16_videomae-k400-pre_16x4x1_kinetics-400_20221013-860a3cd3.pth'
qwen_feature_dir = '/mnt/nvme1n1/zhouzhewen/actionteller/qwen/data/thumos_hidden/frames_pooled'
scale_factor = 1
scheduler = dict(
    max_epoch=100, type='LinearWarmupCosineAnnealingLR', warmup_epoch=5)
solver = dict(
    amp=True,
    clip_grad_norm=1,
    ema=True,
    fp16_compress=True,
    static_graph=True,
    test=dict(batch_size=2, num_workers=2),
    train=dict(batch_size=2, num_workers=2),
    val=dict(batch_size=2, num_workers=2))
window_size = 768
work_dir = '/mnt/nvme1n1/zhouzhewen/actionteller/open-git/exps/thumos/open-git/31_s3_hard_cls28_loc21_seed42'
workflow = dict(
    checkpoint_interval=2,
    checkpoint_start_epoch=30,
    detect_anomaly=False,
    end_epoch=80,
    logging_interval=50,
    val_eval_interval=2,
    val_loss_interval=-1,
    val_start_epoch=40)
