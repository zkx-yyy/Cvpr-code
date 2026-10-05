_base_ = ["e2e_thumos_videomae_s_768x1_160_adapter.py"]

model = dict(
    backbone=dict(
        backbone=dict(embed_dims=768, depth=12, num_heads=12),
        custom=dict(pretrain="/nvme3/stu04/OpenTAD-main/configs/adatad/pretrained/vit-base-p16_videomae-k400-pre_16x4x1_kinetics-400_20221013-860a3cd3.pth"),
    ), # /nvme3/stu04/OpenTAD-main/configs/adatad/pretrained/vit-base-p16_videomae-k400-pre_16x4x1_kinetics-400_20221013-860a3cd3.pth; /nvme3/stu04/OpenTAD-main/configs/adatad/pretrained/adatad_thumos_actionformer_videomae_b_768x1_160_adapter_epoch_51_c3872325.pth
    projection=dict(in_channels=768),
)

optimizer = dict(backbone=dict(custom=[dict(name="adapter", lr=1e-4, weight_decay=0.05,), # e-4
                                       dict(name="SSA", lr=1e-4, weight_decay=0.05),
                                       dict(name="verb_guided_cross_attention", lr=1e-4, weight_decay=0.05),
                                       ]),
                 # git_backbone=dict(custom=[dict(name="lora", lr=1e-4, weight_decay=0.05)])
                 )

# work_dir = "exps/thumos/open-git/VTT_CatMLPResi0.2_b_768x1_160"
# work_dir = "exps/thumos/open-git/SA_SSA_映射_text也映射_2.0版MaskNce_3视觉塔_B+FPN" # 对齐标签_null=0_emb_cross_Linear_gtf_B
work_dir = "exps/thumos/open-git/ddebug"
# work_dir = "exps/thumos/open-git/SE_QWEN_帧频级"