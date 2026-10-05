_base_ = ["e2e_thumos_videomae_b_768x1_160_01_qwen_original_fcl_baseline.py"]

model = dict(
    backbone=dict(
        custom=dict(
            use_infonce=True,
            infonce_loss_weight=0.1,
            infonce_warmup_epochs=10,
            infonce_temperature=0.07,
            infonce_lambda_bg=0.25,
            fusion_init_scale=0.1,
        )
    )
)

work_dir = "exps/thumos/open-git/02_qwen_fusion_fixed_fcl"
