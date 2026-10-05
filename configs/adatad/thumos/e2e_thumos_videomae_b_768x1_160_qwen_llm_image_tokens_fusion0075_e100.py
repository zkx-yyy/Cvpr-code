_base_ = ["../../../实验配置/03_qwen_image_tokens_e75.py"]

model = dict(
    backbone=dict(
        custom=dict(
            fusion_init_scale=0.075,
        )
    )
)

workflow = dict(end_epoch=100)

work_dir = "exps/thumos/open-git/08_qwen_image_tokens_fusion0075_e100"
