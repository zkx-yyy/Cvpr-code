_base_ = ["e2e_thumos_videomae_b_768x1_160_qwen_llm_image_tokens_paper_fcl.py"]

model = dict(
    backbone=dict(
        custom=dict(
            fusion_init_scale=0.15,
        )
    )
)

work_dir = "exps/thumos/open-git/07_qwen_image_tokens_fusion015"
