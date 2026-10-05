_base_ = ["e2e_thumos_videomae_b_768x1_160_qwen_llm_image_tokens_paper_fcl.py"]

model = dict(
    backbone=dict(
        custom=dict(
            infonce_loss_weight=0.05,
        )
    )
)

work_dir = "exps/thumos/open-git/06_qwen_image_tokens_fcl005"
