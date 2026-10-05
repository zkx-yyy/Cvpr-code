_base_ = ["e2e_thumos_videomae_b_768x1_160_qwen_fusion_fixed_fcl.py"]

model = dict(
    backbone=dict(
        custom=dict(
            qwen_feature_dir=(
                "/mnt/nvme1n1/zhouzhewen/actionteller/qwen/data/thumos_hidden/frames_llm_image_tokens"
            ),
        )
    )
)

work_dir = "exps/thumos/open-git/03_qwen_llm_image_tokens_paper_fcl"
