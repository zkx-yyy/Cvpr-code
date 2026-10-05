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

optimizer = dict(
    backbone=dict(
        custom=[
            dict(name="adapter", lr=1e-4, weight_decay=0.025),
            dict(name="SSA", lr=1e-4, weight_decay=0.025),
            dict(name="verb_guided_cross_attention", lr=1e-4, weight_decay=0.025),
        ]
    )
)

work_dir = "exps/thumos/open-git/04_qwen_llm_image_tokens_wd025"

