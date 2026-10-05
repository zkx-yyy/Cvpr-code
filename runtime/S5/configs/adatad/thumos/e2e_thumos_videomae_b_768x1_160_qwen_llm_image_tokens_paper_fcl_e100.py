_base_ = ["e2e_thumos_videomae_b_768x1_160_qwen_llm_image_tokens_paper_fcl.py"]

# Continue the original 03 run through its complete 100-epoch cosine schedule.
workflow = dict(end_epoch=100)

work_dir = "exps/thumos/open-git/03_qwen_llm_image_tokens_paper_fcl"
