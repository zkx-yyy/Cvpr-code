_base_ = ["e2e_thumos_videomae_b_768x1_160_adapter.py"]

# Reproduction config for the 72.77 mAP THUMOS run, adapted to this machine.
data_root = "/mnt/nvme1n1/zhouzhewen/OpenTAD-main/data/thumos-14"
annotation_path = f"{data_root}/annotations/thumos_14_anno.json"
class_map = f"{data_root}/annotations/category_idx.txt"
data_path = f"{data_root}/raw_data/video"
qwen_feature_dir = "/mnt/nvme1n1/zhouzhewen/actionteller/qwen/data/thumos_hidden/frames_pooled"
pretrain_path = "/mnt/nvme1n1/zhouzhewen/OpenTAD-main/configs/adatad/pretrained/vit-base-p16_videomae-k400-pre_16x4x1_kinetics-400_20221013-860a3cd3.pth"

dataset = dict(
    train=dict(ann_file=annotation_path, class_map=class_map, data_path=data_path),
    val=dict(ann_file=annotation_path, class_map=class_map, data_path=data_path),
    test=dict(ann_file=annotation_path, class_map=class_map, data_path=data_path),
)

evaluation = dict(ground_truth_filename=annotation_path)

model = dict(
    backbone=dict(
        custom=dict(
            pretrain=pretrain_path,
            qwen_feature_dir=qwen_feature_dir,
            qwen_raw_dim=3584,
            infonce_loss_weight=0.5,
            infonce_temperature=0.07,
            infonce_lambda_bg=0.2,
        ),
    ),
)

# Keep the logged schedule shape, but stop at 75 epochs as requested.
scheduler = dict(type="LinearWarmupCosineAnnealingLR", warmup_epoch=5, max_epoch=100)
workflow = dict(
    logging_interval=50,
    checkpoint_interval=2,
    checkpoint_start_epoch=30,
    val_loss_interval=-1,
    val_eval_interval=2,
    val_start_epoch=40,
    end_epoch=75,
    detect_anomaly=False,
)

work_dir = "exps/thumos/open-git/01_qwen_original_fcl_baseline"
