# 实验配置

本目录统一保存现有检测实验的完整生效配置，按实验编号命名。整理日期：2026-10-04。

## 保存范围

- 共 23 个独立 Python 配置：01–07、09、15–18、24–33；03 分为最初 75 轮与后续 100 轮两个配置。
- 08 按要求保持在原位置，本目录没有创建 08 的重复配置：[08 原配置](../configs/adatad/thumos/e2e_thumos_videomae_b_768x1_160_qwen_llm_image_tokens_fusion0075_e100.py)。08 仅把继承路径改为本目录的 `03_qwen_image_tokens_e75.py`；已逐项验证合并后的全部参数与清理前一致。
- 全部配置从各次训练日志中的完整 `Config:` 段提取，包含启动时的覆盖参数，不是简单复制实验目录里可能过时的基础 `.py`。
- 本目录配置已展开继承，可以独立解析，不依赖其他实验配置。数据、特征和预训练权重仍引用原来路径。
- 主目录 `configs/` 中 8 份已归档的重复 Qwen 配置已删除；08 保持原位并改用集中配置作为继承来源。`exps/` 中的配置本次未清理，模型实现、日志、权重和同级代码快照均未修改。
- Probe 结果 JSON、提取脚本和控制脚本不属于检测训练配置，没有混入本目录。

## 配置索引

| 文件 | 实验目录 | 配置训练轮数 | 模式 | seed 记录 |
|---|---|---:|---|---|
| [01_qwen_original_fcl.py](01_qwen_original_fcl.py) | `01_qwen_original_fcl_baseline` | 75 | `—` | 日志 Config 未记录 |
| [02_qwen_fusion_fixed_fcl.py](02_qwen_fusion_fixed_fcl.py) | `02_qwen_fusion_fixed_fcl` | 75 | `—` | 日志 Config 未记录 |
| [03_qwen_image_tokens_e75.py](03_qwen_image_tokens_e75.py) | `03_qwen_llm_image_tokens_paper_fcl` | 75 | `—` | 日志 Config 未记录 |
| [03_qwen_image_tokens_e100.py](03_qwen_image_tokens_e100.py) | `03_qwen_llm_image_tokens_paper_fcl` | 100 | `—` | 日志 Config 未记录 |
| [04_qwen_image_tokens_wd0p025.py](04_qwen_image_tokens_wd0p025.py) | `04_qwen_llm_image_tokens_wd025` | 75 | `—` | 日志 Config 未记录 |
| [05_qwen_image_tokens_seed2026.py](05_qwen_image_tokens_seed2026.py) | `05_qwen_llm_image_tokens_seed2026` | 75 | `—` | 2026 |
| [06_qwen_image_tokens_fcl0p05.py](06_qwen_image_tokens_fcl0p05.py) | `06_qwen_image_tokens_fcl005` | 75 | `—` | 日志 Config 未记录 |
| [07_qwen_image_tokens_fusion0p15.py](07_qwen_image_tokens_fusion0p15.py) | `07_qwen_image_tokens_fusion015` | 75 | `—` | 日志 Config 未记录 |
| [09_qwen_dual_branch_cls0p075_loc0p025.py](09_qwen_dual_branch_cls0p075_loc0p025.py) | `09_qwen_dual_branch_cls0075_loc0025` | 100 | `—` | 日志 Config 未记录 |
| [15_single_H7.py](15_single_H7.py) | `15_qwen_single_layer_h7_depth025_e100` | 100 | `—` | 日志 Config 未记录 |
| [16_single_H14.py](16_single_H14.py) | `16_qwen_single_layer_h14_depth050_e100` | 100 | `—` | 日志 Config 未记录 |
| [17_single_H21.py](17_single_H21.py) | `17_qwen_single_layer_h21_depth075_e100` | 100 | `—` | 日志 Config 未记录 |
| [18_single_H28.py](18_single_H28.py) | `18_qwen_single_layer_h28_depth100_e100` | 100 | `—` | 日志 Config 未记录 |
| [24_S0_H28_seed42_run_a_e10.py](24_S0_H28_seed42_run_a_e10.py) | `24_s0_h28_repro_seed42_run_a` | 10 | `—` | 42 |
| [25_S0_H28_seed42_run_b_e10.py](25_S0_H28_seed42_run_b_e10.py) | `25_s0_h28_repro_seed42_run_b` | 10 | `—` | 42 |
| [26_S0_H28_seed42.py](26_S0_H28_seed42.py) | `26_s0_h28_seed42` | 100 | `—` | 42 |
| [27_S0_H28_seed2026.py](27_S0_H28_seed2026.py) | `27_s0_h28_seed2026` | 100 | `—` | 2026 |
| [28_S0_H28_seed3407.py](28_S0_H28_seed3407.py) | `28_s0_h28_seed3407` | 100 | `—` | 3407 |
| [29_S1_mean_H7_H14_H21_H28.py](29_S1_mean_H7_H14_H21_H28.py) | `29_s1_mean_h7_h14_h21_h28_seed42` | 80 | `—` | 42 |
| [30_S2_shared_learned.py](30_S2_shared_learned.py) | `30_s2_shared_learned_h7_h14_h21_h28_seed42` | 80 | `—` | 42 |
| [31_S3_clsH28_locH21.py](31_S3_clsH28_locH21.py) | `31_s3_hard_cls28_loc21_seed42` | 80 | `hard` | 42 |
| [32_S4_task_learned.py](32_S4_task_learned.py) | `32_s4_task_learned_h7_h14_h21_h28_seed42` | 80 | `learned` | 42 |
| [33_S5_video_dynamic.py](33_S5_video_dynamic.py) | `33_s5_video_dynamic_h7_h14_h21_h28_seed42` | 80 | `video` | 42 |

## 使用时需要明确的设置

1. **模型版本**：此 Code 副本中的 S3、S4、S5 分别使用 `runtime/S3/tools/train.py`、`runtime/S4/tools/train.py`、`runtime/S5/tools/train.py`，完整路径写在各配置文件头部。S5 的 `video` 模式尚未合入主目录模型；本次只集中配置，没有合并模型实现。早期实验缺少同样的独立源码快照，整理配置不等于确认历史源码完全一致。
2. **seed 与运行编号**：现有训练脚本从命令行读取 `--seed`、`--id`，不会根据配置文件名自动设定。索引中的 seed 来源于实验命名，日志 `Config:` 本身不包含这些启动参数；未记录的项没有推测填充。
3. **输出目录**：日志中的 `work_dir` 已经追加过 `gpuN_idM/`，本目录只去除这一后缀，让训练脚本正常追加一次。其余参数逐项等于原日志配置。新运行应通过 `--cfg-options work_dir=...` 指向新的输出目录，避免覆盖历史实验。
4. **训练轮数不是完成状态**：例如 17、18 的配置上限为 100 轮，但历史运行中止于进入 epoch 80 附近；24、25 是 10 轮短程复现实验。此处没有把它们改写成新的训练方案。
5. **03 的续训**：100 轮文件保存续训阶段的参数。是否从原检查点续训，由启动时的 `--resume` 决定，配置文件不会自动恢复检查点。
6. **实际特征字段**：模型实际使用 `model.backbone.custom.qwen_feature_dir`。部分历史配置还带有未使用的顶层同名字段，本次为保存原配置保留该字段，不能据此判断实际输入特征。

以下仅为从工程根目录启动 S5 的命令示例，本次没有执行。示例采用新的输出目录；GPU 可见性需按机器当前情况设置：

```bash
torchrun --standalone --nproc_per_node=2 \
  runtime/S5/tools/train.py \
  实验配置/33_S5_video_dynamic.py \
  --seed 42 --id 0 \
  --cfg-options work_dir=exps/thumos/open-git/s5_from_organized_config
```

## 原工程中的配置记录位置（历史来源，不是本副本的启动入口）

- `01_qwen_original_fcl.py` ← `exps/thumos/open-git/01_qwen_original_fcl_baseline/gpu2_id0/log.json:3`
- `02_qwen_fusion_fixed_fcl.py` ← `exps/thumos/open-git/02_qwen_fusion_fixed_fcl/gpu2_id1/log.json:3`
- `03_qwen_image_tokens_e75.py` ← `exps/thumos/open-git/03_qwen_llm_image_tokens_paper_fcl/gpu2_id0/log.json:3`
- `03_qwen_image_tokens_e100.py` ← `exps/thumos/open-git/03_qwen_llm_image_tokens_paper_fcl/gpu2_id0/log.json:1157`
- `04_qwen_image_tokens_wd0p025.py` ← `exps/thumos/open-git/04_qwen_llm_image_tokens_wd025/gpu2_id0/log.json:3`
- `05_qwen_image_tokens_seed2026.py` ← `exps/thumos/open-git/05_qwen_llm_image_tokens_seed2026/gpu2_id0/log.json:3`
- `06_qwen_image_tokens_fcl0p05.py` ← `exps/thumos/open-git/06_qwen_image_tokens_fcl005/gpu2_id0/log.json:3`
- `07_qwen_image_tokens_fusion0p15.py` ← `exps/thumos/open-git/07_qwen_image_tokens_fusion015/gpu2_id0/log.json:3`
- `09_qwen_dual_branch_cls0p075_loc0p025.py` ← `exps/thumos/open-git/09_qwen_dual_branch_cls0075_loc0025/gpu2_id0/log.json:3`
- `15_single_H7.py` ← `exps/thumos/open-git/15_qwen_single_layer_h7_depth025_e100/gpu2_id0/log.json:3`
- `16_single_H14.py` ← `exps/thumos/open-git/16_qwen_single_layer_h14_depth050_e100/gpu2_id0/log.json:3`
- `17_single_H21.py` ← `exps/thumos/open-git/17_qwen_single_layer_h21_depth075_e100/gpu2_id0/log.json:3`
- `18_single_H28.py` ← `exps/thumos/open-git/18_qwen_single_layer_h28_depth100_e100/gpu2_id0/log.json:3`
- `24_S0_H28_seed42_run_a_e10.py` ← `exps/thumos/open-git/24_s0_h28_repro_seed42_run_a/gpu2_id0/log.json:3`
- `25_S0_H28_seed42_run_b_e10.py` ← `exps/thumos/open-git/25_s0_h28_repro_seed42_run_b/gpu2_id0/log.json:3`
- `26_S0_H28_seed42.py` ← `exps/thumos/open-git/26_s0_h28_seed42/gpu2_id0/log.json:3`
- `27_S0_H28_seed2026.py` ← `exps/thumos/open-git/27_s0_h28_seed2026/gpu2_id0/log.json:3`
- `28_S0_H28_seed3407.py` ← `exps/thumos/open-git/28_s0_h28_seed3407/gpu2_id0/log.json:3`
- `29_S1_mean_H7_H14_H21_H28.py` ← `exps/thumos/open-git/29_s1_mean_h7_h14_h21_h28_seed42/gpu2_id0/log.json:3`
- `30_S2_shared_learned.py` ← `exps/thumos/open-git/30_s2_shared_learned_h7_h14_h21_h28_seed42/gpu2_id0/log.json:3`
- `31_S3_clsH28_locH21.py` ← `exps/thumos/open-git/31_s3_hard_cls28_loc21_seed42/gpu2_id0/log.json:3`
- `32_S4_task_learned.py` ← `exps/thumos/open-git/32_s4_task_learned_h7_h14_h21_h28_seed42/gpu2_id0/log.json:3`
- `33_S5_video_dynamic.py` ← `exps/thumos/open-git/33_s5_video_dynamic_h7_h14_h21_h28_seed42/gpu2_id0/log.json:3`

## 已完成的校验

- 23 个文件均通过实际 `mmengine.Config.fromfile` 加载。
- 加载后逐项比较原日志配置，除说明过的 `work_dir` 后缀还原外完全一致。
- S3/S4/S5 日志配置与原独立配置比较，仅 `work_dir` 存在训练入口追加的后缀差异。
- 08 原位置不变，继承路径更新后仍可加载，合并后的全部参数与清理前完全一致。
- 未启动训练、评估、特征提取或 GPU 任务。

## 主目录重复配置清理

已删除原 `configs/adatad/thumos/` 中以下重复文件；集中版本见上面的配置索引。01 集中版本使用历史日志中的实际输出路径，其他训练参数与旧模板一致。

- `e2e_thumos_videomae_b_768x1_160_01_qwen_original_fcl_baseline.py`
- `e2e_thumos_videomae_b_768x1_160_qwen_fusion_fixed_fcl.py`
- `e2e_thumos_videomae_b_768x1_160_qwen_llm_image_tokens_fcl005.py`
- `e2e_thumos_videomae_b_768x1_160_qwen_llm_image_tokens_fusion015.py`
- `e2e_thumos_videomae_b_768x1_160_qwen_llm_image_tokens_paper_fcl_e100.py`
- `e2e_thumos_videomae_b_768x1_160_qwen_llm_image_tokens_paper_fcl.py`
- `e2e_thumos_videomae_b_768x1_160_qwen_llm_image_tokens_seed2026.py`
- `e2e_thumos_videomae_b_768x1_160_qwen_llm_image_tokens_wd025.py`
