#!/bin/bash
# NPU 本地调试脚本
# 使用 dummy dataset 快速验证训练链路是否打通

set -e

echo "========================================="
echo "32x DA-VAE NPU 本地调试"
echo "========================================="

# ---- NPU 环境变量（根据实际环境调整）----
# export ASCEND_RT_VISIBLE_DEVICES=0
# export TASK_QUEUE_ENABLE=1
# export COMBINED_ENABLE=1
# export ACLNN_CACHE_LIMIT=100000

# ---- 用 dummy dataset 跑 20 步，验证前向/反向/保存 ----
python3 train.py \
    --yml_path train_vae32x_from_edit_gt.yaml \
    --dummy_dataset \
    --train_batch_size 2 \
    --max_train_steps 20 \
    --visualization_steps 10 \
    --val_visualization_steps 20 \
    --checkpointing_steps 10 \
    --learning_rate 1.0e-4 \
    --discriminator_learning_rate 1.0e-4 \
    --mixed_precision bf16 \
    --output_dir outputs/debug_npu \
    --seed 2333

echo "========================================="
echo "调试完成，输出目录: outputs/debug_npu"
echo "========================================="
