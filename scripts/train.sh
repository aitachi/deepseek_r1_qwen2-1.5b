#!/bin/bash

# 运行训练流程的脚本

# 设置环境变量
export KMP_DUPLICATE_LIB_OK=TRUE
export WANDB_DISABLED=true
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

# 确保目录存在
mkdir -p models
mkdir -p checkpoints

# 检查CUDA可用性
echo "检查CUDA可用性..."
python -c "import torch; print('CUDA可用:', torch.cuda.is_available())"

# 检查上一步是否成功的函数
check_status() {
    if [ $? -ne 0 ]; then
        echo "错误: 上一步骤失败。退出执行。"
        exit 1
    fi
}

# 1. 如果需要则下载模型
if [ ! -d "./models/qwen2-1.5b-instruct" ]; then
    echo "下载模型..."
    python src/models/qwen2model.py
    check_status
else
    echo "模型已下载，跳过下载步骤。"
fi

# 2. 运行冷启动训练（如果需要）
if [ ! -d "./models/1_cot_start_completed" ]; then
    echo "开始冷启动训练..."
    python src/training/cold_start.py
    check_status
else
    echo "冷启动已完成，跳过冷启动步骤。"
fi

# 3. 运行拒绝采样SFT
echo "开始拒绝采样SFT训练..."
python src/training/2_rejection_sampling_sft.py
check_status

# 4. 如果前一步成功，运行推理RL
echo "开始推理RL训练..."
python src/training/3_reasoning_rl.py
check_status

# 5. 运行所有场景RL
echo "开始所有场景RL训练..."
python src/training/4_all_scenarios_rl.py
check_status

echo "训练流程成功完成！"
echo "所有检查点已保存到: $(pwd)/checkpoints/"
