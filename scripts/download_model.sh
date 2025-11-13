#!/bin/bash

# 下载Qwen2模型的脚本

# 设置环境变量
export KMP_DUPLICATE_LIB_OK=TRUE
export WANDB_DISABLED=true

# 确保模型目录存在
mkdir -p models

echo "开始下载模型..."
python src/models/qwen2model.py

echo "模型下载完成！"
echo "模型保存到: $(pwd)/models/qwen2-1.5b-instruct"