# 全流程复现Deepseek-R1增强型大语言模型

本项目仅用2-3条数据复现每个R1流程，跑通整个项目后，可以再找相关的数据置换全流程
模型选择的是阿里的qwen2-1.5b小模型
旨在增强语言模型的推理能力，特别关注Qwen2模型。它结合了拒绝采样与监督微调(SFT)和强化学习(RL)方法，以提高模型在推理任务上的表现。
复现deepseek-R1全流程

## 功能特点

- Qwen2-1.5B-Instruct 模型的下载与设置
- 拒绝采样SFT以提高推理能力
- 强化学习优化模型在各种场景下的输出
- 训练过程资源监控
- 支持推理和非推理任务

## 系统要求

- Python 3.12+
- PyTorch 2.0+
- Transformers 4.30+
- 支持CUDA的GPU，至少8GB显存（推荐）

## 项目结构

├── README.md
├── requirements.txt
├── src/ # 源代码
│ ├── models/ # 模型定义和下载功能
│ ├── training/ # 训练脚本
│ └── utils/ # 工具函数
├── scripts/ # 便捷执行的Shell脚本
├── models/ # 下载的模型存储
└── checkpoints/ # 训练得到的模型检查点

## 安装

1. 克隆仓库:
   
   ```bash
   git clone https://github.com/yourusername/reasoning-enhancement-llm.git
   cd reasoning-enhancement-llm
   ```

2. 安装依赖:
   
   ```bash
   pip install -r requirements.txt
   ```

3. 下载Qwen2模型:
   
   ```bash
   bash scripts/download_model.sh
   ```
   
   使用方法
   运行完整流程

要运行完整的训练流程:
bash

bash scripts/train.sh

这将执行:
    如果模型未下载，则下载模型
    运行拒绝采样SFT
    运行推理RL
    运行所有场景RL

单独运行各步骤
拒绝采样SFT

这种方法为推理任务生成多个候选响应，并选择最佳的进行监督微调:
bash

python src/training/rejection_sampling_sft.py

推理RL

专门应用强化学习来增强推理能力:

```bash

  python src/training/reasoning_rl.py
```

所有场景RL
在各种场景下用RL训练模型，包括潜在有害提示:

```bash
python src/training/all_scenarios_rl.py
```

训练流程
训练遵循以下关键步骤:

```txt
 从 models/ 目录加载基础模型 (Qwen2-1.5B-Instruct)
 通过拒绝采样生成训练数据
 使用包含推理链的特殊格式数据进行微调
 使用自定义奖励函数的RL技术进一步优化
 将检查点保存到 checkpoints/ 目录
```
