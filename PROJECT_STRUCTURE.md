# 项目完整结构

## 核心目录说明

```
deepseek_r1_qwen2-1.5b/
│
├── README.md                          # 英文项目文档
├── README_CN.md                       # 中文项目文档
├── requirements.txt                   # Python 依赖包列表
├── LICENSE                            # MIT 开源许可证
├── DeepSeek_R1.pdf                   # DeepSeek-R1 论文原文
│
├── algorithms/                        # 核心算法实现
│   ├── grpo_trainer.py               # GRPO 算法完整实现
│   ├── ppo_trainer.py                # PPO 算法完整实现
│   └── dpo_trainer.py                # DPO 算法完整实现
│
├── data/                              # 数据集目录
│   └── sample_reasoning_data.json    # 10条示例推理数据
│
├── run_comparison.py                  # 算法对比主脚本
│
├── GRPO_PPO_DPO/                     # 旧版对比实验结果
│   ├── compare_coding.py             # 早期对比代码
│   ├── *_policy.pth                  # 训练好的模型权重
│   ├── *.csv                         # 实验数据表格
│   └── *.png                         # 可视化图表
│
├── src/                               # 原有训练代码(可选)
│   ├── models/                       # 模型定义
│   │   ├── __init__.py
│   │   └── qwen2model.py            # Qwen2 模型封装
│   │
│   ├── training/                     # DeepSeek-R1 四阶段训练
│   │   ├── __init__.py
│   │   ├── 1_cot_star.py            # 冷启动训练
│   │   ├── 2_rejection_sampling_sft.py  # 拒绝采样SFT
│   │   ├── 3_reasoning_rl.py         # 推理导向RL
│   │   └── 4_all_scenarios_rl.py     # 全场景RL
│   │
│   ├── utils/                        # 工具函数
│   │   ├── __init__.py
│   │   └── monitoring.py            # 资源监控
│   │
│   └── fig/                          # 可视化脚本与图表
│       ├── figture.py
│       ├── figture_r1.py
│       └── *.png                     # 生成的图表
│
├── scripts/                           # 便捷脚本
│   ├── download_model.sh             # 模型下载脚本
│   └── train.sh                      # 训练启动脚本
│
├── checkpoints/                       # 模型检查点(训练时生成)
│   ├── grpo_model/
│   │   ├── epoch_1/
│   │   ├── epoch_2/
│   │   └── final/
│   ├── ppo_model/
│   │   ├── epoch_1/
│   │   ├── epoch_2/
│   │   └── final/
│   └── dpo_model/
│       ├── epoch_1/
│       ├── epoch_2/
│       └── final/
│
└── results/                           # 实验结果(运行时生成)
    └── algorithm_comparison/
        ├── COMPARISON_REPORT.md      # 详细对比报告
        ├── comparison_table.csv      # 对比数据表
        ├── full_results.json         # 完整实验数据
        └── *.png                     # 对比可视化图表

```

## 文件说明

### 核心算法实现 (algorithms/)

| 文件 | 说明 | 行数 |
|------|------|------|
| `grpo_trainer.py` | GRPO 完整实现,包含详细数学公式注释 | ~400 |
| `ppo_trainer.py` | PPO 完整实现,包含价值网络 | ~450 |
| `dpo_trainer.py` | DPO 完整实现,偏好优化 | ~350 |

### 数据集 (data/)

| 文件 | 说明 |
|------|------|
| `sample_reasoning_data.json` | 10条数学推理问题,包含:<br>- 代数题(4题)<br>- 微积分题(1题)<br>- 几何题(2题)<br>- 算术题(2题)<br>- 数列题(1题) |

### 主要脚本

| 文件 | 功能 | 用法 |
|------|------|------|
| `run_comparison.py` | 运行三种算法的完整对比实验 | `python run_comparison.py` |
| `scripts/train.sh` | 一键启动训练流程 | `bash scripts/train.sh` |
| `scripts/download_model.sh` | 下载 Qwen2 基础模型 | `bash scripts/download_model.sh` |

### 文档文件

| 文件 | 说明 |
|------|------|
| `README.md` | 英文版完整文档 |
| `README_CN.md` | 中文版完整文档 |
| `DeepSeek_R1.pdf` | DeepSeek-R1 论文原文 |
| `GRPO_PPO算法对比分析.md` | 早期算法分析文档 |
| `QWEN2-R1脚本分析报告.md` | Qwen2-R1 流程分析 |
| `SFT训练中NaN值问题的分析报告.md` | 训练问题分析 |

## 运行时生成的目录

### checkpoints/
训练过程中保存的模型检查点,每个算法包含:
- `epoch_N/`: 每个 epoch 的检查点
- `final/`: 最终训练完成的模型
- `stats.json`: 训练统计数据

### results/
实验结果输出目录,包含:
- 对比报告 (Markdown 格式)
- 数据表格 (CSV 格式)
- 可视化图表 (PNG 格式)
- 完整实验数据 (JSON 格式)

## 代码统计

| 类别 | 文件数 | 总行数 |
|------|--------|--------|
| 核心算法 | 3 | ~1200 |
| 训练脚本 | 4 | ~2700 |
| 工具函数 | 2 | ~150 |
| 对比脚本 | 2 | ~900 |
| **总计** | **11** | **~4950** |

## 依赖关系

```
algorithms/
  ├── grpo_trainer.py
  │   ├── torch
  │   ├── transformers (AutoModelForCausalLM, AutoTokenizer)
  │   └── numpy
  │
  ├── ppo_trainer.py
  │   ├── torch (nn.Module for ValueNetwork)
  │   ├── transformers
  │   └── numpy
  │
  └── dpo_trainer.py
      ├── torch (F.logsigmoid)
      ├── transformers
      └── numpy

run_comparison.py
  ├── algorithms.grpo_trainer
  ├── algorithms.ppo_trainer
  ├── algorithms.dpo_trainer
  ├── matplotlib
  └── pandas
```

## 作者信息

**作者:** Aitachi
**邮箱:** 44158892@qq.com
**GitHub:** https://github.com/aitachi/fast-socialfi

---

**最后更新:** 2025-01-02
