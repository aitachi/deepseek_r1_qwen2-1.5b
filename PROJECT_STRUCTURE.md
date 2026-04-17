# 项目完整结构

## 核心目录说明

```
PPOvDPOvGRPOvDAPO/
│
├── README.md                          # 英文项目文档
├── README_CN.md                       # 中文项目文档
├── requirements.txt                   # Python 依赖包列表
├── LICENSE                            # MIT 开源许可证
├── DeepSeek_R1.pdf                   # DeepSeek-R1 论文原文
├── DAPO.pdf                          # DAPO 论文原文
│
├── algorithms/                        # 核心算法实现
│   ├── ppo_trainer.py                # PPO 算法完整实现
│   ├── dpo_trainer.py                # DPO 算法完整实现
│   ├── grpo_trainer.py               # GRPO 算法完整实现
│   └── dapo_trainer.py               # DAPO 算法完整实现
│
├── data/                              # 数据集目录
│   └── sample_reasoning_data.json    # 10条示例推理数据
│
├── docs/                              # 文档与可视化
│   ├── PPO_Algorithm.md              # PPO 算法详解
│   ├── DPO_Algorithm.md              # DPO 算法详解
│   ├── GRPO_Algorithm.md             # GRPO 算法详解
│   ├── DAPO_Algorithm.md             # DAPO 算法详解
│   ├── Algorithm_Comparison.md       # 四算法对比
│   ├── RL_LLM_Survey_IEEE_EN.pdf     # 英文 IEEE 论文
│   ├── RL_LLM_Survey_IEEE_CN.pdf     # 中文 IEEE 论文
│   ├── ieee_en/                      # 英文 LaTeX 源文件
│   ├── ieee_cn/                      # 中文 LaTeX 源文件
│   └── figures/                      # 所有可视化图表
│
├── run_comparison.py                  # 四算法对比主脚本
│
├── DAPO_GRPO_PPO_DPO/                # 四算法对比实验结果
│   ├── compare_coding.py             # 对比代码
│   ├── *_policy.pth                  # 训练好的模型权重
│   ├── *.csv                         # 实验数据表格
│   └── *.png                         # 可视化图表
│
├── GRPO_PPO_DPO/                     # 旧版三算法对比实验结果
│
├── src/                               # 原有训练代码(可选)
│
├── scripts/                           # 便捷脚本
│   ├── download_model.sh             # 模型下载脚本
│   └── train.sh                      # 训练启动脚本
│
├── checkpoints/                       # 模型检查点(训练时生成)
│   ├── ppo_model/
│   ├── dpo_model/
│   ├── grpo_model/
│   └── dapo_model/
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
| `ppo_trainer.py` | PPO 完整实现,包含价值网络与 GAE | ~517 |
| `dpo_trainer.py` | DPO 完整实现,偏好优化 | ~431 |
| `grpo_trainer.py` | GRPO 完整实现,组优势归一化 | ~533 |
| `dapo_trainer.py` | DAPO 完整实现,动态采样与 Token 级损失 | ~738 |

### 数据集 (data/)

| 文件 | 说明 |
|------|------|
| `sample_reasoning_data.json` | 10条数学推理问题,包含:<br>- 代数题(4题)<br>- 微积分题(1题)<br>- 几何题(2题)<br>- 算术题(2题)<br>- 数列题(1题) |

### 主要脚本

| 文件 | 功能 | 用法 |
|------|------|------|
| `run_comparison.py` | 运行四种算法的完整对比实验 | `python run_comparison.py` |
| `scripts/train.sh` | 一键启动训练流程 | `bash scripts/train.sh` |
| `scripts/download_model.sh` | 下载 Qwen2 基础模型 | `bash scripts/download_model.sh` |

### 文档文件

| 文件 | 说明 |
|------|------|
| `README.md` | 英文版完整文档(四算法) |
| `README_CN.md` | 中文版完整文档(四算法) |
| `DeepSeek_R1.pdf` | DeepSeek-R1 论文原文 |
| `DAPO.pdf` | DAPO 论文原文 |
| `RL_Four_Algorithms_Comparison.md` | 四算法全面对比分析 |
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
| 核心算法 | 4 | ~2200 |
| 训练脚本 | 4 | ~2700 |
| 文档与可视化 | 8+ | ~2000 |
| 对比脚本 | 3 | ~1300 |
| **总计** | **19+** | **~8200** |

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
**GitHub:** https://github.com/aitachi/PPOvDPOvGRPOvDAPO

---

**最后更新:** 2025-04-07
