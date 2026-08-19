# PPO vs DPO vs GRPO vs DAPO: 全面对比分析

> **Comprehensive Comparison of Reinforcement Learning Algorithms for LLM Training**
> 大语言模型强化学习算法全面对比分析

---

## 1. 算法概览 / Algorithm Overview

| 特征 / Feature | PPO | DPO | GRPO | DAPO |
|:---|:---|:---|:---|:---|
| **全称** | Proximal Policy Optimization | Direct Preference Optimization | Group Relative Policy Optimization | Dynamic Advantage Policy Optimization |
| **来源论文** | Schulman et al., 2017 | Rafailov et al., 2023 | DeepSeek-AI, 2025 | ByteDance, 2025 |
| **需要价值网络** | Yes | No | No | No |
| **需要参考模型** | No | Yes | Yes | Yes |
| **采样方式** | 单样本 | 偏好对 | 组采样(G个) | 动态组采样 |
| **优势估计** | GAE | 无 | 组归一化 | 动态组归一化 |
| **损失粒度** | 序列级 | 序列级 | 序列级 | Token级 |
| **KL约束** | 无(含熵正则) | 隐式(β参数) | 显式KL惩罚 | 显式KL惩罚 |

---

## 2. 数学公式对比 / Mathematical Formulation Comparison

### 2.1 核心目标函数

#### PPO (Proximal Policy Optimization)
$$L_{PPO}(\theta) = \mathbb{E}_t\left[\min\left(r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\varepsilon, 1+\varepsilon)\hat{A}_t\right)\right] - c_1 L^{VF}(\theta) + c_2 S[\pi_\theta]$$

- **优势来源**: GAE (Generalized Advantage Estimation)
- **约束方式**: 裁剪概率比
- **额外组件**: 价值损失 + 熵正则化

#### DPO (Direct Preference Optimization)
$$L_{DPO}(\theta) = -\mathbb{E}_{(x,y_w,y_l)}\left[\log\sigma\left(\beta\log\frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta\log\frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\right)\right]$$

- **优势来源**: 隐式奖励差 (Bradley-Terry模型)
- **约束方式**: 参考模型KL (隐式)
- **额外组件**: 无

#### GRPO (Group Relative Policy Optimization)
$$J_{GRPO}(\theta) = \mathbb{E}\left[\frac{1}{G}\sum_{i=1}^{G}\min\left(\rho_i\hat{A}_i, \text{clip}(\rho_i, 1-\varepsilon, 1+\varepsilon)\hat{A}_i\right) - \beta D_{KL}(\pi_\theta \| \pi_{ref})\right]$$

- **优势来源**: 组统计归一化 $\hat{A}_i = \frac{r_i - \mu}{\sigma}$
- **约束方式**: 裁剪 + KL惩罚
- **额外组件**: KL散度惩罚

#### DAPO (Dynamic Advantage Policy Optimization)
$$L_{DAPO}(\theta) = -\mathbb{E}\left[\frac{1}{\sum|o_i|}\sum_{i=1}^{G}\frac{1}{|o_i|}\sum_{t=1}^{|o_i|}\min\left(r_t\hat{A}_i, \text{clip}(r_t)\hat{A}_i\right) - \beta D_{KL}^{token}\right]$$

- **优势来源**: 动态组归一化 + 过长过滤
- **约束方式**: Token级裁剪 + KL
- **额外组件**: 动态采样 + 过长过滤 + Token级损失

### 2.2 优势函数对比

| 算法 | 优势计算方式 |
|:---|:---|
| **PPO** | $A_t^{GAE} = \sum_{l=0}^{\infty}(\gamma\lambda)^l \delta_{t+l}$, 需要 $V_\phi(s)$ |
| **DPO** | 隐式: $\Delta r = \beta(\log r_w - \log r_l)$, 无显式优势 |
| **GRPO** | $\hat{A}_i = \frac{r_i - \text{mean}(r_{1:G})}{\text{std}(r_{1:G})}$, 组统计 |
| **DAPO** | $\hat{A}_i = \frac{r_i - \mu_{valid}}{\sigma_{valid}}$, 仅有效样本 |

### 2.3 KL散度对比

| 算法 | KL处理方式 |
|:---|:---|
| **PPO** | 无KL约束, 使用熵正则 $c_2 S[\pi_\theta]$ 替代 |
| **DPO** | 隐式约束, $\beta$ 控制偏离参考模型的程度 |
| **GRPO** | $D_{KL} = \frac{\pi_{ref}}{\pi_\theta} - \log\frac{\pi_{ref}}{\pi_\theta} - 1$ (序列级) |
| **DAPO** | $D_{KL}^{token} = \frac{1}{T}\sum_t KL_t$ (Token级平均) |

---

## 3. 可视化对比 / Visual Comparisons

### 3.1 损失曲线对比
![Loss Curves Comparison](figures/comparison_loss_curves_2d.png)

> **图释**: 四种算法的训练损失随步数变化曲线。PPO和GRPO初始损失较高(>2.0), DPO损失最低且收敛最快。DAPO在token级损失下表现出最稳定的下降趋势。

### 3.2 奖励曲线对比
![Reward Curves Comparison](figures/comparison_reward_curves_2d.png)

> **图释**: 四种算法的平均奖励提升曲线。DAPO最终奖励最高(>9.0), GRPO次之, PPO和DPO较为接近。DAPO的token级优化使其能更精确地优化奖励信号。

### 3.3 雷达图对比
![Radar Chart](figures/comparison_radar_chart.png)

> **图释**: 六维度对比 - 采样效率、训练速度、内存效率、实现简单性、最终性能、稳定性。DAPO在采样效率和最终性能上领先; DPO在稳定性和实现简单性上最优; PPO在最终性能上表现良好但内存效率最差。

### 3.4 柱状图对比
![Bar Chart](figures/comparison_bar_chart_2d.png)

> **图释**: 四维指标量化对比。PPO内存使用最高(因需价值网络); DAPO采样效率最高(90/100); GRPO在训练速度和最终奖励间取得最佳平衡。

### 3.5 三维损失曲面
![3D Surface](figures/comparison_3d_surface.png)

> **图释**: 损失关于学习率(log10)和裁剪参数(epsilon)的三维曲面。PPO对超参数最敏感(曲面最陡); DPO最为鲁棒; GRPO和DAPO在中等学习率附近有平坦的最优区域。

### 3.6 三维性能权衡
![3D Tradeoff](figures/comparison_3d_tradeoff.png)

> **图释**: 训练效率-最终性能-内存使用的三维散点图。每个点集群代表一个算法的多次运行。GRPO在效率-性能权衡上最优(高效率+高性能); PPO牺牲了内存和效率但获得良好性能。

---

## 4. 损失函数详细对比 / Loss Function Comparison

### 4.1 损失函数结构

```
PPO:    L = L_CLIP - c1*L_VF + c2*Entropy      (3个组件)
DPO:    L = -log sigma(beta*(log_rw - log_rl))   (1个组件)
GRPO:   L = L_CLIP + beta*KL                     (2个组件)
DAPO:   L = L_CLIP_token + beta*KL_token          (2个组件, token级)
```

### 4.2 梯度特性

| 特性 | PPO | DPO | GRPO | DAPO |
|:---|:---|:---|:---|:---|
| 梯度裁剪 | 概率比裁剪 | 无 | 概率比裁剪 | 概率比裁剪 |
| 梯度归一化 | 无 | 隐式(σ函数) | 无 | 长度归一化(1/|o_i|) |
| 梯度方差 | 中等 | 低 | 中等 | 低(token级) |

### 4.3 收敛特性

| 特性 | PPO | DPO | GRPO | DAPO |
|:---|:---|:---|:---|:---|
| 收敛速度 | 中等 | 快 | 中等 | 快 |
| 训练稳定性 | 良好 | 优秀 | 良好 | 良好 |
| 最优奖励 | 高 | 中等 | 高 | 最高 |
| 超参敏感度 | 中等 | 低 | 中等 | 中等 |

---

## 5. 计算资源对比 / Computational Resource Comparison

| 资源需求 | PPO | DPO | GRPO | DAPO |
|:---|:---|:---|:---|:---|
| **GPU显存** | 最高(2个网络) | 中等(2个模型) | 中等(2个模型) | 中等(2个模型) |
| **采样次数/问题** | 1 | 2 | G (通常16) | G (动态4~32) |
| **前向传播/步** | 3+次 | 4次 | 2G次 | 2G次 |
| **训练时间(相对)** | 1.0x | 0.7x | 1.2x | 1.3x |

---

## 6. 适用场景对比 / Use Case Comparison

| 场景 | 推荐算法 | 理由 |
|:---|:---|:---|
| 通用对齐 / General Alignment | DPO | 简单稳定, 无需在线采样 |
| 数学推理 / Math Reasoning | DAPO | Token级优化, 高最终奖励 |
| 代码生成 / Code Generation | GRPO | 平衡效率与性能 |
| 通用RLHF / Standard RLHF | PPO | 成熟稳定, 大量实践验证 |
| 资源受限 / Resource Limited | DPO | 最少计算资源需求 |
| 追求最优性能 | DAPO | 最高采样效率和最终奖励 |

---

## 7. 总结 / Summary

### 算法演进路线

```
PPO (2017)  →  DPO (2023)  →  GRPO (2025)  →  DAPO (2025)
  │                │                │                │
  ├─ 价值网络      ├─ 去除奖励模型   ├─ 去除价值网络   ├─ 动态采样
  ├─ GAE优势       ├─ 偏好对训练     ├─ 组相对优势     ├─ 过长过滤
  ├─ 裁剪目标      ├─ 隐式KL        ├─ 显式KL        ├─ Token级损失
  └─ 熵正则        └─ 简单稳定      └─ 高采样效率    └─ 最高性能
```

### 关键洞察

1. **价值网络不是必须的**: GRPO和DAPO通过组统计完全替代了价值网络, 大幅降低内存和计算开销
2. **Token级优化更精细**: DAPO的token级损失避免了长度偏差, 在推理任务上表现最优
3. **动态机制提升效率**: DAPO的动态组大小根据任务难度自适应调整, 兼顾效率和性能
4. **简单即最优(某些场景)**: DPO在简单对齐任务上仍然是最佳选择, 因为它不需要在线采样

---

*文档生成时间: 2025*
*项目地址: [PPOvDPOvGRPOvDAPO](https://github.com/aitachi/PPOvDPOvGRPOvDAPO)*
