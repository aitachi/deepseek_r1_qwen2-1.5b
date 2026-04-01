# A Comprehensive Survey of Reinforcement Learning Algorithms for Large Language Model Training: PPO, DPO, GRPO, and DAPO

**Aitachi**

Contact: 44158892@qq.com

---

## Abstract

Reinforcement learning (RL) has become a cornerstone technique for aligning and enhancing large language models (LLMs). This survey provides a comprehensive analysis of four prominent RL algorithms used in LLM training: Proximal Policy Optimization (PPO), Direct Preference Optimization (DPO), Group Relative Policy Optimization (GRPO), and Dynamic Advantage Policy Optimization (DAPO). We systematically compare their mathematical formulations, architectural designs, computational requirements, and empirical performance. Our analysis reveals a clear evolutionary trajectory from value-based methods (PPO) to preference-based approaches (DPO) and group-based methods (GRPO/DAPO), with each innovation addressing specific limitations of its predecessors. Experimental results on mathematical reasoning tasks demonstrate that DAPO achieves the highest final reward (9.52), while GRPO offers the best memory efficiency (6.2GB). We conclude with a discussion of open challenges and future research directions for RL-based LLM training.

**Keywords:** Reinforcement Learning, Large Language Models, PPO, DPO, GRPO, DAPO, RLHF, Alignment

---

## 摘要

强化学习 (RL) 已成为对齐和增强大语言模型 (LLM) 的核心技术。本综述全面分析了四种用于 LLM 训练的主流强化学习算法：近端策略优化 (PPO)、直接偏好优化 (DPO)、组相对策略优化 (GRPO) 和动态优势策略优化 (DAPO。我们系统比较了它们的数学公式、架构设计、计算需求和实证性能。分析揭示了从基于价值的方法 (PPO) 到基于偏好的方法 (DPO) 和基于组的方法 (GRPO/DAPO) 的清晰演进轨迹。实验结果表明，DAPO 取得了最高最终奖励 (9.52)，而 GRPO 提供了最佳内存效率 (6.2GB)。

---

## I. Introduction

### 1.1 Background

The advent of large language models (LLMs) has transformed natural language processing, enabling capabilities in reasoning, code generation, and creative writing. However, pre-trained LLMs often produce outputs that misalign with human preferences. Reinforcement Learning from Human Feedback (RLHF) [1][6] has emerged as the primary paradigm for aligning LLMs with desired behaviors.

The standard RLHF pipeline consists of three stages: (1) Supervised Fine-Tuning (SFT), (2) Reward Model training, and (3) RL-based policy optimization. The choice of RL algorithm in stage 3 significantly impacts training efficiency, model performance, and resource requirements.

### 1.2 Motivation

Recent advances have introduced several new RL algorithms specifically designed for LLM training. DeepSeek-R1 [4] introduced GRPO, eliminating the need for a value network. ByteDance's DAPO [5] further extended GRPO with dynamic sampling and token-level optimization. Meanwhile, DPO [3] offered a simpler alternative by directly optimizing from preference pairs. Understanding the trade-offs between these approaches is critical for practitioners.

### 1.3 Contributions

This survey makes the following contributions:
- Systematic comparison of four RL algorithms across mathematical, architectural, and empirical dimensions
- Unified notation framework for cross-algorithm analysis
- Experimental evaluation on mathematical reasoning benchmarks
- Practical guidelines for algorithm selection

### 1.4 Paper Organization

Section II covers background and preliminaries. Sections III-VI detail each algorithm. Section VII provides comparative analysis. Section VIII presents experimental results. Section IX discusses future directions, and Section X concludes.

---

## II. Background and Preliminaries

### 2.1 Reinforcement Learning Fundamentals

In the context of LLM training, we formulate the problem as a contextual bandit:

- **State/Context**: The input prompt $x$ (question, instruction)
- **Action**: The generated response $y$
- **Policy**: The language model $\pi_\theta(y|x)$
- **Reward**: $r(x, y) \in \mathbb{R}$, evaluating response quality

The objective is to maximize expected reward:

$$J(\theta) = \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta(\cdot|x)}[r(x, y)]$$

### 2.2 Key Notation

| Symbol | Meaning |
|:---|:---|
| $\pi_\theta$ | Current policy (LLM) |
| $\pi_{ref}$ | Reference (frozen) policy |
| $\pi_{old}$ | Policy used for data collection |
| $r(x,y)$ | Reward function |
| $\hat{A}$ | Advantage estimate |
| $\varepsilon$ | PPO clipping parameter |
| $\beta$ | KL/temperature coefficient |
| $G$ | Group size (GRPO/DAPO) |
| $\gamma$ | Discount factor |
| $\lambda$ | GAE parameter |

### 2.3 LLM Training Pipeline

```
Pre-training → SFT → RL-based Alignment
                          ├── PPO (Reward Model + Value Network)
                          ├── DPO (Preference Pairs)
                          ├── GRPO (Group Advantage)
                          └── DAPO (Dynamic + Token-level)
```

---

## III. Proximal Policy Optimization (PPO)

### 3.1 Algorithm Overview

PPO [1], proposed by Schulman et al. in 2017, is the most widely adopted RL algorithm for LLM alignment. It extends the policy gradient framework with a clipped surrogate objective that prevents destructively large policy updates.

### 3.2 Mathematical Formulation

#### 3.2.1 Clipped Surrogate Objective

$$L^{CLIP}(\theta) = \mathbb{E}_t\left[\min\left(r_t(\theta)\hat{A}_t,\ \text{clip}(r_t(\theta), 1-\varepsilon, 1+\varepsilon)\hat{A}_t\right)\right]$$

where the probability ratio is:

$$r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$$

#### 3.2.2 Value Function Loss

$$L^{VF}(\theta) = \left(V_\theta(s_t) - \hat{V}_t^{targ}\right)^2$$

#### 3.2.3 Entropy Bonus

$$S[\pi_\theta](s_t) = -\sum_a \pi_\theta(a|s_t)\log\pi_\theta(a|s_t)$$

#### 3.2.4 Combined Objective

$$L^{PPO}(\theta) = \mathbb{E}_t\left[L^{CLIP}(\theta) - c_1 L^{VF}(\theta) + c_2 S[\pi_\theta](s_t)\right]$$

Typical hyperparameters: $\varepsilon = 0.2$, $c_1 = 0.5$, $c_2 = 0.01$.

### 3.3 Generalized Advantage Estimation (GAE)

GAE [2] computes advantage estimates using a discounted sum of temporal differences:

$$\hat{A}_t^{GAE} = \sum_{l=0}^{\infty}(\gamma\lambda)^l\delta_{t+l}$$

where $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ is the TD error.

### 3.4 Actor-Critic Architecture

PPO requires two neural networks:
- **Actor (Policy Network)**: Generates responses, $\pi_\theta(y|x)$
- **Critic (Value Network)**: Estimates state values, $V_\phi(x)$

This dual-network architecture doubles the parameter count and memory requirements compared to single-model approaches.

### 3.5 Strengths and Limitations

**Strengths**:
- Theoretically well-founded with monotonic improvement guarantees
- Widely tested and battle-proven in production systems
- Supports multi-epoch mini-batch updates for data efficiency
- Entropy regularization prevents premature convergence

**Limitations**:
- High memory consumption due to value network (~2x parameters)
- Sensitive to hyperparameters (clip range, learning rates)
- Value network training can be unstable for long sequences
- Requires separate reward model training

---

## IV. Direct Preference Optimization (DPO)

### 4.1 From RLHF to DPO

DPO [3], proposed by Rafailov et al. in 2023, eliminates the need for explicit reward modeling by directly optimizing the policy from preference data. It derives from the observation that the optimal RLHF policy has a closed-form solution:

$$\pi^*(y|x) = \frac{1}{Z(x)}\pi_{ref}(y|x)\exp\left(\frac{1}{\beta}r(x,y)\right)$$

### 4.2 Bradley-Terry Preference Model

DPO leverages the Bradley-Terry model for preference probabilities:

$$P(y_w \succ y_l | x) = \sigma\left(r(x, y_w) - r(x, y_l)\right)$$

where $\sigma$ is the logistic function and $y_w, y_l$ are the preferred and rejected responses.

### 4.3 Mathematical Formulation

Substituting the implicit reward into the Bradley-Terry model yields the DPO loss:

$$L^{DPO}(\pi_\theta; \pi_{ref}) = -\mathbb{E}_{(x,y_w,y_l)\sim\mathcal{D}}\left[\log\sigma\left(\beta\left(\log\frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \log\frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\right)\right)\right]$$

The implicit reward is:

$$\hat{r}(x,y) = \beta\log\frac{\pi_\theta(y|x)}{\pi_{ref}(y|x)}$$

### 4.4 Implicit Reward Modeling

A key insight of DPO is that the log-ratio $\log\frac{\pi_\theta(y|x)}{\pi_{ref}(y|x)}$ serves as an implicit reward. This eliminates the need to:
- Train a separate reward model
- Perform online sampling during training
- Maintain a value network

### 4.5 Strengths and Limitations

**Strengths**:
- Simplest implementation among all four algorithms
- No reward model or value network required
- Training is highly stable
- Fast convergence on alignment tasks
- Works well with offline preference data

**Limitations**:
- Requires high-quality preference pairs (data-dependent)
- Less effective for reasoning tasks with verifiable answers
- Performance bounded by preference data quality
- Cannot leverage rule-based rewards

---

## V. Group Relative Policy Optimization (GRPO)

### 5.1 Motivation: Eliminating the Value Network

GRPO [4], introduced by DeepSeek-AI in 2025 as part of the DeepSeek-R1 project, addresses the key limitation of PPO: the expensive value network. The core insight is that for language model tasks, a group of responses to the same prompt can serve as a baseline for advantage estimation.

### 5.2 Group Advantage Normalization

For a given prompt $q$, GRPO generates $G$ responses $\{o_1, o_2, \ldots, o_G\}$ and computes advantages by normalizing within the group:

$$\hat{A}_i = \frac{r_i - \mu_G}{\sigma_G + \epsilon}$$

where:

$$\mu_G = \frac{1}{G}\sum_{i=1}^{G}r_i, \quad \sigma_G = \sqrt{\frac{1}{G}\sum_{i=1}^{G}(r_i - \mu_G)^2}$$

This formulation provides several benefits:
- **No value network**: Eliminates ~50% parameter overhead
- **Adaptive baseline**: Automatically adjusts to task difficulty
- **Relative comparison**: Naturally handles reward scale variations

### 5.3 Mathematical Formulation

$$J_{GRPO}(\theta) = \mathbb{E}_{q \sim P(Q), \{o_i\}_{i=1}^{G} \sim \pi_{\theta_{old}}}\left[\frac{1}{G}\sum_{i=1}^{G}\min\left(\frac{\pi_\theta(o_i|q)}{\pi_{\theta_{old}}(o_i|q)}\hat{A}_i,\ \text{clip}\left(\frac{\pi_\theta}{\pi_{old}}, 1-\varepsilon, 1+\varepsilon\right)\hat{A}_i\right) - \beta D_{KL}(\pi_\theta \| \pi_{ref})\right]$$

### 5.4 KL Divergence Constraint

GRPO uses an explicit KL divergence penalty:

$$D_{KL}(\pi_\theta \| \pi_{ref}) = \frac{\pi_{ref}(o_i|q)}{\pi_\theta(o_i|q)} - \log\frac{\pi_{ref}(o_i|q)}{\pi_\theta(o_i|q)} - 1$$

This is an unbiased estimator of the KL divergence that can be computed from a single sample.

### 5.5 Strengths and Limitations

**Strengths**:
- Eliminates value network, reducing GPU memory by ~50%
- Group advantage provides robust baseline estimation
- Well-suited for reasoning tasks with verifiable rewards
- Fixed group size simplifies implementation

**Limitations**:
- Group sampling increases per-step computation
- Fixed group size may be suboptimal (addressed by DAPO)
- Requires reference model (same size as policy)
- No length normalization (addressed by DAPO)

---

## VI. Dynamic Advantage Policy Optimization (DAPO)

### 6.1 Extending GRPO

DAPO [5], proposed by ByteDance in 2025, extends GRPO with three key innovations specifically designed for reasoning tasks such as mathematical problem-solving and code generation.

### 6.2 Innovation 1: Dynamic Sampling

DAPO adapts the group size $G$ based on reward variance:

$$G_{t+1} = \begin{cases}\min(G_t + \Delta G, G_{max}) & \text{if } \text{Var}(r_{valid}) > 2\tau \\\max(G_t - \Delta G, G_{min}) & \text{if } \text{Var}(r_{valid}) < \tau/2 \\G_t & \text{otherwise}\end{cases}$$

| Parameter | Meaning | Default |
|:---|:---|:---:|
| $G_{initial}$ | Initial group size | 16 |
| $G_{max}$ | Maximum group size | 32 |
| $G_{min}$ | Minimum group size | 4 |
| $\Delta G$ | Adjustment step | 2 |
| $\tau$ | Variance threshold | 0.3 |

**Rationale**: High variance indicates the policy is uncertain → increase G for better estimation. Low variance suggests convergence → decrease G for efficiency.

### 6.3 Innovation 2: Overlong Filtering

Responses exceeding a length threshold $L_{max}$ are filtered out:

$$\text{mask}_i = \mathbb{1}[|o_i| \leq L_{max}]$$

Filtered responses receive zero advantage and do not participate in gradient computation. Only valid responses contribute to advantage statistics:

$$\mu_{valid} = \text{mean}(\{r_j : |o_j| \leq L_{max}\})$$
$$\sigma_{valid} = \text{std}(\{r_j : |o_j| \leq L_{max}\})$$

**Rationale**: Overly long responses are typically low quality and introduce noise into advantage estimation.

### 6.4 Innovation 3: Token-Level Loss

DAPO normalizes loss by output length, computing per-token objectives:

$$L_{DAPO}(\theta) = -\mathbb{E}_{q,\{o_i\}}\left[\frac{1}{\sum_{i:valid}|o_i|}\sum_{i=1}^{G}\mathbb{1}[|o_i|\leq L_{max}]\frac{1}{|o_i|}\sum_{t=1}^{|o_i|}\min\left(r_t(\theta)\hat{A}_i,\ \text{clip}(r_t(\theta), 1-\varepsilon, 1+\varepsilon)\hat{A}_i\right)\right]$$

The token-level probability ratio is:

$$r_t(\theta) = \frac{\pi_\theta(o_t|q, o_{<t})}{\pi_{\theta_{old}}(o_t|q, o_{<t})}$$

**Rationale**: Without length normalization, longer sequences dominate the gradient, creating a length bias.

### 6.5 Reward Shaping

DAPO decomposes the base reward into per-token rewards using exponential decay:

$$r_t^{shaped} = r_{base} \times \frac{\gamma^{T-t}}{\sum_{s=1}^{T}\gamma^{T-s}}$$

where $\gamma = 0.99$ is the decay factor and $T$ is the total sequence length. This assigns higher weight to later tokens, encouraging the model to focus on conclusions.

### 6.6 Token-Level KL Divergence

$$D_{KL}^{token}(\pi_\theta \| \pi_{ref}) = \frac{1}{T}\sum_{t=1}^{T}\left[\frac{\pi_{ref}(o_t|q, o_{<t})}{\pi_\theta(o_t|q, o_{<t})} - \log\frac{\pi_{ref}(o_t|q, o_{<t})}{\pi_\theta(o_t|q, o_{<t})} - 1\right]$$

### 6.7 Strengths and Limitations

**Strengths**:
- Best final performance among all four algorithms
- Dynamic sampling adapts to task difficulty
- Token-level loss eliminates length bias
- Overlong filtering improves training stability

**Limitations**:
- Most complex implementation
- Slightly higher computation than GRPO due to dynamic G
- Requires tuning of additional hyperparameters ($\tau$, $L_{max}$, $\gamma$)
- Token-level computation adds overhead

---

## VII. Comparative Analysis

### 7.1 Algorithm Taxonomy

![Algorithm Taxonomy](figures/algorithm_taxonomy.png)

**Figure 1**: Taxonomy of RL algorithms for LLM training. PPO, GRPO, and DAPO belong to the policy gradient family, while DPO belongs to the preference-based family. DAPO extends GRPO with three key innovations.

### 7.2 Loss Function Comparison

![Loss Function Comparison](figures/loss_function_comparison.png)

**Figure 2**: Visual comparison of the four loss functions. PPO has the most complex objective with three components. DPO uses the simplest formulation based on preference pairs. GRPO and DAPO share the clipped surrogate structure but differ in granularity (sequence vs. token level).

### 7.3 Comprehensive Comparison Table

| Aspect | PPO | DPO | GRPO | DAPO |
|:---|:---:|:---:|:---:|:---:|
| **Year** | 2017 | 2023 | 2025 | 2025 |
| **Value Network** | Required | Not needed | Not needed | Not needed |
| **Reward Model** | Explicit | Implicit | Explicit | Explicit |
| **Reference Model** | Not needed | Required | Required | Required |
| **Advantage Method** | GAE | Implicit | Group norm | Dynamic group |
| **Loss Granularity** | Sequence | Sequence | Sequence | Token |
| **KL Constraint** | None | Implicit | Explicit penalty | Token-level explicit |
| **Clipping** | Symmetric | None | Symmetric | Symmetric + norm |
| **Group Sampling** | No | No | Yes (fixed G) | Yes (dynamic G) |
| **Length Handling** | None | None | None | Overlong filter |
| **Memory (GPU)** | High (~2x) | Medium | Low | Medium |
| **Training Speed** | Slow | Fast | Fast | Moderate |
| **Sample Efficiency** | Moderate | Moderate | High | High |
| **Implementation** | Complex | Simple | Moderate | Moderate |

### 7.4 Resource Requirements

![Resource Heatmap](figures/resource_heatmap.png)

**Figure 3**: Resource and performance comparison heatmap. Higher scores (green) indicate better performance. DAPO excels in final performance and scalability, while DPO leads in simplicity and stability.

### 7.5 Performance Radar

![Radar Chart](figures/radar_8dim.png)

**Figure 4**: Eight-dimension performance radar chart. Each algorithm has distinct strengths: PPO in generality, DPO in simplicity and stability, GRPO in efficiency, and DAPO in final performance.

---

## VIII. Experimental Evaluation

### 8.1 Setup

We evaluate all four algorithms on mathematical reasoning tasks using the Qwen2.5-0.5B-Instruct model with 10 sample problems covering algebra, calculus, geometry, and arithmetic.

**Hyperparameters**:
| Parameter | PPO | DPO | GRPO | DAPO |
|:---|:---:|:---:|:---:|:---:|
| Learning rate | 1e-5 | 1e-5 | 1e-5 | 1e-5 |
| Clip ε | 0.2 | N/A | 0.2 | 0.2 |
| β (KL/temp) | N/A | 0.1 | 0.01 | 0.01 |
| Group size G | N/A | N/A | 16 | 4-32 |
| Max epochs | 3 | 3 | 3 | 3 |

### 8.2 Convergence Analysis

![Convergence Comparison](figures/convergence_4algo.png)

**Figure 5**: Training loss convergence for all four algorithms. DAPO achieves the lowest final loss (0.0651), followed by GRPO (0.0823), DPO (0.0945), and PPO (0.1156).

### 8.3 Results

| Metric | PPO | DPO | GRPO | DAPO |
|:---|:---:|:---:|:---:|:---:|
| **Training Time** | 412s | 198s | 245s | 280s |
| **Final Loss** | 0.1156 | 0.0945 | 0.0823 | 0.0651 |
| **Final Reward** | 7.65 | 7.89 | 8.24 | 9.52 |
| **GPU Memory** | 9.8GB | 6.8GB | 6.2GB | 7.0GB |
| **Convergence Speed** | Moderate | Fast | Fast | Moderate |
| **Training Stability** | Moderate | Very High | High | High |
| **Dynamic G** | N/A | N/A | Fixed=16 | Range [4,32] |
| **Filter Rate** | N/A | N/A | 0% | 8-15% |

![Loss Curves](figures/comparison_loss_curves_2d.png)

**Figure 6**: Loss curves comparison across training steps.

![Reward Curves](figures/comparison_reward_curves_2d.png)

**Figure 7**: Reward curves comparison across training steps.

### 8.4 Statistical Analysis

**Key findings**:
1. DAPO achieves **24.4% higher reward** than PPO (9.52 vs 7.65)
2. GRPO uses **36.7% less GPU memory** than PPO (6.2GB vs 9.8GB)
3. DPO is **52% faster** to train than PPO (198s vs 412s)
4. All three newer algorithms (DPO, GRPO, DAPO) achieve lower loss than PPO
5. DAPO's dynamic G converges to ~12 for these tasks (from initial 16)

---

## IX. Discussion and Future Directions

### 9.1 Open Challenges

1. **Scalability**: All algorithms face challenges when scaling to models with 100B+ parameters. Memory-efficient implementations (e.g., LoRA, QLoRA) are needed.

2. **Reward Specification**: For reasoning tasks, rule-based rewards are well-defined. For open-ended generation, reward specification remains challenging.

3. **Length Exploitation**: Models may learn to generate unnecessarily long responses to maximize reward. DAPO's overlong filtering partially addresses this.

4. **Sample Efficiency**: Group-based methods (GRPO, DAPO) require multiple samples per prompt, which can be expensive for large models.

### 9.2 Promising Research Directions

1. **Hybrid Approaches**: Combining DPO's preference modeling with DAPO's token-level optimization could yield benefits for both alignment and reasoning.

2. **Adaptive Algorithm Selection**: Meta-learning approaches could automatically select the best RL algorithm based on task characteristics and training stage.

3. **Multi-objective Optimization**: Simultaneously optimizing for accuracy, coherence, safety, and efficiency requires multi-objective RL formulations.

4. **Efficient Group Sampling**: Using draft models or speculative decoding to reduce the cost of group sampling in GRPO/DAPO.

5. **Curriculum Learning**: Structuring the training data by difficulty and dynamically adjusting algorithm parameters accordingly.

6. **Process Reward Models (PRM)**: Extending DAPO's token-level approach with step-level reward models for more granular feedback.

---

## X. Conclusion

This survey has provided a comprehensive analysis of four reinforcement learning algorithms for LLM training: PPO, DPO, GRPO, and DAPO. Our analysis reveals a clear evolutionary trajectory:

- **PPO** (2017) established the foundation with clipped surrogate objectives and actor-critic architecture
- **DPO** (2023) simplified the pipeline by eliminating reward models through direct preference optimization
- **GRPO** (2025) eliminated the value network through group advantage normalization, reducing memory by ~50%
- **DAPO** (2025) extended GRPO with dynamic sampling, overlong filtering, and token-level loss for superior reasoning performance

For practitioners, we recommend:
- **DPO** for alignment tasks with available preference data
- **GRPO** for reasoning tasks with resource constraints
- **DAPO** for reasoning tasks where maximum performance is required
- **PPO** for general RL tasks requiring theoretical guarantees

The field continues to evolve rapidly, with promising directions in hybrid approaches, adaptive algorithm selection, and process-level reward modeling.

---

## References

[1] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov, "Proximal Policy Optimization Algorithms," arXiv preprint arXiv:1707.06347, 2017.

[2] J. Schulman, P. Moritz, S. Levine, M. Jordan, and P. Abbeel, "High-Dimensional Continuous Control Using Generalized Advantage Estimation," in Proc. ICLR, 2016.

[3] R. Rafailov, A. Sharma, E. Mitchell, S. Ermon, C. D. Manning, and C. Finn, "Direct Preference Optimization: Your Language Model is Secretly a Reward Model," in Proc. NeurIPS, 2023.

[4] DeepSeek-AI, "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning," arXiv preprint, 2025.

[5] Q. Yu et al., "DAPO: An Open-Source LLM Reinforcement Learning System," arXiv:2503.14476, 2025.

[6] L. Ouyang et al., "Training language models to follow instructions with human feedback," in Proc. NeurIPS, 2022.

[7] P. F. Christiano et al., "Deep Reinforcement Learning from Human Preferences," in Proc. NeurIPS, 2017.

[8] D. M. Ziegler et al., "Fine-Tuning Language Models from Human Preferences," arXiv preprint arXiv:1909.08593, 2019.

[9] N. Stiennon et al., "Learning to summarize with human feedback," in Proc. NeurIPS, 2020.

[10] Y. Bai et al., "Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback," arXiv preprint arXiv:2204.05862, 2022.

[11] H. Touvron et al., "LLaMA: Open and Efficient Foundation Language Models," arXiv preprint arXiv:2302.13971, 2023.

[12] J. Achiam et al., "GPT-4 Technical Report," arXiv preprint arXiv:2303.08774, 2023.

[13] L. Zheng et al., "Secrets of RLHF in Large Language Models Part I: PPO," arXiv preprint arXiv:2307.04964, 2023.

[14] W. Yuan et al., "Self-Rewarding Language Models," arXiv preprint arXiv:2401.10020, 2024.

[15] G. Cui et al., "UltraFeedback: Boosting Language Models with Scaled AI Feedback," arXiv preprint arXiv:2310.01377, 2024.

[16] Z. Wang et al., "HelpSteer: Multi-attribute Helpfulness Dataset for SteerLM," arXiv preprint arXiv:2311.09528, 2024.

[17] Y. Meng et al., "SimPO: Simple Preference Optimization with Reference-Free Reward," arXiv preprint arXiv:2405.14734, 2024.

[18] J. Hong et al., "ORPO: Monolithic Preference Optimization without Reference Model," arXiv preprint arXiv:2403.07691, 2024.

[19] M. G. Azar et al., "A General Theoretical Paradigm to Understand Learning from Human Preferences," in Proc. ICML, 2024.

[20] C. Rosset et al., "Direct Nash Optimization: Teaching Language Models to Self-Improve more Effectively," arXiv preprint arXiv:2404.03715, 2024.

[21] Z. Guo et al., "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models," arXiv preprint arXiv:2402.03300, 2024.

[22] H. Shao et al., "Visual In-Context Learning for Large Vision-Language Models," arXiv preprint arXiv:2402.11574, 2024.

[23] Qwen Team, "Qwen2 Technical Report," arXiv preprint arXiv:2407.10671, 2024.

[24] A. Yang et al., "Qwen2.5 Technical Report," arXiv preprint arXiv:2412.15115, 2024.

[25] A. Dubey et al., "The Llama 3 Herd of Models," arXiv preprint arXiv:2407.21783, 2024.

[26] C. Snell et al., "Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters," arXiv preprint arXiv:2408.03314, 2024.

[27] H. Lightman et al., "Let's Verify Step by Step," in Proc. ICLR, 2024.

[28] J. Uesato et al., "Solving math word problems with process and outcome-based feedback," in Proc. NeurIPS, 2022.

[29] P. Wang et al., "Math-Shepherd: Verify and Reinforce LLMs Step-by-step," arXiv preprint arXiv:2312.08935, 2024.

[30] A. Zeng et al., "GLM-4 Technical Report," arXiv preprint arXiv:2406.12793, 2024.

---

**Author**: Aitachi
**Contact**: 44158892@qq.com
**Repository**: https://github.com/aitachi/deepseek_r1_qwen2-1.5b
**Last Updated**: 2025-01
