# GRPO算法与PPO算法比较研究报告
基于DeepSeek-R1论文的强化学习算法分析

## 摘要
本报告详细分析了DeepSeek-R1论文中提出的GRPO（Group-based Reward Proximal Optimization）算法，并与经典的PPO（Proximal Policy Optimization）算法进行了深入比较。GRPO作为PPO的一个重要变体，主要在基线估计方法上进行了创新，通过同一问题的多个输出样本估计基线，从而消除了对独立价值网络的需求。本报告通过算法分析、数学原理、流程图解释以及模拟实验，全面展示了GRPO算法的优势、创新点及其在大型语言模型训练中的应用价值。

## 1. 算法原理介绍
### 1.1 PPO算法核心
PPO（Proximal Policy Optimization）是强化学习中广泛使用的策略优化算法，其核心在于通过近端策略优化来稳定训练过程。PPO通常使用一个独立的价值网络来估计状态价值函数，作为计算优势值的基线。
```python
def ppo_loss(policy, old_policy, states, actions, rewards, values):
    # 计算优势值
    advantages = rewards - values
    
    # 计算策略比率
    ratio = policy.prob(actions, states) / old_policy.prob(actions, states)
    
    # 计算裁剪后的目标函数
    clip_adv = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantages
    loss = -torch.min(ratio * advantages, clip_adv)
    
    # 添加KL惩罚项
    kl_div = calculate_kl_divergence(policy, reference_policy)
    
    return loss.mean() + beta * kl_div
```

### 1.2 GRPO算法核心
GRPO（Group-based Reward Proximal Optimization）是DeepSeek-R1论文中提出的PPO变体，其创新点在于基线估计方法。GRPO不使用独立的价值网络，而是从同一问题的多个输出样本中估计基线。
```python
def grpo_loss(policy, old_policy, questions, outputs_groups, rewards_groups, reference_policy=None):
    total_loss = 0
    
    for q_idx, question in enumerate(questions):
        outputs = outputs_groups[q_idx]  # 对同一问题的G个输出
        rewards = rewards_groups[q_idx]  # 对应的G个奖励
        
        # 计算这组输出的平均奖励和标准差作为基线
        mean_reward = np.mean(rewards)
        std_reward = np.std(rewards) + 1e-8  # 避免除零
        
        group_loss = 0
        for i, (output, reward) in enumerate(zip(outputs, rewards)):
            # 计算标准化的优势值
            advantage = (reward - mean_reward) / std_reward
            
            # 计算新旧策略的概率比
            ratio = policy.prob(output, question) / old_policy.prob(output, question)
            
            # 应用PPO的裁剪技巧
            clipped_ratio = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
            surrogate_loss = -torch.min(ratio * advantage, clipped_ratio * advantage)
            
            group_loss += surrogate_loss
        
        total_loss += group_loss / len(outputs)
    
    # 添加KL惩罚项，如果有参考策略
    if reference_policy is not None:
        kl_div = calculate_kl_divergence(policy, reference_policy)
        total_loss += beta * kl_div
    
    return total_loss / len(questions)
```
### 1.3 数学表达式
根据DeepSeek-R1论文，GRPO算法的数学表达式为：
![图片](https://github.com/user-attachments/assets/fdebef9a-03c8-443f-8e6c-0a73899b5d31)
其中，优势值 (Advantage) 通过组内标准化计算：
![图片](https://github.com/user-attachments/assets/19c0f5d6-e720-450d-8dfd-198d1fd7c1ce)

#### 公式解释
目标函数  由以下几部分组成：
期望计算：对问题分布  采样问题，对策略  采样 G 个回答 。
采用 Proximal Policy Optimization (PPO) 风格的目标，计算当前策略  相对于旧策略  的比率，并结合裁剪函数控制更新幅度。
KL 散度正则项  约束策略不偏离参考策略  过远，防止模型崩溃，系数  控制正则项强度。

#### 裁剪机制：

计算策略比率  来衡量策略变化。
使用裁剪函数  限制策略比率变动范围，防止更新过大。
选择未裁剪值和裁剪值的最小值，确保策略更新方向符合优势值 。

#### 优势计算：

组内标准化计算优势值 ，提升训练稳定性。
计算每个样本奖励  相对于该组奖励均值的偏差，并归一化。
该标准化方式减少不同问题的奖励尺度影响，使训练更加稳定。

#### 结论
GRPO 算法结合了 PPO 目标函数的裁剪优化方式，同时引入了组内标准化的优势值计算方式，并使用 KL 散度正则化约束策略更新，确保训练稳定性并提升模型性能。

## 2. GRPO与PPO的关键区别
<table>
  <tr style="background-color: #ADD8E6;">
    <th>比较维度</th>
    <th>PPO算法</th>
    <th>GRPO算法</th>
  </tr>
  <tr>
    <td>基线估计方法</td>
    <td>使用独立的价值网络估计基线</td>
    <td>使用同一问题多个回答的奖励分布估计基线</td>
  </tr>
  <tr>
    <td>计算资源需求</td>
    <td>需要额外的价值网络，通常与策略网络大小相当</td>
    <td>不需要额外的价值网络，节省约50%的计算资源</td>
  </tr>
  <tr>
    <td>基线估计稳定性</td>
    <td>可能受到价值网络训练质量影响</td>
    <td>基于实际奖励分布，在相似问题上可能提供更稳定的估计</td>
  </tr>
  <tr>
    <td>实现复杂度</td>
    <td>需要训练和维护额外的价值网络</td>
    <td>实现更简单，不需要额外的网络架构</td>
  </tr>
  <tr>
    <td>适用场景</td>
    <td>广泛适用于各类强化学习任务</td>
    <td>特别适合大型语言模型的训练，尤其是推理能力训练</td>
  </tr>
  <tr>
    <td>内存占用</td>
    <td>较高（需要存储额外的价值网络参数）</td>
    <td>较低（节省约50%的GPU内存）</td>
  </tr>
</table>

## 3. 算法流程对比
### 3.1 PPO算法流程图
![图片](https://github.com/user-attachments/assets/12ec5145-3cdc-41b0-881d-1cbcb06aa824)

### 3.2 GRPO算法流程
![图片](https://github.com/user-attachments/assets/d2b41f93-52e9-4336-ac3a-65a5fdc263b0)

### 3.3 关键流程区别解析
#### 数据收集方式：
- **PPO**：对每个状态采集一个动作和奖励
- **GRPO**：对每个问题采集多个（G 个）输出和对应奖励

#### 基线计算：
- **PPO**：使用单独的价值网络 \( V(s) \) 估计状态价值
- **GRPO**：使用同一问题下 \( G \) 个输出的平均奖励作为基线

#### 优势值计算：
- **PPO**：
 ![图片](https://github.com/user-attachments/assets/ff2789ff-9d33-4327-ba85-3cc7c57463ce)

- **GRPO**：
![图片](https://github.com/user-attachments/assets/44ddd433-51b1-44d5-abe1-a80f79b0e77b)

其中：
- \( A_{\text{PPO}}(s,a) \) 表示 PPO 算法的优势值计算。
- \( A_{\text{GRPO}}(s,a) \) 表示 GRPO 算法的优势值计算。
- \( r \) 为当前奖励值。
- \( V(s) \) 是状态 \( s \) 的价值估计（由单独的价值网络计算）。
- \( G \) 是 GRPO 采样的不同回答数量。
- \( r_1, r_2, \dots, r_G \) 是针对相同问题的不同回答对应的奖励值。
- \( \frac{1}{G} \sum_{i=1}^{G} r_i \) 计算的是这些奖励值的均值（mean）。
- 分母部分是标准差（std），用于对优势值进行归一化处理，提高稳定性。

## 4. 模拟实验分析
### 4.1 计算资源消耗对比
![图片](https://github.com/user-attachments/assets/7e1a2c9d-6079-49e8-b306-1876cee400a2)

### 4.2 训练稳定性对比
![图片](https://github.com/user-attachments/assets/1fcbf660-5ab8-4606-850b-6dae021753de)

### 4.3 性能表现对比
![图片](https://github.com/user-attachments/assets/3cbb7f5e-08ed-42ce-9364-6fc5a5b41065)

### 4.4 实验分析结论
基于上述模拟实验，我们可以得出以下结论：
#### 1.资源效率：GRPO通过省略价值网络，在内存使用和计算资源方面表现出明显优势，约节省50%的GPU内存。这使得在有限资源条件下可以训练更大的模型或使用更大的批量。
#### 2.训练稳定性：在训练过程中，GRPO展现出更稳定的学习曲线，波动更小，特别是在训练的初期阶段。这可能是由于基于组内奖励分布的基线估计方法更适合处理大型语言模型的训练。
#### 3.性能表现：从最终性能看，两种算法达到的最终性能水平相近，但GRPO在相同训练步数下可能达到更好的中期性能。DeepSeek-R1-Zero模型在AIME 2024上达到了71.0%的pass@1分数，与OpenAI-o1-0912相当，验证了GRPO方法的有效性。

## 5. GRPO算法的应用场景
### 5.1 大型语言模型训练
GRPO算法特别适合于大型语言模型的强化学习训练，尤其在以下方面：
推理能力提升训练
代码生成优化
数学问题求解能力训练
多轮对话能力训练

### 5.2 资源受限场景
在计算资源有限的情况下，GRPO相比PPO具有明显优势：
可以训练参数量更大的模型
可以使用更大的批量提高训练效率
减少内存占用，降低训练成本

## 6. 结论与展望
通过对GRPO和PPO算法的全面比较分析，我们可以得出以下结论：
#### 创新点：GRPO算法在基线估计方法上的创新，有效消除了对独立价值网络的需求，同时保持了PPO算法的核心优势。
#### 资源效率：GRPO显著降低了计算资源需求，节省约50%的GPU内存和计算资源，使得在大型语言模型上直接应用RL变得更加可行。
#### 实用价值：GRPO特别适合大型语言模型的推理能力训练，如DeepSeek-R1-Zero模型的成功训练所证明的。
#### 训练稳定性：基于组内奖励分布的基线估计可能提供更稳定的训练过程，特别是在处理相似问题时。
#### 未来展望：GRPO算法为大型语言模型的强化学习训练提供了一个更高效的方法，未来可能在以下方向继续发展：

##### 探索最佳的组大小G选择策略，平衡计算效率和基线估计质量
##### 结合其他先进的强化学习技术，如离线RL、模仿学习等
##### 针对特定任务（如数学推理、编程）进一步优化GRPO算法
##### 探索GRPO在多智能体系统中的应用可能


## 写在最后
看到这里，在你深入理解了GRPO和PPO的区别后，恭喜你，使用数据来告诉你GRPO只是训练数据的方式，而非提升大模型的方式，在推理训练中，其实没什么卵用[狗头]
所以李飞飞的s1中，并没有使用强化学习，直接使用了强COT数据，一样打赢了GPT4o
![图片](https://github.com/user-attachments/assets/cd16823f-229e-433f-aaca-c53ea306432a)
![图片](https://github.com/user-attachments/assets/69831a78-db21-4b1d-8fb0-c8659d6efd6f)
![图片](https://github.com/user-attachments/assets/2e40aa62-ba16-4196-8161-cb230821636f)
![图片](https://github.com/user-attachments/assets/4151a2dc-9e92-4c56-b916-488bcff5f642)

### 一些图
![model_comparison_table](https://github.com/user-attachments/assets/870c58a7-db98-4666-8811-bdf5e37dccdb)
![图片](https://github.com/user-attachments/assets/59edef6a-dd92-450f-a562-b80c56827b5e)
![图片](https://github.com/user-attachments/assets/5c29b054-a0ac-4302-a682-316550cb0364)














