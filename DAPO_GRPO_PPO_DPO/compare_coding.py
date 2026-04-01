import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import time
from torch.distributions import Categorical
from tqdm import tqdm
import os
import matplotlib

# 确保支持中文显示
matplotlib.use('Agg')  # 使用非交互式后端，避免显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False   # 用来正常显示负号

# 设置随机种子，确保结果可复现
def set_seed(seed):
    """设置所有随机种子以确保实验可复现"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


set_seed(42)

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")


# 模拟数据生成
class SimplifiedMathDataset:
    """简化的数学推理数据集，用于生成模拟推理任务"""

    def __init__(self, num_samples=1000, seq_length=20, vocab_size=1000, problem_type="sum"):
        """
        初始化数据集

        参数:
            num_samples: 样本数量
            seq_length: 序列长度
            vocab_size: 词汇表大小
            problem_type: 问题类型，可以是"sum"(求和)或"product"(求积)
        """
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.problem_type = problem_type

        # 生成模拟问题和相应的答案
        self.questions = []
        self.correct_answers = []
        self.question_texts = []

        for i in range(num_samples):
            # 模拟数学推理问题
            question = torch.randint(1, 10, (seq_length,))

            if problem_type == "sum":
                # 正确答案是序列中数字的总和
                correct_answer = question.sum().item()
                question_text = f"计算以下数字的总和: {', '.join(map(str, question.tolist()))}"
            elif problem_type == "product":
                # 正确答案是序列中数字的乘积
                correct_answer = torch.prod(question).item()
                question_text = f"计算以下数字的乘积: {', '.join(map(str, question.tolist()))}"
            else:
                raise ValueError(f"不支持的问题类型: {problem_type}")

            self.questions.append(question)
            self.correct_answers.append(correct_answer)
            self.question_texts.append(question_text)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return {
            'question': self.questions[idx],
            'correct_answer': self.correct_answers[idx],
            'question_text': self.question_texts[idx]
        }

    def get_batch(self, batch_size):
        """获取一批数据"""
        indices = np.random.choice(len(self), batch_size, replace=False)
        batch_questions = [self.questions[i] for i in indices]
        batch_correct_answers = [self.correct_answers[i] for i in indices]
        batch_question_texts = [self.question_texts[i] for i in indices]

        # 将问题列表转换为张量
        batch_questions_tensor = torch.stack(batch_questions).to(device)

        return {
            'questions': batch_questions_tensor,
            'correct_answers': torch.tensor(batch_correct_answers).to(device),
            'question_texts': batch_question_texts
        }

    def get_sample_data(self, num_samples=5):
        """获取示例数据用于可视化"""
        indices = np.random.choice(len(self), num_samples, replace=False)
        sample_questions = [self.questions[i].tolist() for i in indices]
        sample_answers = [self.correct_answers[i] for i in indices]
        sample_texts = [self.question_texts[i] for i in indices]

        return {
            'questions': sample_questions,
            'answers': sample_answers,
            'texts': sample_texts
        }


# 简化的策略网络模型
class PolicyNetwork(nn.Module):
    """简化的策略网络，用于生成回答"""

    def __init__(self, input_dim, hidden_dim, output_dim):
        """
        初始化策略网络

        参数:
            input_dim: 输入维度
            hidden_dim: 隐藏层维度
            output_dim: 输出维度
        """
        super(PolicyNetwork, self).__init__()

        self.embedding = nn.Embedding(input_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """前向传播"""
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)

        # 使用最后一个时间步的输出
        last_out = lstm_out[:, -1, :]

        x = F.relu(self.fc1(last_out))
        logits = self.fc2(x)

        return logits

    def get_action(self, state, sample=True):
        """根据状态获取动作"""
        logits = self.forward(state)
        probs = F.softmax(logits, dim=-1)

        if sample:
            m = Categorical(probs)
            action = m.sample()
            log_prob = m.log_prob(action)
            return action, log_prob
        else:
            return torch.argmax(probs, dim=-1), None


# 定义奖励函数
def compute_reward(predicted_answers, correct_answers, thresholds=(10, 5, 2)):
    """
    计算奖励值

    参数:
        predicted_answers: 模型预测的答案
        correct_answers: 正确答案
        thresholds: 不同误差范围的阈值

    返回:
        每个样本的奖励值
    """
    rewards = []
    for pred, target in zip(predicted_answers, correct_answers):
        # 计算预测与目标之间的误差
        error = abs(pred - target)

        # 根据误差分配奖励值
        if error == 0:
            reward = 10.0  # 完全正确
        elif error <= thresholds[2]:
            reward = 5.0  # 非常接近
        elif error <= thresholds[1]:
            reward = 2.0  # 接近
        elif error <= thresholds[0]:
            reward = 1.0  # 有些接近
        else:
            reward = 0.0  # 相差太远

        rewards.append(reward)

    return torch.tensor(rewards).to(device)


# GRPO算法实现
class GRPO:
    """
    Group Relative Policy Optimization算法实现

    GRPO是一种无需价值网络的策略优化方法，通过对每个问题采样多个回答，
    计算组内标准化的优势估计来更新策略。
    """

    def __init__(self, policy, optimizer, clip_epsilon=0.2, beta=0.01, group_size=16):
        """
        初始化GRPO

        参数:
            policy: 策略网络
            optimizer: 优化器
            clip_epsilon: 裁剪参数
            beta: KL散度系数
            group_size: 每个问题的采样组大小
        """
        self.policy = policy
        self.optimizer = optimizer
        self.clip_epsilon = clip_epsilon
        self.beta = beta
        self.group_size = group_size

    def update(self, questions, old_policy, dataset):
        """
        更新策略

        参数:
            questions: 问题批次
            old_policy: 旧策略
            dataset: 数据集

        返回:
            损失值
        """
        total_loss = 0

        for q_idx in range(questions.shape[0]):
            question = questions[q_idx].unsqueeze(0)
            correct_answer = dataset.correct_answers[q_idx]

            # 从旧策略采样一组输出
            sampled_actions = []
            sampled_log_probs = []

            for _ in range(self.group_size):
                with torch.no_grad():
                    action, log_prob = old_policy.get_action(question)
                    sampled_actions.append(action.item())
                    sampled_log_probs.append(log_prob.item())

            # 计算每个动作的奖励
            rewards = compute_reward(
                sampled_actions,
                [correct_answer] * self.group_size
            )

            # 计算标准化的优势 (这是GRPO的核心部分)
            advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

            # 计算当前策略下的新log概率
            curr_log_probs = []
            for action in sampled_actions:
                action_tensor = torch.tensor([action]).to(device)
                logits = self.policy(question)
                probs = F.softmax(logits, dim=-1)
                m = Categorical(probs)
                curr_log_prob = m.log_prob(action_tensor)
                curr_log_probs.append(curr_log_prob)

            # 计算比率
            old_log_probs_tensor = torch.tensor(sampled_log_probs).to(device)
            curr_log_probs_tensor = torch.stack(curr_log_probs).squeeze()

            ratios = torch.exp(curr_log_probs_tensor - old_log_probs_tensor)

            # GRPO的裁剪损失
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages

            # 参考策略是旧策略，计算KL散度
            kl_div = old_log_probs_tensor * (old_log_probs_tensor - curr_log_probs_tensor) - 1

            # 最终损失 = -min(surr1, surr2) + beta * KL
            policy_loss = -torch.min(surr1, surr2).mean() + self.beta * kl_div.mean()

            total_loss += policy_loss.item()

            # 反向传播和优化
            self.optimizer.zero_grad()
            policy_loss.backward()
            self.optimizer.step()

        return total_loss / questions.shape[0]


# PPO算法实现（用于比较）
class PPO:
    """
    Proximal Policy Optimization算法实现

    PPO是一种常用的策略优化方法，使用价值网络来估计状态值，
    并通过裁剪技术确保策略更新的稳定性。
    """

    def __init__(self, policy, critic, policy_optimizer, critic_optimizer, clip_epsilon=0.2, value_coef=0.5,
                 entropy_coef=0.01):
        """
        初始化PPO

        参数:
            policy: 策略网络
            critic: 价值网络
            policy_optimizer: 策略优化器
            critic_optimizer: 价值优化器
            clip_epsilon: 裁剪参数
            value_coef: 价值损失系数
            entropy_coef: 熵系数
        """
        self.policy = policy
        self.critic = critic
        self.policy_optimizer = policy_optimizer
        self.critic_optimizer = critic_optimizer
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef

    def update(self, questions, old_policy, dataset, batch_size=16, epochs=4):
        """
        更新策略和价值网络

        参数:
            questions: 问题批次
            old_policy: 旧策略
            dataset: 数据集
            batch_size: 小批次大小
            epochs: 训练周期数

        返回:
            策略损失值和价值损失值
        """
        total_policy_loss = 0
        total_value_loss = 0

        # 收集轨迹
        states = questions
        actions = []
        rewards = []
        old_log_probs = []
        values = []

        # 生成轨迹
        with torch.no_grad():
            for q_idx in range(states.shape[0]):
                question = states[q_idx].unsqueeze(0)
                correct_answer = dataset.correct_answers[q_idx]

                action, log_prob = old_policy.get_action(question)
                value = self.critic(question)

                actions.append(action.item())
                old_log_probs.append(log_prob.item())
                values.append(value.item())

                # 计算奖励
                reward = compute_reward([action.item()], [correct_answer])
                rewards.append(reward.item())

        # 将列表转换为张量
        actions = torch.tensor(actions).to(device)
        old_log_probs = torch.tensor(old_log_probs).to(device)
        rewards = torch.tensor(rewards).to(device)
        values = torch.tensor(values).to(device)

        # 计算优势
        advantages = rewards - values

        # 多个训练周期
        for _ in range(epochs):
            # 小批次更新
            for start_idx in range(0, states.shape[0], batch_size):
                end_idx = min(start_idx + batch_size, states.shape[0])
                batch_indices = range(start_idx, end_idx)

                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_rewards = rewards[batch_indices]
                batch_advantages = advantages[batch_indices]

                # 计算当前策略下的新log概率和熵
                batch_logits = self.policy(batch_states)
                probs = F.softmax(batch_logits, dim=-1)
                m = Categorical(probs)
                batch_curr_log_probs = m.log_prob(batch_actions)
                entropy = m.entropy().mean()

                # 计算比率
                ratios = torch.exp(batch_curr_log_probs - batch_old_log_probs)

                # PPO的裁剪损失
                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(ratios, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * batch_advantages

                # 策略损失 = -min(surr1, surr2) - entropy_coef * entropy
                policy_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy

                # 计算价值损失
                batch_values = self.critic(batch_states).squeeze()
                value_loss = F.mse_loss(batch_values, batch_rewards)

                # 总损失
                loss = policy_loss + self.value_coef * value_loss

                # 更新策略网络
                self.policy_optimizer.zero_grad()
                policy_loss.backward()
                self.policy_optimizer.step()

                # 更新价值网络
                self.critic_optimizer.zero_grad()
                value_loss.backward()
                self.critic_optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()

        avg_policy_loss = total_policy_loss / (states.shape[0] // batch_size * epochs)
        avg_value_loss = total_value_loss / (states.shape[0] // batch_size * epochs)

        return avg_policy_loss, avg_value_loss


# DPO算法实现（用于比较）
class DPO:
    """
    Direct Preference Optimization算法实现

    DPO是一种基于偏好的优化方法，通过比较两个回答的优劣来优化策略，
    无需显式的奖励建模。
    """

    def __init__(self, policy, optimizer, beta=0.1):
        """
        初始化DPO

        参数:
            policy: 策略网络
            optimizer: 优化器
            beta: 温度系数
        """
        self.policy = policy
        self.optimizer = optimizer
        self.beta = beta

    def update(self, questions, reference_policy, dataset):
        """
        更新策略

        参数:
            questions: 问题批次
            reference_policy: 参考策略
            dataset: 数据集

        返回:
            损失值
        """
        total_loss = 0

        for q_idx in range(questions.shape[0]):
            question = questions[q_idx].unsqueeze(0)
            correct_answer = dataset.correct_answers[q_idx]

            # 对每个问题，我们采样两个回答
            with torch.no_grad():
                action1, _ = reference_policy.get_action(question)
                action2, _ = reference_policy.get_action(question)

                # 计算这两个回答的奖励
                reward1 = compute_reward([action1.item()], [correct_answer])
                reward2 = compute_reward([action2.item()], [correct_answer])

            # 确定哪个是更好的回答（较高奖励）和较差的回答（较低奖励）
            if reward1 > reward2:
                chosen_action = action1
                rejected_action = action2
            else:
                chosen_action = action2
                rejected_action = action1

            # 计算策略下两个回答的对数概率
            logits = self.policy(question)
            probs = F.softmax(logits, dim=-1)

            log_prob_chosen = torch.log(probs[0, chosen_action] + 1e-8)
            log_prob_rejected = torch.log(probs[0, rejected_action] + 1e-8)

            # DPO损失：增加被选中回答的概率，减少被拒绝回答的概率
            loss = -torch.log(torch.sigmoid(self.beta * (log_prob_chosen - log_prob_rejected)) + 1e-8)

            total_loss += loss.item()

            # 反向传播和优化
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return total_loss / questions.shape[0]


# 创建可视化目录
os.makedirs('results', exist_ok=True)

# 创建模拟数据集并可视化
vocab_size = 1000
hidden_dim = 128
output_dim = 500  # 可能的答案范围，这里我们简化为0-499
max_answer = output_dim - 1

# 创建求和和求积两种类型的数据集
sum_dataset = SimplifiedMathDataset(num_samples=1000, seq_length=5, vocab_size=vocab_size, problem_type="sum")
product_dataset = SimplifiedMathDataset(num_samples=1000, seq_length=3, vocab_size=vocab_size, problem_type="product")

print(f"已创建求和数据集，包含 {len(sum_dataset)} 个样本")
print(f"已创建求积数据集，包含 {len(product_dataset)} 个样本")

# 可视化数据集样例
sum_samples = sum_dataset.get_sample_data(5)
product_samples = product_dataset.get_sample_data(5)

plt.figure(figsize=(15, 10))
plt.subplot(2, 1, 1)
plt.title("求和问题示例")
for i, (q, a, t) in enumerate(zip(sum_samples['questions'], sum_samples['answers'], sum_samples['texts'])):
    plt.text(0, i, f"问题{i + 1}: {t}\n答案: {a}", fontsize=12)
plt.axis('off')

plt.subplot(2, 1, 2)
plt.title("求积问题示例")
for i, (q, a, t) in enumerate(zip(product_samples['questions'], product_samples['answers'], product_samples['texts'])):
    plt.text(0, i, f"问题{i + 1}: {t}\n答案: {a}", fontsize=12)
plt.axis('off')

plt.tight_layout()
plt.savefig('results/data_samples.png')
print("已保存数据集样例图表到 results/data_samples.png")

# 创建训练和测试集
train_size = int(0.8 * len(sum_dataset))
test_size = len(sum_dataset) - train_size

# 我们使用求和数据集作为主要训练数据
train_dataset = SimplifiedMathDataset(num_samples=train_size, seq_length=5, vocab_size=vocab_size, problem_type="sum")
test_dataset = SimplifiedMathDataset(num_samples=test_size, seq_length=5, vocab_size=vocab_size, problem_type="sum")

print(f"训练集大小: {len(train_dataset)}, 测试集大小: {len(test_dataset)}")

# 初始化策略网络和优化器
policy_grpo = PolicyNetwork(vocab_size, hidden_dim, output_dim).to(device)
policy_ppo = PolicyNetwork(vocab_size, hidden_dim, output_dim).to(device)
policy_dpo = PolicyNetwork(vocab_size, hidden_dim, output_dim).to(device)
critic = PolicyNetwork(vocab_size, hidden_dim, 1).to(device)  # 价值网络只输出一个值

# 确保各策略网络初始状态相同
policy_ppo.load_state_dict(policy_grpo.state_dict())
policy_dpo.load_state_dict(policy_grpo.state_dict())

# 优化器
optimizer_grpo = optim.Adam(policy_grpo.parameters(), lr=0.001)
optimizer_ppo_policy = optim.Adam(policy_ppo.parameters(), lr=0.001)
optimizer_ppo_critic = optim.Adam(critic.parameters(), lr=0.001)
optimizer_dpo = optim.Adam(policy_dpo.parameters(), lr=0.001)

# 初始化算法
grpo = GRPO(policy_grpo, optimizer_grpo, clip_epsilon=0.2, beta=0.01, group_size=16)
ppo = PPO(policy_ppo, critic, optimizer_ppo_policy, optimizer_ppo_critic, clip_epsilon=0.2)
dpo = DPO(policy_dpo, optimizer_dpo, beta=0.1)

# 训练参数
num_epochs = 100
batch_size = 32
eval_interval = 10

# 存储结果
results = {
    'grpo': {'train_loss': [], 'eval_reward': [], 'eval_accuracy': [], 'train_time': []},
    'ppo': {'train_loss': [], 'value_loss': [], 'eval_reward': [], 'eval_accuracy': [], 'train_time': []},
    'dpo': {'train_loss': [], 'eval_reward': [], 'eval_accuracy': [], 'train_time': []}
}

# 记录训练开始时间
start_time = time.time()

# 训练循环
print(f"开始训练，共 {num_epochs} 个周期...")

for epoch in tqdm(range(num_epochs), desc="训练进度"):
    # 获取训练批次
    train_batch = train_dataset.get_batch(batch_size)
    questions = train_batch['questions']

    # 1. 更新GRPO
    # 创建旧策略的副本
    old_policy_grpo = PolicyNetwork(vocab_size, hidden_dim, output_dim).to(device)
    old_policy_grpo.load_state_dict(policy_grpo.state_dict())

    # 记录GRPO开始时间
    grpo_start = time.time()

    # 更新GRPO
    grpo_loss = grpo.update(questions, old_policy_grpo, train_dataset)
    results['grpo']['train_loss'].append(grpo_loss)

    # 记录GRPO训练时间
    results['grpo']['train_time'].append(time.time() - grpo_start)

    # 2. 更新PPO
    # 创建旧策略的副本
    old_policy_ppo = PolicyNetwork(vocab_size, hidden_dim, output_dim).to(device)
    old_policy_ppo.load_state_dict(policy_ppo.state_dict())

    # 记录PPO开始时间
    ppo_start = time.time()

    # 更新PPO
    ppo_policy_loss, ppo_value_loss = ppo.update(questions, old_policy_ppo, train_dataset)
    results['ppo']['train_loss'].append(ppo_policy_loss)
    results['ppo']['value_loss'].append(ppo_value_loss)

    # 记录PPO训练时间
    results['ppo']['train_time'].append(time.time() - ppo_start)

    # 3. 更新DPO
    # 创建参考策略的副本
    reference_policy = PolicyNetwork(vocab_size, hidden_dim, output_dim).to(device)
    reference_policy.load_state_dict(policy_dpo.state_dict())

    # 记录DPO开始时间
    dpo_start = time.time()

    # 更新DPO
    dpo_loss = dpo.update(questions, reference_policy, train_dataset)
    results['dpo']['train_loss'].append(dpo_loss)

    # 记录DPO训练时间
    results['dpo']['train_time'].append(time.time() - dpo_start)

    # 定期评估
    if (epoch + 1) % eval_interval == 0 or epoch == 0:
        print(f"周期 {epoch + 1}/{num_epochs}")

        # 获取评估批次
        eval_batch = test_dataset.get_batch(len(test_dataset))
        eval_questions = eval_batch['questions']
        eval_correct_answers = eval_batch['correct_answers']

        # 评估GRPO
        with torch.no_grad():
            grpo_actions, _ = policy_grpo.get_action(eval_questions, sample=False)
            grpo_rewards = compute_reward(grpo_actions.cpu().numpy(), eval_correct_answers.cpu().numpy())
            grpo_accuracy = (grpo_actions == eval_correct_answers).float().mean().item()

            results['grpo']['eval_reward'].append(grpo_rewards.mean().item())
            results['grpo']['eval_accuracy'].append(grpo_accuracy)

            print(f"GRPO - 平均奖励: {grpo_rewards.mean().item():.4f}, 准确率: {grpo_accuracy:.4f}")

        # 评估PPO
        with torch.no_grad():
            ppo_actions, _ = policy_ppo.get_action(eval_questions, sample=False)
            ppo_rewards = compute_reward(ppo_actions.cpu().numpy(), eval_correct_answers.cpu().numpy())
            ppo_accuracy = (ppo_actions == eval_correct_answers).float().mean().item()

            results['ppo']['eval_reward'].append(ppo_rewards.mean().item())
            results['ppo']['eval_accuracy'].append(ppo_accuracy)

            print(f"PPO - 平均奖励: {ppo_rewards.mean().item():.4f}, 准确率: {ppo_accuracy:.4f}")

        # 评估DPO
        with torch.no_grad():
            dpo_actions, _ = policy_dpo.get_action(eval_questions, sample=False)
            dpo_rewards = compute_reward(dpo_actions.cpu().numpy(), eval_correct_answers.cpu().numpy())
            dpo_accuracy = (dpo_actions == eval_correct_answers).float().mean().item()

            results['dpo']['eval_reward'].append(dpo_rewards.mean().item())
            results['dpo']['eval_accuracy'].append(dpo_accuracy)

            print(f"DPO - 平均奖励: {dpo_rewards.mean().item():.4f}, 准确率: {dpo_accuracy:.4f}")

        print("-" * 50)

# 记录训练总时间
training_duration = time.time() - start_time
print(f"训练完成！总训练时间: {training_duration:.2f}秒")

# 创建训练损失图表
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.plot(results['grpo']['train_loss'], label='GRPO')
plt.plot(results['ppo']['train_loss'], label='PPO')
plt.plot(results['dpo']['train_loss'], label='DPO')
plt.xlabel('训练周期')
plt.ylabel('训练损失')
plt.title('训练损失比较')
plt.legend()
plt.grid(True)

# 创建评估奖励图表
eval_epochs = list(range(0, num_epochs + 1, eval_interval))
if len(eval_epochs) > len(results['grpo']['eval_reward']):
    eval_epochs = eval_epochs[:len(results['grpo']['eval_reward'])]

plt.subplot(2, 2, 2)
plt.plot(eval_epochs, results['grpo']['eval_reward'], label='GRPO', marker='o')
plt.plot(eval_epochs, results['ppo']['eval_reward'], label='PPO', marker='s')
plt.plot(eval_epochs, results['dpo']['eval_reward'], label='DPO', marker='^')
plt.xlabel('训练周期')
plt.ylabel('平均奖励')
plt.title('评估奖励比较')
plt.legend()
plt.grid(True)

# 创建评估准确率图表
plt.subplot(2, 2, 3)
plt.plot(eval_epochs, results['grpo']['eval_accuracy'], label='GRPO', marker='o')
plt.plot(eval_epochs, results['ppo']['eval_accuracy'], label='PPO', marker='s')
plt.plot(eval_epochs, results['dpo']['eval_accuracy'], label='DPO', marker='^')
plt.xlabel('训练周期')
plt.ylabel('准确率')
plt.title('评估准确率比较')
plt.legend()
plt.grid(True)

# 如果有PPO的价值损失，也将其绘制出来
plt.subplot(2, 2, 4)
plt.plot(results['ppo']['value_loss'], label='PPO Value Loss')
plt.xlabel('训练周期')
plt.ylabel('价值损失')
plt.title('PPO价值网络损失')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('results/rl_algorithms_comparison.png')
print("已保存算法比较图表到 results/rl_algorithms_comparison.png")

# 创建最终结果的柱状图比较
final_results = {
    'Algorithm': ['GRPO', 'PPO', 'DPO'],
    'Final Reward': [results['grpo']['eval_reward'][-1],
                     results['ppo']['eval_reward'][-1],
                     results['dpo']['eval_reward'][-1]],
    'Final Accuracy': [results['grpo']['eval_accuracy'][-1],
                       results['ppo']['eval_accuracy'][-1],
                       results['dpo']['eval_accuracy'][-1]]
}

plt.figure(figsize=(12, 6))

# 最终奖励对比
plt.subplot(1, 2, 1)
bars = plt.bar(final_results['Algorithm'], final_results['Final Reward'], color=['#3498db', '#2ecc71', '#e74c3c'])
plt.ylabel('平均奖励')
plt.title('最终评估奖励比较')
plt.ylim([0, max(final_results['Final Reward']) * 1.2])

# 在每个柱子上方添加具体数值
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2., height + 0.1,
             f'{height:.4f}', ha='center', va='bottom')

# 最终准确率对比
plt.subplot(1, 2, 2)
bars = plt.bar(final_results['Algorithm'], final_results['Final Accuracy'], color=['#3498db', '#2ecc71', '#e74c3c'])
plt.ylabel('准确率')
plt.title('最终评估准确率比较')
plt.ylim([0, max(1.0, max(final_results['Final Accuracy']) * 1.2)])

# 在每个柱子上方添加具体数值
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2., height + 0.02,
             f'{height:.4f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('results/final_results_comparison.png')
print("已保存最终结果比较图表到 results/final_results_comparison.png")

# 创建训练时间比较图表
plt.figure(figsize=(10, 6))
avg_times = {
    'GRPO': np.mean(results['grpo']['train_time']),
    'PPO': np.mean(results['ppo']['train_time']),
    'DPO': np.mean(results['dpo']['train_time'])
}

plt.bar(avg_times.keys(), avg_times.values(), color=['#3498db', '#2ecc71', '#e74c3c'])
plt.ylabel('平均每轮训练时间 (秒)')
plt.title('算法训练时间比较')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 在每个柱子上方添加具体数值
for i, (alg, time_val) in enumerate(avg_times.items()):
    plt.text(i, time_val + 0.01, f'{time_val:.4f}s', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('results/training_time_comparison.png')
print("已保存训练时间比较图表到 results/training_time_comparison.png")

# 创建训练时间随周期变化的趋势图
plt.figure(figsize=(10, 6))

plt.plot(results['grpo']['train_time'], label='GRPO', marker='o', markersize=3)
plt.plot(results['ppo']['train_time'], label='PPO', marker='s', markersize=3)
plt.plot(results['dpo']['train_time'], label='DPO', marker='^', markersize=3)

plt.xlabel('训练周期')
plt.ylabel('训练时间 (秒)')
plt.title('算法训练时间随周期变化趋势')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('results/training_time_trends.png')
print("已保存训练时间趋势图表到 results/training_time_trends.png")

# 创建算法性能雷达图
categories = ['准确率', '奖励值', '训练速度', '内存效率', '实现复杂度']
N = len(categories)

# 为每个算法创建性能评分（这里使用模拟评分，根据实际结果调整）
grpo_scores = [results['grpo']['eval_accuracy'][-1],
               results['grpo']['eval_reward'][-1] / 10,  # 归一化奖励
               0.9,  # 训练速度评分（基于不需要价值网络）
               0.85,  # 内存效率评分（基于不需要额外网络）
               0.8]  # 实现复杂度评分（相对简单）

ppo_scores = [results['ppo']['eval_accuracy'][-1],
              results['ppo']['eval_reward'][-1] / 10,  # 归一化奖励
              0.7,  # 训练速度评分（需要价值网络，略慢）
              0.7,  # 内存效率评分（需要额外价值网络）
              0.65]  # 实现复杂度评分（较复杂）

dpo_scores = [results['dpo']['eval_accuracy'][-1],
              results['dpo']['eval_reward'][-1] / 10,  # 归一化奖励
              0.8,  # 训练速度评分
              0.8,  # 内存效率评分
              0.75]  # 实现复杂度评分

# 角度计算
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]  # 闭合雷达图

# 添加得分
grpo_scores += grpo_scores[:1]
ppo_scores += ppo_scores[:1]
dpo_scores += dpo_scores[:1]

# 绘制雷达图
plt.figure(figsize=(10, 8))
ax = plt.subplot(111, polar=True)

plt.xticks(angles[:-1], categories, size=12)
ax.set_rlabel_position(0)
plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.6", "0.8", "1.0"], color="grey", size=10)
plt.ylim(0, 1)

# 绘制每个算法的雷达图
ax.plot(angles, grpo_scores, 'o-', linewidth=2, label='GRPO', color='#3498db')
ax.fill(angles, grpo_scores, alpha=0.25, color='#3498db')

ax.plot(angles, ppo_scores, 'o-', linewidth=2, label='PPO', color='#2ecc71')
ax.fill(angles, ppo_scores, alpha=0.25, color='#2ecc71')

ax.plot(angles, dpo_scores, 'o-', linewidth=2, label='DPO', color='#e74c3c')
ax.fill(angles, dpo_scores, alpha=0.25, color='#e74c3c')

plt.title('GRPO, PPO 和 DPO 算法性能比较', size=15)
plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

plt.tight_layout()
plt.savefig('results/algorithm_radar_chart.png')
print("已保存算法性能雷达图到 results/algorithm_radar_chart.png")

# 创建总结表格
summary_data = {
    '算法': ['GRPO', 'PPO', 'DPO'],
    '最终准确率': [f"{results['grpo']['eval_accuracy'][-1]:.4f}",
                   f"{results['ppo']['eval_accuracy'][-1]:.4f}",
                   f"{results['dpo']['eval_accuracy'][-1]:.4f}"],
    '最终平均奖励': [f"{results['grpo']['eval_reward'][-1]:.4f}",
                     f"{results['ppo']['eval_reward'][-1]:.4f}",
                     f"{results['dpo']['eval_reward'][-1]:.4f}"],
    '平均训练时间/周期': [f"{np.mean(results['grpo']['train_time']):.4f}s",
                          f"{np.mean(results['ppo']['train_time']):.4f}s",
                          f"{np.mean(results['dpo']['train_time']):.4f}s"],
    '内存占用': ['低', '高', '中'],
    '实现复杂度': ['简单', '复杂', '中等']
}

# 创建DataFrame
summary_df = pd.DataFrame(summary_data)

# 保存为CSV
summary_df.to_csv('results/algorithm_comparison_summary.csv', index=False)

# 打印表格摘要
print("\n算法比较总结:")
print(summary_df.to_string(index=False))

# 保存最终模型
torch.save(policy_grpo.state_dict(), 'results/grpo_policy.pth')
torch.save(policy_ppo.state_dict(), 'results/ppo_policy.pth')
torch.save(policy_dpo.state_dict(), 'results/dpo_policy.pth')
torch.save(critic.state_dict(), 'results/ppo_critic.pth')

print("\n模型已保存到results目录。实验完成!")

# 将模拟数据样例转换为CSV
sum_samples_df = pd.DataFrame({
    'Question': sum_samples['texts'],
    'Answer': sum_samples['answers']
})

product_samples_df = pd.DataFrame({
    'Question': product_samples['texts'],
    'Answer': product_samples['answers']
})

sum_samples_df.to_csv('results/sum_samples.csv', index=False)
product_samples_df.to_csv('results/product_samples.csv', index=False)

print("模拟数据样例已保存为CSV文件。")

# 创建算法特点比较表
algorithm_features = {
    '特点': [
        '需要价值网络',
        '参数效率',
        '样本效率',
        '计算复杂度',
        '内存占用',
        '实现难度',
        '稳定性',
        '改进空间'
    ],
    'GRPO': [
        '否',
        '高',
        '中-高',
        '中',
        '低',
        '低',
        '高',
        '高'
    ],
    'PPO': [
        '是',
        '中',
        '中',
        '高',
        '高',
        '高',
        '中-高',
        '中'
    ],
    'DPO': [
        '否',
        '高',
        '低-中',
        '低',
        '中',
        '中',
        '中',
        '中'
    ]
}

algorithm_features_df = pd.DataFrame(algorithm_features)
algorithm_features_df.to_csv('results/algorithm_features.csv', index=False)

print("算法特点比较表已保存为CSV文件。")

# 绘制算法对比优劣图
plt.figure(figsize=(12, 8))

# 定义各算法的优缺点
algorithms = {
    'GRPO': {
        '优点': [
            '无需价值网络',
            '内存占用低',
            '训练速度快',
            '实现简单',
            '样本利用效率高'
        ],
        '缺点': [
            '调整组大小需要实验',
            '在某些特定任务上可能不如PPO'
        ]
    },
    'PPO': {
        '优点': [
            '理论基础扎实',
            '适用范围广',
            '稳定性较好',
            '大型强化学习研究中最常用'
        ],
        '缺点': [
            '需要额外价值网络',
            '内存占用大',
            '训练速度较慢',
            '实现复杂度高'
        ]
    },
    'DPO': {
        '优点': [
            '参数效率高',
            '直接优化偏好',
            '实现相对简单',
            '适合偏好学习'
        ],
        '缺点': [
            '依赖于高质量的偏好对',
            '难以处理连续型奖励信号',
            '在复杂任务上收敛可能不稳定',
            '样本效率较低'
        ]
    }
}

# 绘制优缺点对比
for i, (alg, pros_cons) in enumerate(algorithms.items()):
    plt.subplot(3, 1, i + 1)
    plt.title(f"{alg} 算法的优缺点", fontsize=14)

    pros = pros_cons['优点']
    cons = pros_cons['缺点']

    # 左侧显示优点，右侧显示缺点
    max_items = max(len(pros), len(cons))

    for j, pro in enumerate(pros):
        plt.text(0.01, 1 - (j + 1) / (max_items + 1), f"✅ {pro}", fontsize=12)

    for j, con in enumerate(cons):
        plt.text(0.5, 1 - (j + 1) / (max_items + 1), f"❌ {con}", fontsize=12)

    plt.axis('off')

plt.tight_layout()
plt.savefig('results/algorithm_pros_cons.png')
print("已保存算法优缺点对比图表到 results/algorithm_pros_cons.png")

print("\n所有实验数据和图表已保存到results目录。")
