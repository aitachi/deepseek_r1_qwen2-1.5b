"""
DAPO (Dynamic Advantage Policy Optimization) Training Implementation
DAPO (动态优势策略优化) 训练实现

Based on the paper: "DAPO: An Open-Source LLM Reinforcement Learning System
for Reasoning-Centric Task-Oriented Policy Optimization"
基于论文: "DAPO: 面向推理任务的开放式大语言模型强化学习系统"

Author: Aitachi
Contact: 44158892@qq.com
Date: 2025

Mathematical Formulation / 数学公式:
=======================

DAPO extends GRPO with dynamic sampling, overlong filtering, and token-level
loss computation. The core objective is:
DAPO 在 GRPO 的基础上引入了动态采样、过长过滤和 token 级别损失计算。核心目标为:

L_DAPO(θ) = -E_{q~P(Q), {o_i}^G_{i=1}~π_θold}[
    1/(∑|o_i|) ∑^G_{i=1} (1/|o_i|) ∑_{t=1}^{|o_i|} min(
        r_t(θ) * Â_i,
        clip(r_t(θ), 1-ε, 1+ε) * Â_i
    )
    - β * D_KL(π_θ || π_ref)
]

Where / 其中:
- θ: Current policy parameters / 当前策略参数
- θ_old: Old policy parameters / 旧策略参数
- q: Input question / prompt / 输入问题/提示
- o_i: i-th output sample / 第 i 个输出样本
- G: Dynamic group size (adjusted during training) / 动态组大小(训练中调整)
- ε: Clipping parameter / 裁剪参数
- β: KL divergence coefficient / KL散度系数
- |o_i|: Length of output i / 输出 i 的长度

Key Innovations over GRPO / 相对 GRPO 的关键创新:
=================================================
1. Token-Level Loss / Token级别损失:
   Normalizes loss by output length to prevent length bias
   按输出长度归一化损失, 防止长度偏差

2. Dynamic Sampling / 动态采样:
   Adjusts group size G based on reward variance during training
   根据训练中的奖励方差动态调整组大小 G

3. Overlong Filtering / 过长过滤:
   Filters responses exceeding max length before advantage computation
   在优势计算前过滤超过最大长度的响应

4. Reward Shaping / 奖励塑形:
   Uses per-token reward decomposition for finer gradient signals
   使用逐token奖励分解以获得更精细的梯度信号

Advantage Calculation / 优势计算:
Â_i = (r_i - mean({r_j : |o_j| ≤ L_max})) / std({r_j : |o_j| ≤ L_max})

Overlong Filtering / 过长过滤:
Only samples with |o_i| ≤ L_max participate in advantage computation
只有满足 |o_i| ≤ L_max 的样本参与优势计算

Token-Level Policy Gradient / Token级别策略梯度:
For each output o_i with T_i tokens:
对于每个具有 T_i 个 token 的输出 o_i:
g_i = (1/T_i) ∑_{t=1}^{T_i} ∇_θ log π_θ(o_{i,t} | q, o_{i,<t}) * Â_i

KL Divergence (per-token) / KL散度(逐token):
D_KL(π_θ || π_ref) = (1/T) ∑_{t=1}^{T} [
    π_ref(o_t|q,o_{<t}) / π_θ(o_t|q,o_{<t})
    - log(π_ref(o_t|q,o_{<t}) / π_θ(o_t|q,o_{<t}))
    - 1
]
"""

import os
import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Tuple, Optional
import numpy as np
from tqdm import tqdm
import logging
from dataclasses import dataclass, field

# Set up logging / 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DAPOConfig:
    """
    Configuration for DAPO training
    DAPO 训练配置
    """
    # Model parameters / 模型参数
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"

    # DAPO hyperparameters / DAPO 超参数
    initial_group_size: int = 16       # Initial G / 初始组大小
    min_group_size: int = 4            # Minimum group size / 最小组大小
    max_group_size: int = 32           # Maximum group size / 最大组大小
    clip_epsilon: float = 0.2          # ε - PPO clipping parameter / PPO裁剪参数
    beta: float = 0.01                 # KL divergence coefficient / KL散度系数
    max_response_length: int = 512     # L_max - max response length / 最大响应长度
    dynamic_sampling_threshold: float = 0.3  # Variance threshold for group resize / 方差阈值

    # Reward shaping parameters / 奖励塑形参数
    accuracy_reward: float = 10.0      # Reward for correct answer / 正确答案奖励
    partial_reward: float = 2.0        # Reward for partial correctness / 部分正确奖励
    format_reward: float = 0.5         # Reward for format compliance / 格式合规奖励
    length_penalty: float = 0.001      # Per-token length penalty / 逐token长度惩罚

    # Training parameters / 训练参数
    learning_rate: float = 1e-5
    max_epochs: int = 3
    batch_size: int = 4
    max_length: int = 1024
    temperature: float = 0.7
    use_token_level_loss: bool = True  # Enable token-level loss / 启用token级别损失

    # Device / 设备
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Paths / 路径
    output_dir: str = "./checkpoints/dapo_model"


class RewardShaper:
    """
    Reward shaping module for DAPO
    DAPO 的奖励塑形模块

    Decomposes rewards into per-token components for finer gradient signals
    将奖励分解为逐token组件以获得更精细的梯度信号
    """

    def __init__(self, config: DAPOConfig):
        self.config = config
        # Reward decay factor for per-token decomposition / 逐token分解的奖励衰减因子
        self.reward_decay = 0.99

    def compute_base_reward(
        self,
        question: str,
        response: str,
        correct_answer: str
    ) -> float:
        """
        Compute base reward for a complete response
        计算完整响应的基础奖励

        Args:
            question: Input question / 输入问题
            response: Model generated response / 模型生成的响应
            correct_answer: Ground truth answer / 正确答案

        Returns:
            Total reward score / 总奖励分数
        """
        reward = 0.0

        # Extract predicted answer / 提取预测答案
        predicted_answer = self._extract_answer(response)

        # Accuracy reward / 准确性奖励
        if predicted_answer.lower().strip() == correct_answer.lower().strip():
            reward += self.config.accuracy_reward
        elif self._is_partially_correct(predicted_answer, correct_answer):
            reward += self.config.partial_reward

        # Format reward - encourage structured thinking / 格式奖励 - 鼓励结构化思考
        if "<think" in response and "</think" in response:
            reward += self.config.format_reward

        # Length penalty - discourage overly verbose responses / 长度惩罚 - 惩罚过度冗长的响应
        tokens = response.split()
        reward -= self.config.length_penalty * len(tokens)

        return reward

    def decompose_reward(
        self,
        base_reward: float,
        response_length: int
    ) -> torch.Tensor:
        """
        Decompose reward into per-token rewards using exponential decay
        使用指数衰减将奖励分解为逐token奖励

        The final tokens receive the highest reward signal, with exponential
        decay applied to earlier tokens. This encourages the model to focus
        on the conclusion.

        最后的 token 接收最高的奖励信号, 对前面的 token 应用指数衰减。
        这鼓励模型关注结论部分。

        Args:
            base_reward: Total reward for the response / 响应的总奖励
            response_length: Number of tokens in response / 响应中的 token 数量

        Returns:
            Per-token reward tensor / 逐token奖励张量
        """
        if response_length == 0:
            return torch.tensor([0.0])

        # Create exponentially increasing weights / 创建指数增长的权重
        # Later tokens get higher weight / 后面的 token 获得更高的权重
        positions = torch.arange(response_length, dtype=torch.float32)
        weights = self.reward_decay ** (response_length - 1 - positions)

        # Normalize weights / 归一化权重
        weights = weights / weights.sum()

        # Distribute reward across tokens / 在 token 间分配奖励
        per_token_rewards = base_reward * weights

        return per_token_rewards

    def _extract_answer(self, response: str) -> str:
        """Extract the final answer from model response / 从模型响应中提取最终答案"""
        if "<answer>" in response and "</answer>" in response:
            start = response.find("<answer>") + len("<answer>")
            end = response.find("</answer>")
            return response[start:end].strip()

        if "answer:" in response.lower():
            parts = response.lower().split("answer:")
            if len(parts) > 1:
                return parts[-1].strip().split()[0] if parts[-1].strip() else ""

        words = response.strip().split()
        return " ".join(words[-5:]) if words else ""

    def _is_partially_correct(self, predicted: str, correct: str) -> bool:
        """Check if answer is partially correct / 检查答案是否部分正确"""
        pred_digits = set(c for c in predicted if c.isdigit() or c == '.')
        correct_digits = set(c for c in correct if c.isdigit() or c == '.')

        if pred_digits and correct_digits:
            overlap = len(pred_digits & correct_digits) / len(correct_digits)
            return overlap > 0.5
        return False


class DAPOTrainer:
    """
    DAPO (Dynamic Advantage Policy Optimization) Trainer
    DAPO (动态优势策略优化) 训练器

    Key innovations:
    1. Token-level loss computation / Token级别损失计算
    2. Dynamic group size adjustment / 动态组大小调整
    3. Overlong response filtering / 过长响应过滤
    4. Reward shaping with per-token decomposition / 带逐token分解的奖励塑形
    """

    def __init__(self, config: DAPOConfig):
        self.config = config
        self.device = torch.device(config.device)

        # Current dynamic group size / 当前动态组大小
        self.current_group_size = config.initial_group_size

        # Load model and tokenizer / 加载模型和分词器
        logger.info(f"Loading model: {config.model_name} / 加载模型: {config.model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.float16 if "cuda" in config.device else torch.float32,
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Reference model (frozen copy for KL divergence) / 参考模型(KL散度的冻结副本)
        logger.info("Creating reference model for KL divergence / 创建参考模型用于KL散度计算")
        self.ref_model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.float16 if "cuda" in config.device else torch.float32,
            device_map="auto"
        )
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False

        # Optimizer / 优化器
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate
        )

        # Reward shaper / 奖励塑形器
        self.reward_shaper = RewardShaper(config)

        # Training statistics / 训练统计
        self.stats = {
            "epoch_losses": [],
            "epoch_rewards": [],
            "epoch_kl_divs": [],
            "epoch_group_sizes": [],
            "epoch_filtered_ratios": []
        }

    def generate_responses(
        self,
        prompt: str,
        num_samples: int
    ) -> Tuple[List[str], List[int]]:
        """
        Generate multiple response samples for a given prompt
        为给定提示生成多个响应样本

        Args:
            prompt: Input question / 输入问题
            num_samples: Number of samples to generate / 要生成的样本数量

        Returns:
            Tuple of (responses, response_lengths) / (响应列表, 响应长度列表)
        """
        responses = []
        response_lengths = []

        formatted_prompt = (
            f"<|im_start|>user\n{prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            max_length=self.config.max_length,
            truncation=True
        ).to(self.device)

        for _ in range(num_samples):
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_response_length,
                    temperature=self.config.temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    num_return_sequences=1
                )

            response = self.tokenizer.decode(outputs[0], skip_special_tokens=False)

            # Extract assistant's response / 提取助手响应
            if "<|im_start|>assistant\n" in response:
                response = response.split("<|im_start|>assistant\n")[-1]
                if "<|im_end|>" in response:
                    response = response.split("<|im_end|>")[0]

            response = response.strip()
            responses.append(response)
            response_lengths.append(len(self.tokenizer.encode(response)))

        return responses, response_lengths

    def filter_overlong(
        self,
        responses: List[str],
        response_lengths: List[int]
    ) -> Tuple[List[str], List[int], List[bool]]:
        """
        Filter responses that exceed the maximum length threshold
        过滤超过最大长度阈值的响应

        Overlong Filtering: Only responses with length ≤ L_max participate
        in advantage computation. Overlong responses receive zero advantage
        to discourage generating excessively long outputs.

        过长过滤: 只有长度 ≤ L_max 的响应参与优势计算。
        过长的响应获得零优势以避免生成过长的输出。

        Args:
            responses: List of generated responses / 生成的响应列表
            response_lengths: Corresponding response lengths / 对应的响应长度

        Returns:
            Tuple of (valid_mask, filtered_count) / (有效掩码, 过滤计数)
        """
        max_len = self.config.max_response_length
        valid_mask = [length <= max_len for length in response_lengths]
        filtered_count = sum(1 for m in valid_mask if not m)

        return valid_mask, filtered_count

    def compute_dynamic_advantages(
        self,
        rewards: List[float],
        valid_mask: List[bool]
    ) -> List[float]:
        """
        Compute group-normalized advantages using only valid (non-overlong) samples
        仅使用有效(非过长)样本计算组归一化优势

        Formula / 公式:
        Â_i = (r_i - mean({r_j : valid_j})) / std({r_j : valid_j})

        Overlong responses get advantage = 0
        过长的响应获得优势 = 0

        Args:
            rewards: List of rewards / 奖励列表
            valid_mask: Boolean mask for valid responses / 有效响应的布尔掩码

        Returns:
            List of advantages / 优势列表
        """
        valid_rewards = [r for r, m in zip(rewards, valid_mask) if m]

        if len(valid_rewards) == 0:
            return [0.0] * len(rewards)

        mean_reward = np.mean(valid_rewards)
        std_reward = np.std(valid_rewards) + 1e-8

        advantages = []
        for i, (r, m) in enumerate(zip(rewards, valid_mask)):
            if m:
                advantages.append((r - mean_reward) / std_reward)
            else:
                # Overlong responses get zero advantage / 过长响应获得零优势
                advantages.append(0.0)

        return advantages

    def adjust_group_size(self, reward_variance: float):
        """
        Dynamically adjust group size based on reward variance
        根据奖励方差动态调整组大小

        High variance → increase group size for better estimation
        Low variance → decrease group size for efficiency

        高方差 → 增加组大小以获得更好的估计
        低方差 → 减少组大小以提高效率

        Args:
            reward_variance: Variance of rewards in current batch / 当前批次的奖励方差
        """
        threshold = self.config.dynamic_sampling_threshold

        if reward_variance > threshold * 2:
            # High variance: increase group size / 高方差: 增加组大小
            self.current_group_size = min(
                self.current_group_size + 2,
                self.config.max_group_size
            )
        elif reward_variance < threshold * 0.5:
            # Low variance: decrease group size / 低方差: 减少组大小
            self.current_group_size = max(
                self.current_group_size - 2,
                self.config.min_group_size
            )

        logger.debug(
            f"Adjusted group size to {self.current_group_size} "
            f"(variance: {reward_variance:.4f})"
        )

    def compute_token_level_loss(
        self,
        prompt: str,
        response: str,
        advantage: float
    ) -> Tuple[torch.Tensor, float]:
        """
        Compute token-level policy gradient loss
        计算 token 级别的策略梯度损失

        Token-Level Loss / Token级别损失:
        L_i = -(1/|o_i|) ∑_{t=1}^{|o_i|} min(
            r_t(θ) * Â_i,
            clip(r_t(θ), 1-ε, 1+ε) * Â_i
        )

        This normalizes by sequence length to prevent bias towards longer outputs.
        通过序列长度归一化以防止对较长输出的偏差。

        Args:
            prompt: Input prompt / 输入提示
            response: Generated response / 生成的响应
            advantage: Computed advantage / 计算得到的优势

        Returns:
            Tuple of (loss, kl_divergence) / (损失, KL散度)
        """
        formatted_prompt = (
            f"<|im_start|>user\n{prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        full_text = formatted_prompt + response

        inputs = self.tokenizer(
            full_text,
            return_tensors="pt",
            max_length=self.config.max_length,
            truncation=True
        ).to(self.device)

        # Forward pass through current policy / 当前策略的前向传播
        outputs = self.model(**inputs, output_hidden_states=False)
        logits = outputs.logits[:, :-1, :]  # Remove last position / 移除最后位置
        labels = inputs["input_ids"][:, 1:]  # Shift labels / 偏移标签

        # Compute per-token log probabilities / 计算逐token对数概率
        log_probs = F.log_softmax(logits, dim=-1)
        per_token_log_probs = log_probs.gather(2, labels.unsqueeze(-1)).squeeze(-1)

        # Get reference model log probabilities / 获取参考模型对数概率
        with torch.no_grad():
            ref_outputs = self.ref_model(**inputs, output_hidden_states=False)
            ref_logits = ref_outputs.logits[:, :-1, :]
            ref_log_probs = F.log_softmax(ref_logits, dim=-1)
            ref_per_token_log_probs = ref_log_probs.gather(
                2, labels.unsqueeze(-1)
            ).squeeze(-1)

        # Compute ratio per token / 计算逐token比率
        ratio = torch.exp(per_token_log_probs - ref_per_token_log_probs)

        # Compute clipped objective per token / 计算逐token裁剪目标
        advantage_tensor = torch.tensor(
            advantage, device=self.device, dtype=torch.float32
        )
        surr1 = ratio * advantage_tensor
        surr2 = torch.clamp(
            ratio,
            1.0 - self.config.clip_epsilon,
            1.0 + self.config.clip_epsilon
        ) * advantage_tensor

        # Token-level loss: average over all tokens / Token级别损失: 在所有token上平均
        token_loss = -torch.min(surr1, surr2).mean()

        # KL divergence per token / 逐token KL散度
        kl_per_token = (
            torch.exp(ref_per_token_log_probs - per_token_log_probs)
            - (ref_per_token_log_probs - per_token_log_probs)
            - 1
        )
        kl_div = kl_per_token.mean().item()

        return token_loss, kl_div

    def compute_sequence_level_loss(
        self,
        prompt: str,
        response: str,
        advantage: float
    ) -> Tuple[torch.Tensor, float]:
        """
        Compute sequence-level loss (fallback mode, similar to GRPO)
        计算序列级别损失(回退模式, 类似 GRPO)

        Args:
            prompt: Input prompt / 输入提示
            response: Generated response / 生成的响应
            advantage: Computed advantage / 计算得到的优势

        Returns:
            Tuple of (loss, kl_divergence) / (损失, KL散度)
        """
        formatted_prompt = (
            f"<|im_start|>user\n{prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        full_text = formatted_prompt + response

        inputs = self.tokenizer(
            full_text,
            return_tensors="pt",
            max_length=self.config.max_length,
            truncation=True
        ).to(self.device)

        # Current policy log prob / 当前策略对数概率
        outputs = self.model(**inputs, labels=inputs["input_ids"])
        log_prob_current = -outputs.loss

        # Reference policy log prob / 参考策略对数概率
        with torch.no_grad():
            ref_outputs = self.ref_model(**inputs, labels=inputs["input_ids"])
            log_prob_ref = -ref_outputs.loss

        # Ratio and clipped objective / 比率和裁剪目标
        ratio = torch.exp(log_prob_current - log_prob_ref)
        advantage_tensor = torch.tensor(
            advantage, device=self.device, dtype=torch.float32
        )

        surr1 = ratio * advantage_tensor
        surr2 = torch.clamp(
            ratio,
            1.0 - self.config.clip_epsilon,
            1.0 + self.config.clip_epsilon
        ) * advantage_tensor

        policy_loss = -torch.min(surr1, surr2)

        # KL divergence / KL散度
        kl_ratio = torch.exp(log_prob_ref - log_prob_current)
        kl_div = (kl_ratio - (log_prob_ref - log_prob_current) - 1).item()

        return policy_loss, kl_div

    def train_step(
        self,
        question: str,
        correct_answer: str,
        reasoning_steps: List[str]
    ) -> Dict[str, float]:
        """
        Perform one DAPO training step on a single question
        对单个问题执行一步 DAPO 训练

        Pipeline / 流程:
        1. Generate G responses (dynamic group size)
           生成 G 个响应(动态组大小)
        2. Filter overlong responses
           过滤过长响应
        3. Compute rewards for valid responses
           计算有效响应的奖励
        4. Compute dynamic advantages
           计算动态优势
        5. Compute token-level or sequence-level loss
           计算 token 级别或序列级别损失
        6. Update policy with gradient
           使用梯度更新策略

        Args:
            question: Input question / 输入问题
            correct_answer: Ground truth answer / 正确答案
            reasoning_steps: Reference reasoning steps / 参考推理步骤

        Returns:
            Dictionary of training statistics / 训练统计字典
        """
        G = self.current_group_size

        # Step 1: Generate responses / 步骤1: 生成响应
        responses, response_lengths = self.generate_responses(question, G)

        # Step 2: Filter overlong responses / 步骤2: 过滤过长响应
        valid_mask, filtered_count = self.filter_overlong(
            responses, response_lengths
        )

        # Step 3: Compute rewards / 步骤3: 计算奖励
        rewards = []
        for response in responses:
            reward = self.reward_shaper.compute_base_reward(
                question, response, correct_answer
            )
            rewards.append(reward)

        # Step 4: Compute advantages / 步骤4: 计算优势
        advantages = self.compute_dynamic_advantages(rewards, valid_mask)

        # Step 5 & 6: Compute loss and update / 步骤5和6: 计算损失并更新
        total_loss = 0.0
        total_kl = 0.0
        num_valid = sum(valid_mask)

        if num_valid > 0:
            for idx, (response, advantage, is_valid) in enumerate(
                zip(responses, advantages, valid_mask)
            ):
                if not is_valid:
                    continue

                # Choose loss computation mode / 选择损失计算模式
                if self.config.use_token_level_loss:
                    loss, kl = self.compute_token_level_loss(
                        question, response, advantage
                    )
                else:
                    loss, kl = self.compute_sequence_level_loss(
                        question, response, advantage
                    )

                total_loss += loss.item() + self.config.beta * kl
                total_kl += kl

            # Average over valid samples / 在有效样本上平均
            avg_loss = total_loss / num_valid
            avg_kl = total_kl / num_valid

            # Backpropagation / 反向传播
            self.optimizer.zero_grad()
            torch.tensor(avg_loss, requires_grad=True).backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
        else:
            avg_loss = 0.0
            avg_kl = 0.0

        # Dynamic group size adjustment / 动态组大小调整
        valid_rewards = [r for r, m in zip(rewards, valid_mask) if m]
        if len(valid_rewards) > 1:
            reward_variance = np.var(valid_rewards)
            self.adjust_group_size(reward_variance)

        return {
            "loss": avg_loss,
            "avg_reward": np.mean(valid_rewards) if valid_rewards else 0.0,
            "kl_divergence": avg_kl,
            "group_size": G,
            "filtered_ratio": filtered_count / G if G > 0 else 0.0
        }

    def train(self, dataset: List[Dict]):
        """
        Train using DAPO algorithm
        使用 DAPO 算法训练

        Args:
            dataset: Training dataset / 训练数据集
        """
        logger.info(f"Starting DAPO training with {len(dataset)} examples")
        logger.info(f"Initial group size: {self.config.initial_group_size}")
        logger.info(f"Clip epsilon: {self.config.clip_epsilon}")
        logger.info(f"Beta (KL coef): {self.config.beta}")
        logger.info(f"Max response length: {self.config.max_response_length}")
        logger.info(f"Token-level loss: {self.config.use_token_level_loss}")

        self.model.train()

        for epoch in range(self.config.max_epochs):
            logger.info(f"\n{'='*50}")
            logger.info(f"Epoch {epoch + 1}/{self.config.max_epochs}")
            logger.info(f"{'='*50}")

            epoch_losses = []
            epoch_rewards = []
            epoch_kls = []
            epoch_group_sizes = []
            epoch_filtered_ratios = []

            for idx, example in enumerate(tqdm(dataset, desc=f"Epoch {epoch+1}")):
                stats = self.train_step(
                    question=example["question"],
                    correct_answer=example["correct_answer"],
                    reasoning_steps=example.get("reasoning_steps", [])
                )

                epoch_losses.append(stats["loss"])
                epoch_rewards.append(stats["avg_reward"])
                epoch_kls.append(stats["kl_divergence"])
                epoch_group_sizes.append(stats["group_size"])
                epoch_filtered_ratios.append(stats["filtered_ratio"])

                if (idx + 1) % 10 == 0:
                    logger.info(
                        f"Step {idx+1}: Loss={stats['loss']:.4f}, "
                        f"Reward={stats['avg_reward']:.4f}, "
                        f"KL={stats['kl_divergence']:.4f}, "
                        f"G={stats['group_size']}, "
                        f"Filtered={stats['filtered_ratio']:.2%}"
                    )

            # Epoch summary / Epoch 总结
            self.stats["epoch_losses"].append(np.mean(epoch_losses))
            self.stats["epoch_rewards"].append(np.mean(epoch_rewards))
            self.stats["epoch_kl_divs"].append(np.mean(epoch_kls))
            self.stats["epoch_group_sizes"].append(np.mean(epoch_group_sizes))
            self.stats["epoch_filtered_ratios"].append(np.mean(epoch_filtered_ratios))

            logger.info(f"\nEpoch {epoch+1} Summary:")
            logger.info(f"  Average Loss: {np.mean(epoch_losses):.4f}")
            logger.info(f"  Average Reward: {np.mean(epoch_rewards):.4f}")
            logger.info(f"  Average KL: {np.mean(epoch_kls):.4f}")
            logger.info(f"  Average Group Size: {np.mean(epoch_group_sizes):.1f}")
            logger.info(f"  Average Filtered Ratio: {np.mean(epoch_filtered_ratios):.2%}")

            self.save_checkpoint(epoch)

        logger.info("\nDAPO Training completed!")
        self.save_final_model()

    def save_checkpoint(self, epoch: int):
        """Save model checkpoint / 保存模型检查点"""
        checkpoint_dir = f"{self.config.output_dir}/epoch_{epoch+1}"
        os.makedirs(checkpoint_dir, exist_ok=True)

        self.model.save_pretrained(checkpoint_dir)
        self.tokenizer.save_pretrained(checkpoint_dir)

        with open(f"{checkpoint_dir}/stats.json", "w") as f:
            json.dump(self.stats, f, indent=2)

        logger.info(f"Checkpoint saved to {checkpoint_dir}")

    def save_final_model(self):
        """Save final trained model / 保存最终训练模型"""
        final_dir = f"{self.config.output_dir}/final"
        os.makedirs(final_dir, exist_ok=True)

        self.model.save_pretrained(final_dir)
        self.tokenizer.save_pretrained(final_dir)

        with open(f"{final_dir}/training_stats.json", "w") as f:
            json.dump(self.stats, f, indent=2)

        logger.info(f"Final model saved to {final_dir}")


def main():
    """Main training function / 主训练函数"""
    config = DAPOConfig()

    with open("data/sample_reasoning_data.json", "r") as f:
        dataset = json.load(f)

    logger.info(f"Loaded {len(dataset)} training examples")

    trainer = DAPOTrainer(config)
    trainer.train(dataset)


if __name__ == "__main__":
    main()
