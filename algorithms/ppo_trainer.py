"""
PPO (Proximal Policy Optimization) Training Implementation

This implementation provides a comparison baseline for GRPO.

Author: Aitachi
Contact: 44158892@qq.com
Date: 2025

Mathematical Formulation:
=======================

PPO optimizes the following objective function:

L^CLIP(θ) = E_t[min(r_t(θ) * Â_t, clip(r_t(θ), 1-ε, 1+ε) * Â_t)]

Where:
- r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)  # Probability ratio
- Â_t = A_t = R_t - V(s_t)                   # Advantage estimate
- ε: Clipping parameter (typically 0.2)
- V(s_t): Value function (critic network)

The complete PPO loss includes:
L_PPO(θ) = L^CLIP(θ) - c_1 * L^VF(θ) + c_2 * S[π_θ](s_t)

Where:
- L^VF(θ) = (V_θ(s_t) - V^target_t)^2  # Value function loss
- S[π_θ](s_t): Entropy bonus for exploration
- c_1, c_2: Loss coefficients

Key Differences from GRPO:
=========================
1. PPO requires a separate value network (critic) to estimate V(s_t)
2. GRPO uses group-based advantage normalization, PPO uses GAE or TD
3. PPO has entropy bonus term, GRPO uses KL divergence penalty
4. GRPO is more sample-efficient (doesn't need value network training)
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Tuple
import numpy as np
from tqdm import tqdm
import logging
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PPOConfig:
    """Configuration for PPO training"""

    # Model parameters
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"

    # PPO hyperparameters
    clip_epsilon: float = 0.2  # ε - PPO clipping parameter
    value_coef: float = 0.5  # c_1 - value loss coefficient
    entropy_coef: float = 0.01  # c_2 - entropy coefficient
    gamma: float = 0.99  # Discount factor
    gae_lambda: float = 0.95  # GAE parameter

    # Training parameters
    learning_rate_policy: float = 1e-5
    learning_rate_value: float = 3e-5
    max_epochs: int = 3
    ppo_epochs: int = 4  # Number of PPO update epochs per batch
    batch_size: int = 4
    max_length: int = 512
    temperature: float = 0.7

    # Reward parameters
    accuracy_reward: float = 10.0
    partial_reward: float = 2.0

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Paths
    output_dir: str = "./checkpoints/ppo_model"


class ValueNetwork(nn.Module):
    """
    Value Network (Critic) for PPO.

    This network estimates V(s), the expected return from state s.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Compute state value.

        Args:
            hidden_states: Hidden states from language model

        Returns:
            Value estimate V(s)
        """
        return self.value_head(hidden_states).squeeze(-1)


class PPOTrainer:
    """
    PPO (Proximal Policy Optimization) Trainer

    This trainer implements standard PPO with a separate value network.
    """

    def __init__(self, config: PPOConfig):
        self.config = config
        self.device = torch.device(config.device)

        # Load policy network (actor)
        logger.info(f"Loading policy model: {config.model_name}")
        self.policy_model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.float16 if "cuda" in config.device else torch.float32,
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Create value network (critic)
        logger.info("Creating value network")
        hidden_size = self.policy_model.config.hidden_size
        self.value_model = ValueNetwork(hidden_size).to(self.device)

        # Optimizers
        self.policy_optimizer = torch.optim.AdamW(
            self.policy_model.parameters(),
            lr=config.learning_rate_policy
        )
        self.value_optimizer = torch.optim.AdamW(
            self.value_model.parameters(),
            lr=config.learning_rate_value
        )

        # Training statistics
        self.stats = {
            "epoch_policy_losses": [],
            "epoch_value_losses": [],
            "epoch_rewards": [],
            "epoch_entropies": []
        }

    def generate_response(self, prompt: str) -> Tuple[str, torch.Tensor, torch.Tensor]:
        """
        Generate a single response and return log probability and value.

        Args:
            prompt: Input question

        Returns:
            Tuple of (response, log_prob, value_estimate)
        """
        formatted_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            max_length=self.config.max_length,
            truncation=True
        ).to(self.device)

        # Generate response
        with torch.no_grad():
            outputs = self.policy_model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=self.config.temperature,
                do_sample=True,
                output_hidden_states=True,
                return_dict_in_generate=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

        # Get generated text
        response_ids = outputs.sequences[0]
        response = self.tokenizer.decode(response_ids, skip_special_tokens=False)

        # Extract assistant's response
        if "<|im_start|>assistant\n" in response:
            response = response.split("<|im_start|>assistant\n")[-1]
            if "<|im_end|>" in response:
                response = response.split("<|im_end|>")[0]

        # Compute log probability for the generated sequence
        full_text = formatted_prompt + response.strip()
        full_inputs = self.tokenizer(
            full_text,
            return_tensors="pt",
            max_length=self.config.max_length,
            truncation=True
        ).to(self.device)

        with torch.no_grad():
            policy_outputs = self.policy_model(**full_inputs, output_hidden_states=True)
            log_prob = -policy_outputs.loss

            # Get value estimate from last hidden state
            last_hidden_state = policy_outputs.hidden_states[-1][:, -1, :]
            value = self.value_model(last_hidden_state)

        return response.strip(), log_prob, value

    def compute_gae(
        self,
        rewards: List[float],
        values: List[torch.Tensor],
        dones: List[bool]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Generalized Advantage Estimation (GAE).

        Formula:
        δ_t = r_t + γ * V(s_{t+1}) - V(s_t)
        A_t = δ_t + (γλ)δ_{t+1} + (γλ)^2 δ_{t+2} + ...

        Args:
            rewards: List of rewards
            values: List of value estimates
            dones: List of done flags

        Returns:
            Tuple of (advantages, returns)
        """
        advantages = []
        returns = []
        gae = 0
        next_value = 0

        # Reverse iteration for GAE calculation
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[t]
                next_value = 0
            else:
                next_non_terminal = 1.0 - dones[t]
                next_value = values[t + 1]

            delta = rewards[t] + self.config.gamma * next_value * next_non_terminal - values[t]
            gae = delta + self.config.gamma * self.config.gae_lambda * next_non_terminal * gae

            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])

        advantages = torch.tensor(advantages, device=self.device, dtype=torch.float32)
        returns = torch.tensor(returns, device=self.device, dtype=torch.float32)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages, returns

    def ppo_update(
        self,
        prompts: List[str],
        responses: List[str],
        old_log_probs: List[torch.Tensor],
        old_values: List[torch.Tensor],
        advantages: torch.Tensor,
        returns: torch.Tensor
    ) -> Dict[str, float]:
        """
        Perform PPO update for multiple epochs.

        Args:
            prompts: List of input prompts
            responses: List of generated responses
            old_log_probs: Log probabilities from old policy
            old_values: Value estimates from old critic
            advantages: Computed advantages
            returns: Computed returns

        Returns:
            Dictionary of training statistics
        """
        policy_losses = []
        value_losses = []
        entropies = []

        for epoch in range(self.config.ppo_epochs):
            for idx in range(len(prompts)):
                # Prepare input
                formatted_prompt = f"<|im_start|>user\n{prompts[idx]}<|im_end|>\n<|im_start|>assistant\n"
                full_text = formatted_prompt + responses[idx]

                inputs = self.tokenizer(
                    full_text,
                    return_tensors="pt",
                    max_length=self.config.max_length,
                    truncation=True
                ).to(self.device)

                # Forward pass through policy
                policy_outputs = self.policy_model(**inputs, output_hidden_states=True)
                curr_log_prob = -policy_outputs.loss

                # Compute entropy (for exploration bonus)
                logits = policy_outputs.logits[:, :-1, :]  # Remove last token
                probs = F.softmax(logits, dim=-1)
                log_probs_dist = F.log_softmax(logits, dim=-1)
                entropy = -(probs * log_probs_dist).sum(dim=-1).mean()

                # Compute policy ratio
                ratio = torch.exp(curr_log_prob - old_log_probs[idx])

                # PPO clipped objective
                advantage = advantages[idx]
                surr1 = ratio * advantage
                surr2 = torch.clamp(
                    ratio,
                    1.0 - self.config.clip_epsilon,
                    1.0 + self.config.clip_epsilon
                ) * advantage

                policy_loss = -torch.min(surr1, surr2)

                # Forward pass through value network
                last_hidden_state = policy_outputs.hidden_states[-1][:, -1, :]
                curr_value = self.value_model(last_hidden_state)

                # Value loss (MSE)
                value_loss = F.mse_loss(curr_value, returns[idx])

                # Total loss
                total_loss = (
                    policy_loss
                    + self.config.value_coef * value_loss
                    - self.config.entropy_coef * entropy
                )

                # Backward and optimize
                self.policy_optimizer.zero_grad()
                self.value_optimizer.zero_grad()
                total_loss.backward()
                self.policy_optimizer.step()
                self.value_optimizer.step()

                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropies.append(entropy.item())

        return {
            "policy_loss": np.mean(policy_losses),
            "value_loss": np.mean(value_losses),
            "entropy": np.mean(entropies)
        }

    def train_step(
        self,
        question: str,
        correct_answer: str
    ) -> Dict[str, float]:
        """
        Perform one PPO training step.

        Args:
            question: Input question
            correct_answer: Ground truth answer

        Returns:
            Training statistics
        """
        # Generate response and get log prob + value
        response, old_log_prob, old_value = self.generate_response(question)

        # Compute reward
        from algorithms.grpo_trainer import RewardModel
        reward_model = RewardModel(self.config)
        reward = reward_model.compute_reward(question, response, correct_answer)

        # For single-step episodes, GAE simplifies
        advantage = torch.tensor([reward], device=self.device) - old_value
        returns = torch.tensor([reward], device=self.device)

        # Normalize advantage
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        # PPO update
        stats = self.ppo_update(
            prompts=[question],
            responses=[response],
            old_log_probs=[old_log_prob],
            old_values=[old_value],
            advantages=advantage,
            returns=returns
        )

        stats["reward"] = reward
        return stats

    def train(self, dataset: List[Dict]):
        """
        Train using PPO algorithm.

        Args:
            dataset: Training dataset
        """
        logger.info(f"Starting PPO training with {len(dataset)} examples")
        logger.info(f"Clip epsilon: {self.config.clip_epsilon}")
        logger.info(f"Value coef: {self.config.value_coef}")
        logger.info(f"Entropy coef: {self.config.entropy_coef}")

        self.policy_model.train()
        self.value_model.train()

        for epoch in range(self.config.max_epochs):
            logger.info(f"\n{'='*50}")
            logger.info(f"Epoch {epoch + 1}/{self.config.max_epochs}")
            logger.info(f"{'='*50}")

            epoch_policy_losses = []
            epoch_value_losses = []
            epoch_rewards = []
            epoch_entropies = []

            for idx, example in enumerate(tqdm(dataset, desc=f"Epoch {epoch+1}")):
                stats = self.train_step(
                    question=example["question"],
                    correct_answer=example["correct_answer"]
                )

                epoch_policy_losses.append(stats["policy_loss"])
                epoch_value_losses.append(stats["value_loss"])
                epoch_rewards.append(stats["reward"])
                epoch_entropies.append(stats["entropy"])

                if (idx + 1) % 10 == 0:
                    logger.info(
                        f"Step {idx+1}: Policy Loss={stats['policy_loss']:.4f}, "
                        f"Value Loss={stats['value_loss']:.4f}, "
                        f"Reward={stats['reward']:.4f}"
                    )

            # Epoch summary
            self.stats["epoch_policy_losses"].append(np.mean(epoch_policy_losses))
            self.stats["epoch_value_losses"].append(np.mean(epoch_value_losses))
            self.stats["epoch_rewards"].append(np.mean(epoch_rewards))
            self.stats["epoch_entropies"].append(np.mean(epoch_entropies))

            logger.info(f"\nEpoch {epoch+1} Summary:")
            logger.info(f"  Avg Policy Loss: {np.mean(epoch_policy_losses):.4f}")
            logger.info(f"  Avg Value Loss: {np.mean(epoch_value_losses):.4f}")
            logger.info(f"  Avg Reward: {np.mean(epoch_rewards):.4f}")
            logger.info(f"  Avg Entropy: {np.mean(epoch_entropies):.4f}")

            self.save_checkpoint(epoch)

        logger.info("\nPPO Training completed!")
        self.save_final_model()

    def save_checkpoint(self, epoch: int):
        """Save model checkpoint"""
        checkpoint_dir = f"{self.config.output_dir}/epoch_{epoch+1}"
        os.makedirs(checkpoint_dir, exist_ok=True)

        self.policy_model.save_pretrained(checkpoint_dir)
        self.value_model.save_pretrained(f"{checkpoint_dir}/value_model")
        self.tokenizer.save_pretrained(checkpoint_dir)

        with open(f"{checkpoint_dir}/stats.json", "w") as f:
            json.dump(self.stats, f, indent=2)

        logger.info(f"Checkpoint saved to {checkpoint_dir}")

    def save_final_model(self):
        """Save final model"""
        final_dir = f"{self.config.output_dir}/final"
        os.makedirs(final_dir, exist_ok=True)

        self.policy_model.save_pretrained(final_dir)
        self.value_model.save_pretrained(f"{final_dir}/value_model")
        self.tokenizer.save_pretrained(final_dir)

        with open(f"{final_dir}/training_stats.json", "w") as f:
            json.dump(self.stats, f, indent=2)

        logger.info(f"Final model saved to {final_dir}")


def main():
    """Main training function"""
    config = PPOConfig()

    with open("data/sample_reasoning_data.json", "r") as f:
        dataset = json.load(f)

    logger.info(f"Loaded {len(dataset)} training examples")

    trainer = PPOTrainer(config)
    trainer.train(dataset)


if __name__ == "__main__":
    main()
