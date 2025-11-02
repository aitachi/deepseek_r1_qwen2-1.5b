"""
GRPO (Group Relative Policy Optimization) Training Implementation

This implementation is based on the DeepSeek-R1 paper:
"DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning"

Author: Aitachi
Contact: 44158892@qq.com
Date: 2025

Mathematical Formulation:
=======================

GRPO optimizes the following objective function:

J_GRPO(θ) = E[q ~ P(Q), {o_i}^G_{i=1} ~ π_{θ_old}(O|q)]
            [
              1/G ∑^G_{i=1} min(
                π_θ(o_i|q) / π_{θ_old}(o_i|q) * A_i,
                clip(π_θ(o_i|q) / π_{θ_old}(o_i|q), 1-ε, 1+ε) * A_i
              )
              - β * D_KL(π_θ || π_ref)
            ]

Where:
- θ: Current policy parameters
- θ_old: Old policy parameters
- q: Input question/prompt
- o_i: i-th output sample
- G: Group size (number of samples per question)
- ε: Clipping parameter (typically 0.2)
- β: KL divergence coefficient
- A_i: Advantage function for sample i

Advantage Calculation:
A_i = (r_i - mean({r_1, r_2, ..., r_G})) / std({r_1, r_2, ..., r_G})

Where r_i is the reward for output o_i.

KL Divergence:
D_KL(π_θ || π_ref) = π_ref(o_i|q)/π_θ(o_i|q) - log(π_ref(o_i|q)/π_θ(o_i|q)) - 1
"""

import os
import json
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from transformers import AutoModel ForCausalLM, AutoTokenizer
from typing import List, Dict, Tuple, Optional
import numpy as np
from tqdm import tqdm
import logging
from dataclasses import dataclass

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class GRPOConfig:
    """Configuration for GRPO training"""

    # Model parameters
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"  # Using smaller model for faster training

    # GRPO hyper parameters
    group_size: int = 16  # G in the formula - number of samples per question
    clip_epsilon: float = 0.2  # ε in the formula - PPO clipping parameter
    beta: float = 0.01  # β in the formula - KL divergence coefficient

    # Training parameters
    learning_rate: float = 1e-5
    max_epochs: int = 3
    batch_size: int = 4
    max_length: int = 512
    temperature: float = 0.7

    # Reward parameters
    accuracy_reward: float = 10.0  # Reward for correct answer
    partial_reward: float = 2.0  # Reward for partially correct

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Paths
    output_dir: str = "./checkpoints/grpo_model"
    log_dir: str = "./logs/grpo"


class RewardModel:
    """
    Reward model for evaluating generated responses.

    In GRPO, we use rule-based rewards:
    - Accuracy rewards: Check if the response is correct
    - Format rewards: Ensure proper formatting with <think> tags
    """

    def __init__(self, config: GRPOConfig):
        self.config = config

    def compute_reward(
        self,
        question: str,
        response: str,
        correct_answer: str
    ) -> float:
        """
        Compute reward for a generated response.

        Args:
            question: Input question
            response: Model generated response
            correct_answer: Ground truth answer

        Returns:
            Reward score (float)
        """
        reward = 0.0

        # Extract answer from response (typically after "answer:" or in <answer> tags)
        predicted_answer = self._extract_answer(response)

        # Accuracy reward
        if predicted_answer.lower().strip() == correct_answer.lower().strip():
            reward += self.config.accuracy_reward
        elif self._is_partially_correct(predicted_answer, correct_answer):
            reward += self.config.partial_reward

        # Format reward - encourage use of thinking tags
        if "<think>" in response and "</think>" in response:
            reward += 0.5

        return reward

    def _extract_answer(self, response: str) -> str:
        """Extract the final answer from model response"""
        # Try to extract from <answer> tags
        if "<answer>" in response and "</answer>" in response:
            start = response.find("<answer>") + len("<answer>")
            end = response.find("</answer>")
            return response[start:end].strip()

        # Try to find after "answer:" keyword
        if "answer:" in response.lower():
            parts = response.lower().split("answer:")
            if len(parts) > 1:
                return parts[-1].strip().split()[0] if parts[-1].strip() else ""

        # Return last few words as fallback
        words = response.strip().split()
        return " ".join(words[-5:]) if words else ""

    def _is_partially_correct(self, predicted: str, correct: str) -> bool:
        """Check if answer is partially correct"""
        # Simple heuristic - check if key numbers/terms match
        pred_numbers = set(''.join(c for c in predicted if c.isdigit() or c == '.'))
        correct_numbers = set(''.join(c for c in correct if c.isdigit() or c == '.'))

        if pred_numbers and correct_numbers:
            # Check overlap
            overlap = len(pred_numbers & correct_numbers) / len(correct_numbers)
            return overlap > 0.5

        return False


class GRPOTrainer:
    """
    GRPO (Group Relative Policy Optimization) Trainer

    This trainer implements the GRPO algorithm as described in DeepSeek-R1 paper.
    """

    def __init__(self, config: GRPOConfig):
        self.config = config
        self.device = torch.device(config.device)

        # Load model and tokenizer
        logger.info(f"Loading model: {config.model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.float16 if "cuda" in config.device else torch.float32,
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Reference model (frozen copy for KL divergence)
        logger.info("Creating reference model for KL divergence")
        self.ref_model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.float16 if "cuda" in config.device else torch.float32,
            device_map="auto"
        )
        self.ref_model.eval()  # Freeze reference model
        for param in self.ref_model.parameters():
            param.requires_grad = False

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate
        )

        # Reward model
        self.reward_model = RewardModel(config)

        # Training statistics
        self.stats = {
            "epoch_losses": [],
            "epoch_rewards": [],
            "epoch_kl_divs": []
        }

    def generate_responses(
        self,
        prompt: str,
        num_samples: int
    ) -> List[str]:
        """
        Generate multiple response samples for a given prompt.

        Args:
            prompt: Input question/prompt
            num_samples: Number of samples to generate (G in GRPO formula)

        Returns:
            List of generated responses
        """
        responses = []

        # Format prompt with instruction
        formatted_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

        # Tokenize
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            max_length=self.config.max_length,
            truncation=True
        ).to(self.device)

        # Generate multiple samples
        for _ in range(num_samples):
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=self.config.temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    num_return_sequences=1
                )

            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=False)

            # Extract assistant's response
            if "<|im_start|>assistant\n" in response:
                response = response.split("<|im_start|>assistant\n")[-1]
                if "<|im_end|>" in response:
                    response = response.split("<|im_end|>")[0]

            responses.append(response.strip())

        return responses

    def compute_advantages(self, rewards: torch.Tensor) -> torch.Tensor:
        """
        Compute group-normalized advantages.

        Formula: A_i = (r_i - mean(rewards)) / (std(rewards) + eps)

        Args:
            rewards: Tensor of shape (group_size,) containing rewards

        Returns:
            Advantages tensor of same shape
        """
        advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        return advantages

    def compute_kl_divergence(
        self,
        log_probs_current: torch.Tensor,
        log_probs_ref: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute KL divergence between current and reference policy.

        Formula: D_KL = exp(log_prob_ref - log_prob_current)
                        - (log_prob_ref - log_prob_current) - 1

        Args:
            log_probs_current: Log probabilities from current policy
            log_probs_ref: Log probabilities from reference policy

        Returns:
            KL divergence value
        """
        ratio = torch.exp(log_probs_ref - log_probs_current)
        kl = ratio - (log_probs_ref - log_probs_current) - 1
        return kl.mean()

    def train_step(
        self,
        question: str,
        correct_answer: str,
        reasoning_steps: List[str]
    ) -> Dict[str, float]:
        """
        Perform one GRPO training step on a single question.

        Args:
            question: Input question
            correct_answer: Ground truth answer
            reasoning_steps: List of reasoning steps (for reference)

        Returns:
            Dictionary containing step statistics
        """
        # Step 1: Generate group of responses using current policy
        responses = self.generate_responses(question, self.config.group_size)

        # Step 2: Compute rewards for each response
        rewards = []
        for response in responses:
            reward = self.reward_model.compute_reward(
                question, response, correct_answer
            )
            rewards.append(reward)

        rewards = torch.tensor(rewards, device=self.device, dtype=torch.float32)

        # Step 3: Compute advantages (group-normalized)
        advantages = self.compute_advantages(rewards)

        # Step 4: Compute policy ratios and losses
        total_loss = 0.0
        total_kl = 0.0

        formatted_prompt = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"

        for idx, response in enumerate(responses):
            # Tokenize prompt + response
            full_text = formatted_prompt + response
            inputs = self.tokenizer(
                full_text,
                return_tensors="pt",
                max_length=self.config.max_length,
                truncation=True
            ).to(self.device)

            # Get log probabilities from current policy
            outputs_current = self.model(**inputs, labels=inputs["input_ids"])
            log_prob_current = -outputs_current.loss

            # Get log probabilities from reference policy (frozen)
            with torch.no_grad():
                outputs_ref = self.ref_model(**inputs, labels=inputs["input_ids"])
                log_prob_ref = -outputs_ref.loss

            # Compute ratio: π_θ(o|q) / π_θ_old(o|q)
            ratio = torch.exp(log_prob_current - log_prob_ref)

            # GRPO clipped objective
            advantage = advantages[idx]
            surr1 = ratio * advantage
            surr2 = torch.clamp(
                ratio,
                1.0 - self.config.clip_epsilon,
                1.0 + self.config.clip_epsilon
            ) * advantage

            # Policy loss (negative because we want to maximize)
            policy_loss = -torch.min(surr1, surr2)

            # KL divergence
            kl_div = self.compute_kl_divergence(log_prob_current, log_prob_ref)

            # Total loss with KL penalty
            loss = policy_loss + self.config.beta * kl_div

            total_loss += loss.item()
            total_kl += kl_div.item()

        # Average loss over group
        avg_loss = total_loss / self.config.group_size
        avg_kl = total_kl / self.config.group_size

        # Backpropagation
        self.optimizer.zero_grad()
        torch.tensor(avg_loss, requires_grad=True).backward()
        self.optimizer.step()

        return {
            "loss": avg_loss,
            "avg_reward": rewards.mean().item(),
            "max_reward": rewards.max().item(),
            "kl_divergence": avg_kl
        }

    def train(self, dataset: List[Dict]):
        """
        Train the model using GRPO algorithm.

        Args:
            dataset: List of training examples, each containing:
                     - question: str
                     - correct_answer: str
                     - reasoning_steps: List[str]
        """
        logger.info(f"Starting GRPO training with {len(dataset)} examples")
        logger.info(f"Group size: {self.config.group_size}")
        logger.info(f"Clip epsilon: {self.config.clip_epsilon}")
        logger.info(f"Beta (KL coef): {self.config.beta}")

        self.model.train()

        for epoch in range(self.config.max_epochs):
            logger.info(f"\n{'='*50}")
            logger.info(f"Epoch {epoch + 1}/{self.config.max_epochs}")
            logger.info(f"{'='*50}")

            epoch_losses = []
            epoch_rewards = []
            epoch_kls = []

            # Training loop
            for idx, example in enumerate(tqdm(dataset, desc=f"Epoch {epoch+1}")):
                stats = self.train_step(
                    question=example["question"],
                    correct_answer=example["correct_answer"],
                    reasoning_steps=example.get("reasoning_steps", [])
                )

                epoch_losses.append(stats["loss"])
                epoch_rewards.append(stats["avg_reward"])
                epoch_kls.append(stats["kl_divergence"])

                # Log every 10 steps
                if (idx + 1) % 10 == 0:
                    logger.info(
                        f"Step {idx+1}: Loss={stats['loss']:.4f}, "
                        f"Reward={stats['avg_reward']:.4f}, "
                        f"KL={stats['kl_divergence']:.4f}"
                    )

            # Epoch statistics
            avg_loss = np.mean(epoch_losses)
            avg_reward = np.mean(epoch_rewards)
            avg_kl = np.mean(epoch_kls)

            self.stats["epoch_losses"].append(avg_loss)
            self.stats["epoch_rewards"].append(avg_reward)
            self.stats["epoch_kl_divs"].append(avg_kl)

            logger.info(f"\nEpoch {epoch+1} Summary:")
            logger.info(f"  Average Loss: {avg_loss:.4f}")
            logger.info(f"  Average Reward: {avg_reward:.4f}")
            logger.info(f"  Average KL: {avg_kl:.4f}")

            # Save checkpoint
            self.save_checkpoint(epoch)

        logger.info("\nTraining completed!")
        self.save_final_model()

    def save_checkpoint(self, epoch: int):
        """Save model checkpoint"""
        checkpoint_dir = f"{self.config.output_dir}/epoch_{epoch+1}"
        os.makedirs(checkpoint_dir, exist_ok=True)

        self.model.save_pretrained(checkpoint_dir)
        self.tokenizer.save_pretrained(checkpoint_dir)

        # Save training stats
        with open(f"{checkpoint_dir}/stats.json", "w") as f:
            json.dump(self.stats, f, indent=2)

        logger.info(f"Checkpoint saved to {checkpoint_dir}")

    def save_final_model(self):
        """Save final trained model"""
        final_dir = f"{self.config.output_dir}/final"
        os.makedirs(final_dir, exist_ok=True)

        self.model.save_pretrained(final_dir)
        self.tokenizer.save_pretrained(final_dir)

        # Save final stats
        with open(f"{final_dir}/training_stats.json", "w") as f:
            json.dump(self.stats, f, indent=2)

        logger.info(f"Final model saved to {final_dir}")


def main():
    """Main training function"""
    # Load configuration
    config = GRPOConfig()

    # Load training data
    with open("data/sample_reasoning_data.json", "r") as f:
        dataset = json.load(f)

    logger.info(f"Loaded {len(dataset)} training examples")

    # Initialize trainer
    trainer = GRPOTrainer(config)

    # Start training
    trainer.train(dataset)


if __name__ == "__main__":
    main()
