"""
DAPO (Decoupled Clip and Dynamic sAmpling Policy Optimization) Training Implementation

This implementation is based on the DAPO paper:
"DAPO: An Open-Source LLM Reinforcement Learning System at Scale"

Author: Aitachi
Contact: 44158892@qq.com
Date: 2025

Mathematical Formulation:
=======================

DAPO optimizes the following objective function:

J_DAPO(θ) = E[(q,a)~D, {o_i}^G_{i=1}~π_{θ_old}(·|q)]
            [
              1/(∑^G_{i=1}|o_i|) ∑^G_{i=1} ∑^|o_i|_{t=1} min(
                r_{i,t}(θ) * Â_{i,t},
                clip(r_{i,t}(θ), 1-ε_low, 1+ε_high) * Â_{i,t}
              )
            ]

subject to: 0 < |{o_i | is_equivalent(a, o_i)}| < G

Where:
- θ: Current policy parameters
- θ_old: Old policy parameters
- q: Input question/prompt
- a: Correct answer
- o_i: i-th output sample
- G: Group size (number of samples per question)
- ε_low: Lower clipping parameter (typically 0.2)
- ε_high: Higher clipping parameter (typically 0.28)
- r_{i,t}(θ): Importance sampling ratio at time t
- Â_{i,t}: Advantage function at time t

Key Techniques:
===============

1. Clip-Higher: Decoupled clipping ranges [1-ε_low, 1+ε_high]
   - Promotes diversity and avoids entropy collapse
   - Allows low-probability tokens to increase more

2. Dynamic Sampling: Filter samples with accuracy = 0 or 1
   - Ensures all samples have effective gradients
   - Maintains consistent batch size

3. Token-Level Policy Gradient Loss: Average over tokens not samples
   - Better handles long CoT scenarios
   - Prevents low-quality long samples from dominating

4. Overlong Reward Shaping: Soft punishment for overlong samples
   - R_length(y) = 0 if |y| ≤ L_max - L_cache
   - R_length(y) = (L_max - L_cache - |y|) / L_cache if in punishment interval
   - R_length(y) = -1 if |y| > L_max

Advantage Calculation:
Â_{i,t} = (R_i - mean({R_1, R_2, ..., R_G})) / (std({R_1, R_2, ..., R_G}) + eps)

Where R_i is the reward for output o_i.
"""

import os
import json
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Tuple, Optional
import numpy as np
from tqdm import tqdm
import logging
from dataclasses import dataclass

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DAPOConfig:
    """Configuration for DAPO training"""

    # Model parameters
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"

    # DAPO hyperparameters
    group_size: int = 16  # G in the formula - number of samples per question
    clip_epsilon_low: float = 0.2  # ε_low - lower clipping parameter
    clip_epsilon_high: float = 0.28  # ε_high - higher clipping parameter (Clip-Higher)

    # Dynamic sampling parameters
    enable_dynamic_sampling: bool = True  # Filter samples with acc=0 or acc=1
    max_sampling_tries: int = 5  # Maximum attempts to fill batch

    # Training parameters
    learning_rate: float = 1e-6  # Lower LR as per paper
    max_epochs: int = 3
    batch_size: int = 512  # Prompt batch size from paper
    gradient_updates_per_rollout: int = 16  # Mini-batch updates
    max_length: int = 16384  # Maximum expected length
    soft_punish_cache: int = 4096  # Additional tokens for soft punishment
    max_generation_tokens: int = 20480  # Max tokens for generation (16384 + 4096)
    temperature: float = 1.0  # Temperature from paper
    top_p: float = 0.7  # Top-p sampling from paper

    # Reward parameters
    correct_reward: float = 1.0  # Reward for correct answer
    incorrect_reward: float = -1.0  # Reward for incorrect answer

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Paths
    output_dir: str = "./checkpoints/dapo_model"
    log_dir: str = "./logs/dapo"


class RewardModel:
    """
    Rule-based reward model for DAPO.

    Uses simple accuracy-based rewards:
    R(ŷ, y) = 1 if is_equivalent(ŷ, y)
    R(ŷ, y) = -1 otherwise

    Plus length-based reward shaping for overlong samples.
    """

    def __init__(self, config: DAPOConfig):
        self.config = config

    def compute_reward(
        self,
        question: str,
        response: str,
        correct_answer: str
    ) -> Tuple[float, bool]:
        """
        Compute reward for a generated response.

        Args:
            question: Input question
            response: Model generated response
            correct_answer: Ground truth answer

        Returns:
            Tuple of (reward, is_correct)
        """
        # Extract answer from response
        predicted_answer = self._extract_answer(response)

        # Check correctness
        is_correct = self._is_equivalent(predicted_answer, correct_answer)

        # Base reward
        base_reward = self.config.correct_reward if is_correct else self.config.incorrect_reward

        # Add length penalty
        length_penalty = self._compute_length_penalty(response)

        total_reward = base_reward + length_penalty

        return total_reward, is_correct

    def _extract_answer(self, response: str) -> str:
        """Extract the final answer from model response"""
        # Try to extract from answer tags or patterns
        if "<answer>" in response and "</answer>" in response:
            start = response.find("<answer>") + len("<answer>")
            end = response.find("</answer>")
            return response[start:end].strip()

        # Try to find after "answer:" keyword
        if "answer:" in response.lower():
            parts = response.lower().split("answer:")
            if len(parts) > 1:
                # Get the part after "answer:" and extract first number/word
                answer_part = parts[-1].strip()
                # Try to extract integer answer (as per paper's dataset transformation)
                import re
                numbers = re.findall(r'-?\d+', answer_part)
                if numbers:
                    return numbers[0]
                # Otherwise return first few words
                words = answer_part.split()
                return words[0] if words else ""

        # Fallback: extract last number from response
        import re
        numbers = re.findall(r'-?\d+', response)
        return numbers[-1] if numbers else ""

    def _is_equivalent(self, predicted: str, correct: str) -> bool:
        """Check if predicted answer is equivalent to correct answer"""
        # Normalize and compare
        pred_normalized = predicted.lower().strip()
        correct_normalized = correct.lower().strip()

        # Direct comparison
        if pred_normalized == correct_normalized:
            return True

        # Try numeric comparison for integer answers
        try:
            pred_num = float(pred_normalized)
            correct_num = float(correct_normalized)
            return abs(pred_num - correct_num) < 1e-6
        except:
            pass

        return False

    def _compute_length_penalty(self, response: str) -> float:
        """
        Compute length-based reward shaping (Overlong Reward Shaping).

        R_length(y) = 0 if |y| ≤ L_max - L_cache
        R_length(y) = (L_max - L_cache - |y|) / L_cache if in punishment interval
        R_length(y) = -1 if |y| > L_max
        """
        response_length = len(response.split())  # Approximate token count

        L_max = self.config.max_length
        L_cache = self.config.soft_punish_cache

        if response_length <= L_max - L_cache:
            return 0.0
        elif response_length <= L_max:
            # Soft punishment interval
            return (L_max - L_cache - response_length) / L_cache
        else:
            # Hard punishment
            return -1.0


class DAPOTrainer:
    """
    DAPO (Decoupled Clip and Dynamic sAmpling Policy Optimization) Trainer

    Implements the DAPO algorithm with four key techniques:
    1. Clip-Higher: Decoupled clipping ranges
    2. Dynamic Sampling: Filter zero-gradient samples
    3. Token-Level Loss: Average over tokens not samples
    4. Overlong Reward Shaping: Length-aware penalties
    """

    def __init__(self, config: DAPOConfig):
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

        # Store old policy parameters (for computing importance ratios)
        self.old_model = None

        # Optimizer - using AdamW with constant learning rate
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate
        )

        # Reward model
        self.reward_model = RewardModel(config)

        # Training statistics
        self.stats = {
            "step_losses": [],
            "step_rewards": [],
            "step_entropies": [],
            "step_mean_lengths": [],
            "step_accuracies": []
        }

        self.current_step = 0

    def update_old_policy(self):
        """Update old policy parameters (π_θ_old ← π_θ)"""
        # Deep copy current model to old model
        if self.old_model is None:
            self.old_model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                torch_dtype=torch.float16 if "cuda" in self.config.device else torch.float32,
                device_map="auto"
            )

        # Copy parameters
        self.old_model.load_state_dict(self.model.state_dict())
        self.old_model.eval()

        # Freeze old model
        for param in self.old_model.parameters():
            param.requires_grad = False

    def generate_responses(
        self,
        prompt: str,
        num_samples: int
    ) -> List[str]:
        """
        Generate multiple response samples for a given prompt using old policy.

        Args:
            prompt: Input question/prompt
            num_samples: Number of samples to generate (G in DAPO formula)

        Returns:
            List of generated responses
        """
        responses = []

        # Use old policy for generation
        model_to_use = self.old_model if self.old_model is not None else self.model
        model_to_use.eval()

        # Format prompt
        formatted_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

        # Tokenize
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            max_length=1024,  # Input prompt length
            truncation=True
        ).to(self.device)

        # Generate multiple samples
        for _ in range(num_samples):
            with torch.no_grad():
                outputs = model_to_use.generate(
                    **inputs,
                    max_new_tokens=min(2048, self.config.max_generation_tokens),  # Limit for efficiency
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
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

    def dynamic_sample_batch(
        self,
        question: str,
        correct_answer: str
    ) -> Tuple[List[str], List[float], List[bool]]:
        """
        Dynamic sampling: Keep sampling until we get samples with 0 < accuracy < 1.

        This ensures all samples have effective gradients (non-zero advantages).

        Returns:
            Tuple of (responses, rewards, correctness_list)
        """
        attempts = 0
        max_attempts = self.config.max_sampling_tries

        while attempts < max_attempts:
            # Generate group of responses
            responses = self.generate_responses(question, self.config.group_size)

            # Compute rewards
            rewards = []
            correctness = []
            for response in responses:
                reward, is_correct = self.reward_model.compute_reward(
                    question, response, correct_answer
                )
                rewards.append(reward)
                correctness.append(is_correct)

            # Check if we have mixed correctness (not all correct, not all incorrect)
            num_correct = sum(correctness)

            if self.config.enable_dynamic_sampling:
                # Dynamic sampling: reject if all same (gradient would be zero)
                if 0 < num_correct < len(correctness):
                    return responses, rewards, correctness
                attempts += 1
            else:
                # No dynamic sampling, accept any batch
                return responses, rewards, correctness

        # If we exhausted attempts, return the last batch anyway
        logger.warning(f"Dynamic sampling: Could not get mixed batch after {max_attempts} attempts")
        return responses, rewards, correctness

    def compute_advantages(self, rewards: torch.Tensor) -> torch.Tensor:
        """
        Compute group-normalized advantages.

        Formula: Â_i = (R_i - mean({R_i})) / (std({R_i}) + eps)

        Args:
            rewards: Tensor of shape (group_size,) containing rewards

        Returns:
            Advantages tensor of same shape
        """
        advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        return advantages

    def compute_token_level_loss(
        self,
        responses: List[str],
        advantages: torch.Tensor,
        question: str
    ) -> Tuple[float, Dict]:
        """
        Compute token-level policy gradient loss (key technique #3).

        Instead of averaging over samples first, we average over all tokens:
        Loss = (1 / ∑|o_i|) * ∑_i ∑_t loss(o_i,t)

        Returns:
            Tuple of (loss_value, statistics_dict)
        """
        total_loss = 0.0
        total_tokens = 0

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

            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]

            # Get number of tokens in this response
            response_tokens = input_ids.shape[1]

            # Get log probabilities from current policy
            outputs_current = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids
            )

            # Get log probabilities from old policy
            with torch.no_grad():
                outputs_old = self.old_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=input_ids
                )

            # Compute per-token log probabilities
            logits_current = outputs_current.logits[:, :-1, :]  # Exclude last token
            logits_old = outputs_old.logits[:, :-1, :]

            labels = input_ids[:, 1:]  # Shift labels

            # Get log probs for actual tokens
            log_probs_current = F.log_softmax(logits_current, dim=-1)
            log_probs_old = F.log_softmax(logits_old, dim=-1)

            # Gather log probs for actual tokens
            token_log_probs_current = log_probs_current.gather(
                -1, labels.unsqueeze(-1)
            ).squeeze(-1)

            token_log_probs_old = log_probs_old.gather(
                -1, labels.unsqueeze(-1)
            ).squeeze(-1)

            # Compute importance sampling ratios for each token
            # r_t = π_θ(o_t | q, o_<t) / π_θ_old(o_t | q, o_<t)
            ratios = torch.exp(token_log_probs_current - token_log_probs_old)

            # DAPO clipped objective with decoupled clip ranges
            advantage = advantages[idx]

            surr1 = ratios * advantage
            surr2 = torch.clamp(
                ratios,
                1.0 - self.config.clip_epsilon_low,
                1.0 + self.config.clip_epsilon_high  # Clip-Higher!
            ) * advantage

            # Token-level loss (negative for maximization)
            token_losses = -torch.min(surr1, surr2)

            # Sum over tokens (not average per sample)
            sample_loss = token_losses.sum()

            total_loss += sample_loss.item()
            total_tokens += response_tokens

        # Average over all tokens across all samples
        avg_loss = total_loss / max(total_tokens, 1)

        stats = {
            "total_tokens": total_tokens,
            "avg_tokens_per_sample": total_tokens / len(responses)
        }

        return avg_loss, stats

    def train_step(
        self,
        question: str,
        correct_answer: str
    ) -> Dict[str, float]:
        """
        Perform one DAPO training step on a single question.

        Algorithm 1 from paper:
        1. Update old policy π_θ_old ← π_θ
        2. Sample G outputs {o_i} ~ π_θ_old(·|q)
        3. Compute rewards {r_i}
        4. Filter out samples (Dynamic Sampling)
        5. Compute advantages Â_i,t
        6. Update policy by maximizing DAPO objective

        Returns:
            Dictionary containing step statistics
        """
        # Step 1: Update old policy
        self.update_old_policy()

        # Steps 2-4: Dynamic sampling
        responses, rewards, correctness = self.dynamic_sample_batch(
            question, correct_answer
        )

        if not responses:
            logger.warning("No valid responses generated, skipping step")
            return {"loss": 0.0, "avg_reward": 0.0, "accuracy": 0.0}

        rewards_tensor = torch.tensor(rewards, device=self.device, dtype=torch.float32)

        # Step 5: Compute advantages
        advantages = self.compute_advantages(rewards_tensor)

        # Step 6: Compute loss and update
        self.model.train()

        loss_value, loss_stats = self.compute_token_level_loss(
            responses, advantages, question
        )

        # Backpropagation
        self.optimizer.zero_grad()
        loss_tensor = torch.tensor(loss_value, device=self.device, requires_grad=True)
        loss_tensor.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        self.optimizer.step()

        # Compute statistics
        accuracy = sum(correctness) / len(correctness)
        avg_length = np.mean([len(r.split()) for r in responses])

        # Compute entropy (diversity metric)
        entropy = self._compute_entropy(responses)

        return {
            "loss": loss_value,
            "avg_reward": rewards_tensor.mean().item(),
            "max_reward": rewards_tensor.max().item(),
            "min_reward": rewards_tensor.min().item(),
            "accuracy": accuracy,
            "avg_length": avg_length,
            "entropy": entropy,
            "num_samples": len(responses),
            **loss_stats
        }

    def _compute_entropy(self, responses: List[str]) -> float:
        """Compute entropy of response distribution (diversity metric)"""
        # Simple heuristic: unique responses ratio
        unique_responses = len(set(responses))
        total_responses = len(responses)
        return unique_responses / total_responses if total_responses > 0 else 0.0

    def train(self, dataset: List[Dict]):
        """
        Train the model using DAPO algorithm.

        Args:
            dataset: List of training examples, each containing:
                     - question: str
                     - correct_answer: str
        """
        logger.info(f"Starting DAPO training with {len(dataset)} examples")
        logger.info(f"Group size: {self.config.group_size}")
        logger.info(f"Clip epsilon (low, high): ({self.config.clip_epsilon_low}, {self.config.clip_epsilon_high})")
        logger.info(f"Dynamic sampling: {self.config.enable_dynamic_sampling}")
        logger.info(f"Token-level loss: Enabled")
        logger.info(f"Overlong reward shaping: Enabled")

        for epoch in range(self.config.max_epochs):
            logger.info(f"\n{'='*60}")
            logger.info(f"Epoch {epoch + 1}/{self.config.max_epochs}")
            logger.info(f"{'='*60}")

            epoch_stats = {
                "losses": [],
                "rewards": [],
                "accuracies": [],
                "lengths": [],
                "entropies": []
            }

            # Training loop
            for idx, example in enumerate(tqdm(dataset, desc=f"Epoch {epoch+1}")):
                stats = self.train_step(
                    question=example["question"],
                    correct_answer=example["correct_answer"]
                )

                # Record statistics
                epoch_stats["losses"].append(stats["loss"])
                epoch_stats["rewards"].append(stats["avg_reward"])
                epoch_stats["accuracies"].append(stats["accuracy"])
                epoch_stats["lengths"].append(stats["avg_length"])
                epoch_stats["entropies"].append(stats["entropy"])

                self.stats["step_losses"].append(stats["loss"])
                self.stats["step_rewards"].append(stats["avg_reward"])
                self.stats["step_accuracies"].append(stats["accuracy"])
                self.stats["step_mean_lengths"].append(stats["avg_length"])
                self.stats["step_entropies"].append(stats["entropy"])

                self.current_step += 1

                # Log every 10 steps
                if (idx + 1) % 10 == 0:
                    logger.info(
                        f"Step {self.current_step}: "
                        f"Loss={stats['loss']:.4f}, "
                        f"Reward={stats['avg_reward']:.4f}, "
                        f"Acc={stats['accuracy']:.2%}, "
                        f"Length={stats['avg_length']:.0f}, "
                        f"Entropy={stats['entropy']:.4f}"
                    )

            # Epoch summary
            logger.info(f"\nEpoch {epoch+1} Summary:")
            logger.info(f"  Average Loss: {np.mean(epoch_stats['losses']):.4f}")
            logger.info(f"  Average Reward: {np.mean(epoch_stats['rewards']):.4f}")
            logger.info(f"  Average Accuracy: {np.mean(epoch_stats['accuracies']):.2%}")
            logger.info(f"  Average Length: {np.mean(epoch_stats['lengths']):.0f}")
            logger.info(f"  Average Entropy: {np.mean(epoch_stats['entropies']):.4f}")

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
        with open(f"{checkpoint_dir}/stats.json", "w", encoding="utf-8") as f:
            json.dump(self.stats, f, indent=2, ensure_ascii=False)

        logger.info(f"Checkpoint saved to {checkpoint_dir}")

    def save_final_model(self):
        """Save final trained model"""
        final_dir = f"{self.config.output_dir}/final"
        os.makedirs(final_dir, exist_ok=True)

        self.model.save_pretrained(final_dir)
        self.tokenizer.save_pretrained(final_dir)

        # Save final stats
        with open(f"{final_dir}/training_stats.json", "w", encoding="utf-8") as f:
            json.dump(self.stats, f, indent=2, ensure_ascii=False)

        logger.info(f"Final model saved to {final_dir}")


def main():
    """Main training function"""
    # Load configuration
    config = DAPOConfig()

    # Load training data
    data_path = "data/sample_reasoning_data.json"
    if not os.path.exists(data_path):
        logger.error(f"Training data not found at {data_path}")
        logger.info("Please prepare training data with format:")
        logger.info('[{"question": "...", "correct_answer": "..."}, ...]')
        return

    with open(data_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    logger.info(f"Loaded {len(dataset)} training examples")

    # Initialize trainer
    trainer = DAPOTrainer(config)

    # Start training
    trainer.train(dataset)


if __name__ == "__main__":
    main()
