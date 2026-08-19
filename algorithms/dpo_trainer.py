"""
DPO (Direct Preference Optimization) Training Implementation

This implementation provides another comparison baseline for GRPO.

Author: Aitachi
Contact: 44158892@qq.com
Date: 2025

Mathematical Formulation:
=======================

DPO optimizes preferences directly without explicit reward modeling.

The DPO loss function is:

L_DPO(π_θ; π_ref) = -E_{(x,y_w,y_l)~D}[
    log σ(β log π_θ(y_w|x)/π_ref(y_w|x) - β log π_θ(y_l|x)/π_ref(y_l|x))
]

Where:
- x: Input prompt/question
- y_w: Preferred (winning) response
- y_l: Rejected (losing) response
- π_θ: Policy being optimized
- π_ref: Reference policy (frozen)
- β: Temperature parameter controlling deviation from reference
- σ: Sigmoid function

Equivalently, this can be written as:

L_DPO = -E[log σ(β * (log r_w - log r_l))]

Where:
- r_w = π_θ(y_w|x) / π_ref(y_w|x)  # Ratio for preferred response
- r_l = π_θ(y_l|x) / π_ref(y_l|x)  # Ratio for rejected response

Key Differences from GRPO and PPO:
=================================
1. DPO works on preference pairs (y_w, y_l) rather than absolute rewards
2. No need for value network (like GRPO, unlike PPO)
3. No need for sampling multiple outputs per question (unlike GRPO)
4. Directly optimizes preference probability via Bradley-Terry model
5. More stable than RLHF but requires preference data

Advantages:
- Simple and stable
- No reward model needed
- No value network needed
- Works well for alignment tasks

Disadvantages:
- Requires preference pairs in training data
- May not be as sample-efficient as GRPO
- Less suitable for tasks with sparse absolute rewards
"""

import os
import json
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Tuple
import numpy as np
from tqdm import tqdm
import logging
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DPOConfig:
    """Configuration for DPO training"""

    # Model parameters
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"

    # DPO hyperparameters
    beta: float = 0.1  # β - temperature parameter for DPO loss

    # Training parameters
    learning_rate: float = 5e-6
    max_epochs: int = 3
    batch_size: int = 4
    max_length: int = 512
    temperature: float = 0.7

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Paths
    output_dir: str = "./checkpoints/dpo_model"


class DPOTrainer:
    """
    DPO (Direct Preference Optimization) Trainer

    This trainer implements DPO for optimizing language models based on preferences.
    """

    def __init__(self, config: DPOConfig):
        self.config = config
        self.device = torch.device(config.device)

        # Load policy model
        logger.info(f"Loading policy model: {config.model_name}")
        self.policy_model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.float16 if "cuda" in config.device else torch.float32,
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load reference model (frozen)
        logger.info("Creating reference model")
        self.ref_model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.float16 if "cuda" in config.device else torch.float32,
            device_map="auto"
        )
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.policy_model.parameters(),
            lr=config.learning_rate
        )

        # Training statistics
        self.stats = {
            "epoch_losses": [],
            "epoch_accuracies": [],  # Preference prediction accuracy
            "epoch_margins": []  # Margin between preferred and rejected
        }

    def generate_response(self, prompt: str) -> str:
        """
        Generate a single response for a prompt.

        Args:
            prompt: Input question

        Returns:
            Generated response
        """
        formatted_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            max_length=self.config.max_length,
            truncation=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.policy_model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=self.config.temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=False)

        # Extract assistant's response
        if "<|im_start|>assistant\n" in response:
            response = response.split("<|im_start|>assistant\n")[-1]
            if "<|im_end|>" in response:
                response = response.split("<|im_end|>")[0]

        return response.strip()

    def create_preference_pair(
        self,
        question: str,
        correct_answer: str
    ) -> Tuple[str, str]:
        """
        Create a preference pair by generating two responses and ranking them.

        Args:
            question: Input question
            correct_answer: Ground truth answer

        Returns:
            Tuple of (preferred_response, rejected_response)
        """
        # Generate two different responses
        response1 = self.generate_response(question)
        response2 = self.generate_response(question)

        # Rank them using reward model
        from algorithms.grpo_trainer import RewardModel
        reward_model = RewardModel(self.config)

        reward1 = reward_model.compute_reward(question, response1, correct_answer)
        reward2 = reward_model.compute_reward(question, response2, correct_answer)

        # Create preference pair
        if reward1 >= reward2:
            return response1, response2
        else:
            return response2, response1

    def compute_log_prob(
        self,
        model: torch.nn.Module,
        prompt: str,
        response: str
    ) -> torch.Tensor:
        """
        Compute log probability of a response given a prompt.

        Args:
            model: Language model
            prompt: Input prompt
            response: Generated response

        Returns:
            Log probability tensor
        """
        formatted_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        full_text = formatted_prompt + response

        inputs = self.tokenizer(
            full_text,
            return_tensors="pt",
            max_length=self.config.max_length,
            truncation=True
        ).to(self.device)

        outputs = model(**inputs, labels=inputs["input_ids"])
        log_prob = -outputs.loss  # Negative loss is log probability

        return log_prob

    def dpo_loss(
        self,
        prompt: str,
        preferred_response: str,
        rejected_response: str
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute DPO loss for a preference pair.

        Formula:
        L = -log σ(β * (log π_θ(y_w|x)/π_ref(y_w|x) - log π_θ(y_l|x)/π_ref(y_l|x)))

        Args:
            prompt: Input prompt
            preferred_response: Preferred (winning) response
            rejected_response: Rejected (losing) response

        Returns:
            Tuple of (loss, statistics_dict)
        """
        # Compute log probs for preferred response
        log_prob_preferred_policy = self.compute_log_prob(
            self.policy_model, prompt, preferred_response
        )

        with torch.no_grad():
            log_prob_preferred_ref = self.compute_log_prob(
                self.ref_model, prompt, preferred_response
            )

        # Compute log probs for rejected response
        log_prob_rejected_policy = self.compute_log_prob(
            self.policy_model, prompt, rejected_response
        )

        with torch.no_grad():
            log_prob_rejected_ref = self.compute_log_prob(
                self.ref_model, prompt, rejected_response
            )

        # Compute log ratios
        log_ratio_preferred = log_prob_preferred_policy - log_prob_preferred_ref
        log_ratio_rejected = log_prob_rejected_policy - log_prob_rejected_ref

        # DPO loss
        logits = self.config.beta * (log_ratio_preferred - log_ratio_rejected)
        loss = -F.logsigmoid(logits)

        # Statistics
        with torch.no_grad():
            preferred_prob = torch.sigmoid(logits)
            margin = log_ratio_preferred - log_ratio_rejected

        stats = {
            "preferred_prob": preferred_prob.item(),
            "margin": margin.item(),
            "correct": 1.0 if preferred_prob.item() > 0.5 else 0.0
        }

        return loss, stats

    def train_step(
        self,
        question: str,
        correct_answer: str
    ) -> Dict[str, float]:
        """
        Perform one DPO training step.

        Args:
            question: Input question
            correct_answer: Ground truth answer

        Returns:
            Training statistics
        """
        # Create preference pair
        preferred, rejected = self.create_preference_pair(question, correct_answer)

        # Compute DPO loss
        loss, stats = self.dpo_loss(question, preferred, rejected)

        # Backward and optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        stats["loss"] = loss.item()
        return stats

    def train(self, dataset: List[Dict]):
        """
        Train using DPO algorithm.

        Args:
            dataset: Training dataset
        """
        logger.info(f"Starting DPO training with {len(dataset)} examples")
        logger.info(f"Beta: {self.config.beta}")

        self.policy_model.train()

        for epoch in range(self.config.max_epochs):
            logger.info(f"\n{'='*50}")
            logger.info(f"Epoch {epoch + 1}/{self.config.max_epochs}")
            logger.info(f"{'='*50}")

            epoch_losses = []
            epoch_accuracies = []
            epoch_margins = []

            for idx, example in enumerate(tqdm(dataset, desc=f"Epoch {epoch+1}")):
                stats = self.train_step(
                    question=example["question"],
                    correct_answer=example["correct_answer"]
                )

                epoch_losses.append(stats["loss"])
                epoch_accuracies.append(stats["correct"])
                epoch_margins.append(stats["margin"])

                if (idx + 1) % 10 == 0:
                    logger.info(
                        f"Step {idx+1}: Loss={stats['loss']:.4f}, "
                        f"Accuracy={stats['correct']:.2f}, "
                        f"Margin={stats['margin']:.4f}"
                    )

            # Epoch summary
            self.stats["epoch_losses"].append(np.mean(epoch_losses))
            self.stats["epoch_accuracies"].append(np.mean(epoch_accuracies))
            self.stats["epoch_margins"].append(np.mean(epoch_margins))

            logger.info(f"\nEpoch {epoch+1} Summary:")
            logger.info(f"  Avg Loss: {np.mean(epoch_losses):.4f}")
            logger.info(f"  Avg Accuracy: {np.mean(epoch_accuracies):.2%}")
            logger.info(f"  Avg Margin: {np.mean(epoch_margins):.4f}")

            self.save_checkpoint(epoch)

        logger.info("\nDPO Training completed!")
        self.save_final_model()

    def save_checkpoint(self, epoch: int):
        """Save model checkpoint"""
        checkpoint_dir = f"{self.config.output_dir}/epoch_{epoch+1}"
        os.makedirs(checkpoint_dir, exist_ok=True)

        self.policy_model.save_pretrained(checkpoint_dir)
        self.tokenizer.save_pretrained(checkpoint_dir)

        with open(f"{checkpoint_dir}/stats.json", "w") as f:
            json.dump(self.stats, f, indent=2)

        logger.info(f"Checkpoint saved to {checkpoint_dir}")

    def save_final_model(self):
        """Save final model"""
        final_dir = f"{self.config.output_dir}/final"
        os.makedirs(final_dir, exist_ok=True)

        self.policy_model.save_pretrained(final_dir)
        self.tokenizer.save_pretrained(final_dir)

        with open(f"{final_dir}/training_stats.json", "w") as f:
            json.dump(self.stats, f, indent=2)

        logger.info(f"Final model saved to {final_dir}")


def main():
    """Main training function"""
    config = DPOConfig()

    with open("data/sample_reasoning_data.json", "r") as f:
        dataset = json.load(f)

    logger.info(f"Loaded {len(dataset)} training examples")

    trainer = DPOTrainer(config)
    trainer.train(dataset)


if __name__ == "__main__":
    main()
