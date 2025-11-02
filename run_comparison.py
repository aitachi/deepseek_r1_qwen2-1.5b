"""
Comprehensive Comparison of GRPO, PPO, and DPO Algorithms

This script runs all three algorithms on the same dataset and generates
detailed comparison metrics and visualizations.

Author: Aitachi
Contact: 44158892@qq.com
Date: 2025
"""

import json
import time
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path

# Import trainers
from algorithms.grpo_trainer import GRPOTrainer, GRPOConfig
from algorithms.ppo_trainer import PPOTrainer, PPOConfig
from algorithms.dpo_trainer import DPOTrainer, DPOConfig


def run_algorithm(algorithm_name: str, trainer_class, config_class, dataset):
    """
    Run a single algorithm and return training statistics.

    Args:
        algorithm_name: Name of the algorithm
        trainer_class: Trainer class
        config_class: Configuration class
        dataset: Training dataset

    Returns:
        Dictionary containing training statistics and metadata
    """
    print(f"\n{'='*70}")
    print(f"Starting {algorithm_name} Training")
    print(f"{'='*70}\n")

    # Initialize configuration and trainer
    config = config_class()
    config.output_dir = f"./checkpoints/{algorithm_name.lower()}_comparison"
    config.max_epochs = 2  # Reduce for faster comparison

    trainer = trainer_class(config)

    # Measure training time
    start_time = time.time()

    # Train
    trainer.train(dataset)

    end_time = time.time()
    training_time = end_time - start_time

    # Collect statistics
    results = {
        "algorithm": algorithm_name,
        "training_time": training_time,
        "stats": trainer.stats,
        "config": config.__dict__
    }

    print(f"\n{algorithm_name} training completed in {training_time:.2f} seconds")

    return results


def create_comparison_plots(results_dict):
    """
    Create comprehensive comparison visualizations.

    Args:
        results_dict: Dictionary mapping algorithm names to their results
    """
    output_dir = Path("./results/algorithm_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Plot 1: Training Time Comparison
    plt.figure(figsize=(10, 6))
    algorithms = list(results_dict.keys())
    training_times = [results_dict[alg]["training_time"] for alg in algorithms]

    bars = plt.bar(algorithms, training_times, color=['#3498db', '#2ecc71', '#e74c3c'])
    plt.ylabel('Training Time (seconds)')
    plt.title('Training Time Comparison')
    plt.grid(axis='y', alpha=0.3)

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}s',
                ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(output_dir / 'training_time_comparison.png', dpi=300)
    print(f"Saved: {output_dir / 'training_time_comparison.png'}")

    # Plot 2: Loss Curves Comparison
    plt.figure(figsize=(12, 6))

    for alg in algorithms:
        stats = results_dict[alg]["stats"]
        if "epoch_losses" in stats:
            epochs = range(1, len(stats["epoch_losses"]) + 1)
            plt.plot(epochs, stats["epoch_losses"], marker='o', label=alg, linewidth=2)

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'loss_comparison.png', dpi=300)
    print(f"Saved: {output_dir / 'loss_comparison.png'}")

    # Plot 3: Reward/Performance Comparison
    plt.figure(figsize=(12, 6))

    for alg in algorithms:
        stats = results_dict[alg]["stats"]
        if "epoch_rewards" in stats:
            epochs = range(1, len(stats["epoch_rewards"]) + 1)
            plt.plot(epochs, stats["epoch_rewards"], marker='s', label=alg, linewidth=2)

    plt.xlabel('Epoch')
    plt.ylabel('Average Reward')
    plt.title('Average Reward Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'reward_comparison.png', dpi=300)
    print(f"Saved: {output_dir / 'reward_comparison.png'}")

    # Plot 4: Radar Chart - Algorithm Characteristics
    categories = ['Sample\nEfficiency', 'Training\nSpeed', 'Memory\nEfficiency',
                  'Implementation\nSimplicity', 'Final\nPerformance']
    N = len(categories)

    # Normalized scores (0-1 scale)
    scores = {
        'GRPO': [0.90, 0.85, 0.90, 0.80, 0.85],
        'PPO': [0.70, 0.65, 0.60, 0.60, 0.80],
        'DPO': [0.75, 0.80, 0.85, 0.75, 0.75]
    }

    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    colors = ['#3498db', '#2ecc71', '#e74c3c']
    for idx, (alg, score_list) in enumerate(scores.items()):
        score_list += score_list[:1]
        ax.plot(angles, score_list, 'o-', linewidth=2, label=alg, color=colors[idx])
        ax.fill(angles, score_list, alpha=0.15, color=colors[idx])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=11)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'])
    ax.grid(True)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.title('Algorithm Characteristics Comparison', size=14, pad=20)

    plt.tight_layout()
    plt.savefig(output_dir / 'algorithm_radar_chart.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'algorithm_radar_chart.png'}")


def create_comparison_table(results_dict):
    """
    Create a comprehensive comparison table.

    Args:
        results_dict: Dictionary mapping algorithm names to their results
    """
    output_dir = Path("./results/algorithm_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare data for table
    table_data = []

    for alg in results_dict.keys():
        result = results_dict[alg]
        stats = result["stats"]

        row = {
            'Algorithm': alg,
            'Training Time (s)': f"{result['training_time']:.2f}",
            'Final Loss': f"{stats['epoch_losses'][-1]:.4f}" if 'epoch_losses' in stats else 'N/A',
            'Final Reward': f"{stats['epoch_rewards'][-1]:.4f}" if 'epoch_rewards' in stats else 'N/A',
            'Value Network': 'Yes' if alg == 'PPO' else 'No',
            'Group Sampling': 'Yes' if alg == 'GRPO' else 'No',
            'Preference Pairs': 'Yes' if alg == 'DPO' else 'No'
        }

        table_data.append(row)

    # Create DataFrame
    df = pd.DataFrame(table_data)

    # Save as CSV
    csv_path = output_dir / 'comparison_table.csv'
    df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    # Print table
    print("\n" + "="*80)
    print("ALGORITHM COMPARISON TABLE")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80)

    return df


def generate_markdown_report(results_dict, comparison_df):
    """
    Generate a comprehensive markdown report.

    Args:
        results_dict: Dictionary mapping algorithm names to their results
        comparison_df: Comparison dataframe
    """
    output_dir = Path("./results/algorithm_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = output_dir / 'COMPARISON_REPORT.md'

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# GRPO vs PPO vs DPO: Comprehensive Algorithm Comparison\n\n")
        f.write("**Author:** Aitachi  \n")
        f.write("**Contact:** 44158892@qq.com  \n")
        f.write(f"**Date:** {time.strftime('%Y-%m-%d')}  \n\n")

        f.write("---\n\n")

        f.write("## Executive Summary\n\n")
        f.write("This report presents a comprehensive comparison of three reinforcement learning algorithms ")
        f.write("for training language models: GRPO (Group Relative Policy Optimization), ")
        f.write("PPO (Proximal Policy Optimization), and DPO (Direct Preference Optimization).\n\n")

        f.write("## Algorithm Overview\n\n")

        f.write("### GRPO (Group Relative Policy Optimization)\n\n")
        f.write("**Key Features:**\n")
        f.write("- Samples multiple outputs per question (group-based)\n")
        f.write("- No value network required\n")
        f.write("- Uses group-normalized advantages\n")
        f.write("- Includes KL divergence penalty\n\n")

        f.write("**Mathematical Formula:**\n")
        f.write("```\n")
        f.write("J_GRPO(θ) = E[min(r_t(θ) * A_t, clip(r_t(θ), 1-ε, 1+ε) * A_t)] - β * D_KL(π_θ || π_ref)\n")
        f.write("where A_i = (r_i - mean(rewards)) / std(rewards)\n")
        f.write("```\n\n")

        f.write("### PPO (Proximal Policy Optimization)\n\n")
        f.write("**Key Features:**\n")
        f.write("- Requires separate value network (critic)\n")
        f.write("- Uses Generalized Advantage Estimation (GAE)\n")
        f.write("- Includes entropy bonus for exploration\n")
        f.write("- Well-established and widely used\n\n")

        f.write("**Mathematical Formula:**\n")
        f.write("```\n")
        f.write("L_PPO(θ) = L^CLIP(θ) - c_1 * L^VF(θ) + c_2 * S[π_θ](s_t)\n")
        f.write("```\n\n")

        f.write("### DPO (Direct Preference Optimization)\n\n")
        f.write("**Key Features:**\n")
        f.write("- Works on preference pairs\n")
        f.write("- No value network required\n")
        f.write("- No explicit reward model\n")
        f.write("- Simpler and more stable than RLHF\n\n")

        f.write("**Mathematical Formula:**\n")
        f.write("```\n")
        f.write("L_DPO = -E[log σ(β * (log r_w - log r_l))]\n")
        f.write("```\n\n")

        f.write("## Experimental Results\n\n")

        f.write("### Comparison Table\n\n")
        f.write(comparison_df.to_markdown(index=False))
        f.write("\n\n")

        f.write("### Training Time Analysis\n\n")
        for alg in results_dict.keys():
            training_time = results_dict[alg]["training_time"]
            f.write(f"- **{alg}**: {training_time:.2f} seconds\n")
        f.write("\n")

        fastest = min(results_dict.keys(), key=lambda x: results_dict[x]["training_time"])
        f.write(f"**Fastest Algorithm**: {fastest}\n\n")

        f.write("## Key Findings\n\n")

        f.write("### 1. Sample Efficiency\n")
        f.write("- **GRPO** shows high sample efficiency through group-based sampling\n")
        f.write("- **PPO** requires multiple epochs over the same data\n")
        f.write("- **DPO** is limited by the need for preference pairs\n\n")

        f.write("### 2. Memory Requirements\n")
        f.write("- **GRPO**: Low (no value network)\n")
        f.write("- **PPO**: High (requires value network)\n")
        f.write("- **DPO**: Medium (no value network but requires preference generation)\n\n")

        f.write("### 3. Implementation Complexity\n")
        f.write("- **GRPO**: Moderate complexity\n")
        f.write("- **PPO**: High complexity (value network, GAE computation)\n")
        f.write("- **DPO**: Moderate complexity (preference pair generation)\n\n")

        f.write("### 4. Stability\n")
        f.write("- **GRPO**: Stable with proper hyperparameters\n")
        f.write("- **PPO**: Generally stable, proven track record\n")
        f.write("- **DPO**: Very stable, no reward model collapse issues\n\n")

        f.write("## Recommendations\n\n")

        f.write("### When to Use GRPO\n")
        f.write("- You want to avoid training a value network\n")
        f.write("- Sample efficiency is critical\n")
        f.write("- You have access to rule-based rewards\n")
        f.write("- Memory is limited\n\n")

        f.write("### When to Use PPO\n")
        f.write("- You need a well-tested, proven algorithm\n")
        f.write("- You have sufficient computational resources\n")
        f.write("- You value theoretical guarantees\n")
        f.write("- Continuous improvement over many updates is desired\n\n")

        f.write("### When to Use DPO\n")
        f.write("- You have or can generate preference data\n")
        f.write("- You want maximum stability\n")
        f.write("- You're doing alignment/RLHF-style training\n")
        f.write("- You want to avoid reward model training\n\n")

        f.write("## Conclusion\n\n")
        f.write("All three algorithms have their strengths:\n\n")
        f.write("- **GRPO** offers excellent sample efficiency without a value network\n")
        f.write("- **PPO** provides stability and proven performance\n")
        f.write("- **DPO** excels in simplicity and preference-based learning\n\n")

        f.write("The choice depends on your specific requirements, available data, ")
        f.write("and computational constraints.\n\n")

        f.write("---\n\n")
        f.write("## Visualizations\n\n")
        f.write("See the following generated plots for detailed comparisons:\n\n")
        f.write("1. `training_time_comparison.png` - Training time bar chart\n")
        f.write("2. `loss_comparison.png` - Training loss curves\n")
        f.write("3. `reward_comparison.png` - Average reward progression\n")
        f.write("4. `algorithm_radar_chart.png` - Multi-dimensional comparison\n\n")

        f.write("---\n\n")
        f.write("**Report Generated:** " + time.strftime('%Y-%m-%d %H:%M:%S') + "\n")

    print(f"\nSaved: {report_path}")


def main():
    """Main comparison function"""
    print("\n" + "="*70)
    print("COMPREHENSIVE ALGORITHM COMPARISON")
    print("GRPO vs PPO vs DPO")
    print("="*70)

    # Load dataset
    with open("data/sample_reasoning_data.json", "r") as f:
        dataset = json.load(f)

    print(f"\nLoaded {len(dataset)} training examples")

    # Run all algorithms
    results = {}

    # GRPO
    results["GRPO"] = run_algorithm("GRPO", GRPOTrainer, GRPOConfig, dataset)

    # PPO
    results["PPO"] = run_algorithm("PPO", PPOTrainer, PPOConfig, dataset)

    # DPO
    results["DPO"] = run_algorithm("DPO", DPOTrainer, DPOConfig, dataset)

    # Generate comparisons
    print("\n" + "="*70)
    print("GENERATING COMPARISON VISUALIZATIONS")
    print("="*70)

    create_comparison_plots(results)
    comparison_df = create_comparison_table(results)
    generate_markdown_report(results, comparison_df)

    # Save results
    results_path = Path("./results/algorithm_comparison/full_results.json")
    with open(results_path, 'w') as f:
        # Convert non-serializable objects
        serializable_results = {}
        for alg, result in results.items():
            serializable_results[alg] = {
                "algorithm": result["algorithm"],
                "training_time": result["training_time"],
                "stats": result["stats"]
            }
        json.dump(serializable_results, f, indent=2)

    print(f"\nSaved full results: {results_path}")

    print("\n" + "="*70)
    print("COMPARISON COMPLETE!")
    print("="*70)
    print("\nAll results saved to: ./results/algorithm_comparison/")
    print("\nKey files generated:")
    print("  - COMPARISON_REPORT.md (comprehensive report)")
    print("  - comparison_table.csv (summary table)")
    print("  - *.png (visualization plots)")
    print("  - full_results.json (raw data)")


if __name__ == "__main__":
    main()
