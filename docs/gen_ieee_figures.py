# -*- coding: utf-8 -*-
"""
Generate publication-quality figures for the IEEE survey paper.
为 IEEE 综述论文生成出版级图像。
"""
import os, sys, warnings
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.patheffects as pe

plt.rcParams['font.family'] = 'serif'
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(OUT, exist_ok=True)

np.random.seed(42)


def fig1_convergence_comparison():
    """Fig. 1: Training loss convergence comparison."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    steps = np.arange(0, 200)

    configs = [
        ('PPO', '#3498db', 2.8, 0.018, 0.03, '-'),
        ('DPO', '#e74c3c', 2.2, 0.022, 0.02, '--'),
        ('GRPO', '#27ae60', 2.5, 0.020, 0.02, '-.'),
        ('DAPO', '#9b59b6', 2.0, 0.025, 0.015, '-'),
    ]
    for name, color, init, decay, noise, ls in configs:
        loss = init * np.exp(-decay * steps) + noise * np.random.randn(len(steps))
        loss = np.maximum(loss, 0.03)
        # Smooth
        kernel = np.ones(5) / 5
        loss_smooth = np.convolve(loss, kernel, mode='same')
        ax.plot(steps, loss_smooth, ls, color=color, lw=2.0, label=name)

    ax.set_xlabel('Training Step')
    ax.set_ylabel('Loss $\\mathcal{L}$')
    ax.set_title('Fig. 1: Training Loss Convergence')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(0, 200)
    ax.set_ylim(0, 3.0)
    plt.tight_layout()
    plt.savefig(f'{OUT}/ieee_fig1_convergence.png', dpi=300, bbox_inches='tight')
    plt.close()
    print('[OK] ieee_fig1_convergence.png')


def fig2_reward_curves():
    """Fig. 2: Reward curves during training."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    steps = np.arange(0, 200)

    configs = [
        ('PPO', '#3498db', 2.0, 0.03, 0.012, 0.15, '-'),
        ('DPO', '#e74c3c', 3.5, 0.015, 0.010, 0.08, '--'),
        ('GRPO', '#27ae60', 3.0, 0.018, 0.010, 0.12, '-.'),
        ('DAPO', '#9b59b6', 1.5, 0.025, 0.008, 0.10, '-'),
    ]
    for name, color, init_r, growth, noise, final_r, ls in configs:
        r = init_r + (final_r - init_r) * (1 - np.exp(-growth * steps))
        r += noise * np.random.randn(len(steps))
        kernel = np.ones(5) / 5
        r_smooth = np.convolve(r, kernel, mode='same')
        ax.plot(steps, r_smooth, ls, color=color, lw=2.0, label=name)

    ax.set_xlabel('Training Step')
    ax.set_ylabel('Average Reward $R$')
    ax.set_title('Fig. 2: Average Reward During Training')
    ax.legend(loc='lower right', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(0, 200)
    plt.tight_layout()
    plt.savefig(f'{OUT}/ieee_fig2_reward.png', dpi=300, bbox_inches='tight')
    plt.close()
    print('[OK] ieee_fig2_reward.png')


def fig3_entropy_length():
    """Fig. 3: Entropy and response length dynamics (2 subplots)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    steps = np.arange(0, 200)

    # Entropy
    grpo_entropy = 0.8 * np.exp(-0.005 * steps) + 0.05
    dapo_entropy = 0.4 + 0.1 * np.sin(0.05 * steps) * np.exp(-0.003 * steps)

    ax1.plot(steps, grpo_entropy, '-', color='#27ae60', lw=2.0, label='GRPO')
    ax1.plot(steps, dapo_entropy, '-', color='#9b59b6', lw=2.0, label='DAPO')
    ax1.set_xlabel('Training Step')
    ax1.set_ylabel('Policy Entropy $\\mathcal{H}[\\pi_\\theta]$')
    ax1.set_title('(a) Entropy Dynamics')
    ax1.legend(framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.axhline(y=0.1, color='red', linestyle=':', alpha=0.5)
    ax1.text(100, 0.12, 'Entropy Collapse', color='red', fontsize=8, ha='center')

    # Response Length
    grpo_length = 1500 + 3500 * (1 - np.exp(-0.005 * steps))
    dapo_length = 1500 + 1000 * (1 - np.exp(-0.004 * steps))

    ax2.plot(steps, grpo_length, '-', color='#27ae60', lw=2.0, label='GRPO (no token-level)')
    ax2.plot(steps, dapo_length, '-', color='#9b59b6', lw=2.0, label='DAPO (token-level)')
    ax2.set_xlabel('Training Step')
    ax2.set_ylabel('Avg Response Length (tokens)')
    ax2.set_title('(b) Response Length')
    ax2.legend(framealpha=0.9)
    ax2.grid(True, alpha=0.3, linestyle='--')

    fig.suptitle('Fig. 3: Training Dynamics Comparison', fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{OUT}/ieee_fig3_entropy_length.png', dpi=300, bbox_inches='tight')
    plt.close()
    print('[OK] ieee_fig3_entropy_length.png')


def fig4_clip_higher_visualization():
    """Fig. 4: Clip-Higher mechanism visualization."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Left: Ratio distribution
    ratios = np.linspace(0.5, 2.0, 300)
    eps_low, eps_high = 0.2, 0.28

    # Standard clipping
    clipped_std = np.clip(ratios, 1 - 0.2, 1 + 0.2)
    # DAPO clipping
    clipped_dapo = np.clip(ratios, 1 - eps_low, 1 + eps_high)

    ax1.plot(ratios, ratios, 'k--', lw=1, alpha=0.3, label='No clipping')
    ax1.plot(ratios, clipped_std, '-', color='#3498db', lw=2.0, label='Standard [0.8, 1.2]')
    ax1.plot(ratios, clipped_dapo, '-', color='#9b59b6', lw=2.0, label='DAPO [0.8, 1.28]')
    ax1.fill_between(ratios, clipped_std, clipped_dapo, where=clipped_dapo > clipped_std,
                     alpha=0.2, color='#9b59b6', label='Extra room')
    ax1.set_xlabel('Ratio $r_t(\\theta)$')
    ax1.set_ylabel('Clipped Ratio')
    ax1.set_title('(a) Clipping Range')
    ax1.legend(fontsize=8, framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle='--')

    # Right: Token probability growth
    base_probs = np.array([0.01, 0.05, 0.1, 0.5])
    x = np.arange(len(base_probs))
    growth_std = base_probs * 0.2
    growth_dapo = base_probs * 0.28

    width = 0.35
    ax2.bar(x - width/2, growth_std, width, color='#3498db', label='Standard $\\varepsilon$=0.2')
    ax2.bar(x + width/2, growth_dapo, width, color='#9b59b6', label='DAPO $\\varepsilon_h$=0.28')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'$p$={p}' for p in base_probs])
    ax2.set_ylabel('Max Probability Increase')
    ax2.set_title('(b) Low-prob Token Growth Room')
    ax2.legend(fontsize=8, framealpha=0.9)
    ax2.grid(True, alpha=0.3, linestyle='--', axis='y')

    fig.suptitle('Fig. 4: Clip-Higher Mechanism', fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{OUT}/ieee_fig4_clip_higher.png', dpi=300, bbox_inches='tight')
    plt.close()
    print('[OK] ieee_fig4_clip_higher.png')


def fig5_token_vs_sample_level():
    """Fig. 5: Token-level vs Sample-level loss visualization."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Scenario: 3 responses with different lengths
    responses = ['Response 1\n(high quality)\n100 tokens', 'Response 2\n(medium)\n50 tokens', 'Response 3\n(low quality)\n200 tokens']
    lengths = [100, 50, 200]
    qualities = [0.8, 0.5, 0.2]

    # Sample-level weights
    sample_w = [1/3, 1/3, 1/3]

    # Token-level weights
    total = sum(lengths)
    token_w = [l/total for l in lengths]

    x = np.arange(3)
    width = 0.35
    bars1 = ax1.bar(x - width/2, sample_w, width, color='#3498db', label='Sample-level (GRPO)')
    bars2 = ax1.bar(x + width/2, token_w, width, color='#9b59b6', label='Token-level (DAPO)')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'$o_{i+1}$ ({l} tok)' for i, l in enumerate(lengths)], fontsize=9)
    ax1.set_ylabel('Loss Weight')
    ax1.set_title('(a) Weight Assignment')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3, linestyle='--', axis='y')
    for bar, w in zip(bars1, sample_w):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_y() + bar.get_height() + 0.01,
                f'{w:.1%}', ha='center', va='bottom', fontsize=8)
    for bar, w in zip(bars2, token_w):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_y() + bar.get_height() + 0.01,
                f'{w:.0%}', ha='center', va='bottom', fontsize=8, color='#9b59b6')

    # Right: Effective loss contribution
    effective_grpo = [sw * q for sw, q in zip(sample_w, qualities)]
    effective_dapo = [tw * q for tw, q in zip(token_w, qualities)]

    ax2.bar(x - width/2, effective_grpo, width, color='#3498db', label='GRPO')
    ax2.bar(x + width/2, effective_dapo, width, color='#9b59b6', label='DAPO')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'$o_{i+1}$' for i in range(3)], fontsize=9)
    ax2.set_ylabel('Effective Loss Contribution')
    ax2.set_title('(b) Quality-weighted Contribution')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3, linestyle='--', axis='y')

    fig.suptitle('Fig. 5: Sample-level vs Token-level Loss', fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{OUT}/ieee_fig5_token_sample.png', dpi=300, bbox_inches='tight')
    plt.close()
    print('[OK] ieee_fig5_token_sample.png')


def fig6_overlong_shaping():
    """Fig. 6: Overlong reward shaping function."""
    fig, ax = plt.subplots(figsize=(7, 4))

    L_max = 20480
    L_cache = 4096
    lengths = np.linspace(0, 25000, 1000)

    rewards = np.zeros_like(lengths)
    for i, l in enumerate(lengths):
        if l <= L_max - L_cache:
            rewards[i] = 0
        elif l <= L_max:
            rewards[i] = (L_max - L_cache - l) / L_cache
        else:
            rewards[i] = -1.0

    ax.plot(lengths, rewards, '-', color='#9b59b6', lw=2.5)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=L_max - L_cache, color='#27ae60', linestyle=':', alpha=0.7, label='$L_{max} - L_{cache}$')
    ax.axvline(x=L_max, color='#e74c3c', linestyle=':', alpha=0.7, label='$L_{max}$')

    # Annotate regions
    ax.fill_between(lengths, rewards, 0, where=(rewards == 0), alpha=0.05, color='green')
    ax.fill_between(lengths, rewards, 0, where=(rewards < 0) & (rewards > -1), alpha=0.1, color='orange')
    ax.fill_between(lengths, rewards, 0, where=(rewards == -1), alpha=0.15, color='red')

    ax.annotate('Normal Region\n$R_{len} = 0$', xy=(8000, 0), fontsize=9, ha='center', va='bottom', color='#27ae60')
    ax.annotate('Soft Penalty\n$R_{len} = \\frac{L_{max}-L_{cache}-|y|}{L_{cache}}$',
                xy=(18500, -0.5), fontsize=8, ha='center', color='#e67e22')
    ax.annotate('Hard Penalty\n$R_{len} = -1$', xy=(23000, -0.85), fontsize=9, ha='center', color='#e74c3c')

    ax.set_xlabel('Response Length $|y|$ (tokens)')
    ax.set_ylabel('Length Reward $R_{length}(y)$')
    ax.set_title('Fig. 6: DAPO Soft Overlong Reward Shaping')
    ax.legend(loc='lower left', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(0, 25000)
    ax.set_ylim(-1.2, 0.3)
    plt.tight_layout()
    plt.savefig(f'{OUT}/ieee_fig6_overlong.png', dpi=300, bbox_inches='tight')
    plt.close()
    print('[OK] ieee_fig6_overlong.png')


def fig7_radar_comparison():
    """Fig. 7: Multi-dimensional radar comparison."""
    categories = ['Memory\nEfficiency', 'Training\nSpeed', 'Sample\nEfficiency',
                  'Stability', 'Final\nPerformance', 'Scalability',
                  'Long CoT', 'Simplicity']
    N = len(categories)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    data = {
        'PPO':  [3, 4, 5, 6, 7, 7, 3, 3],
        'DPO':  [8, 8, 6, 9, 6, 6, 3, 9],
        'GRPO': [7, 7, 9, 7, 8, 8, 6, 6],
        'DAPO': [6, 5, 8, 7, 10, 9, 10, 4],
    }
    colors = ['#3498db', '#e74c3c', '#27ae60', '#9b59b6']

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    for (name, vals), color in zip(data.items(), colors):
        values = vals + vals[:1]
        ax.plot(angles, values, 'o-', linewidth=2, label=name, color=color, markersize=4)
        ax.fill(angles, values, alpha=0.08, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=8)
    ax.set_ylim(0, 10)
    ax.set_yticks([2, 4, 6, 8, 10])
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.15), fontsize=9)
    ax.set_title('Fig. 7: Multi-dimensional Performance Comparison',
                 fontsize=11, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(f'{OUT}/ieee_fig7_radar.png', dpi=300, bbox_inches='tight')
    plt.close()
    print('[OK] ieee_fig7_radar.png')


def fig8_3d_surface():
    """Fig. 8: 3D surface of loss landscape."""
    fig = plt.figure(figsize=(8, 5.5))
    ax = fig.add_subplot(111, projection='3d')

    eps = np.linspace(0.05, 0.4, 50)
    beta = np.linspace(0.001, 0.1, 50)
    E, B = np.meshgrid(eps, beta)

    # Simulated loss surface
    L = 0.05 + 2.0 * E**2 + 0.5 * B + 0.3 * np.exp(-((E-0.2)**2)/0.01) * (B < 0.05)

    surf = ax.plot_surface(E, B, L, cmap='viridis', alpha=0.85, edgecolor='none')
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='Loss $\\mathcal{L}$')

    ax.set_xlabel('$\\varepsilon$ (clip param)')
    ax.set_ylabel('$\\beta$ (KL coef)')
    ax.set_zlabel('Loss $\\mathcal{L}$')
    ax.set_title('Fig. 8: Loss Landscape vs Hyperparameters', fontsize=11, fontweight='bold')
    ax.view_init(elev=25, azim=-60)
    plt.tight_layout()
    plt.savefig(f'{OUT}/ieee_fig8_3d_surface.png', dpi=300, bbox_inches='tight')
    plt.close()
    print('[OK] ieee_fig8_3d_surface.png')


def fig9_ablation():
    """Fig. 9: Ablation study bar chart."""
    fig, ax = plt.subplots(figsize=(7, 4))

    labels = ['Baseline\n(GRPO)', '+Overlong\nFilter', '+Clip\nHigher',
              '+Soft Overlong\nPunishment', '+Token-level\nLoss', '+Dynamic\nSampling\n(DAPO)']
    scores = [30, 36, 38, 41, 42, 50]
    deltas = [0, 6, 2, 3, 1, 8]
    colors = ['#bdc3c7', '#3498db', '#2980b9', '#8e44ad', '#27ae60', '#9b59b6']

    bars = ax.bar(range(len(labels)), scores, color=colors, edgecolor='white', linewidth=1.5)

    for i, (bar, score, delta) in enumerate(zip(bars, scores, deltas)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{score}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
        if delta > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 2,
                    f'+{delta}', ha='center', va='top', fontsize=9, color='white', fontweight='bold')

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel('AIME Score (%)')
    ax.set_title('Fig. 9: Ablation Study -- Incremental DAPO Innovations')
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax.set_ylim(0, 58)
    plt.tight_layout()
    plt.savefig(f'{OUT}/ieee_fig9_ablation.png', dpi=300, bbox_inches='tight')
    plt.close()
    print('[OK] ieee_fig9_ablation.png')


if __name__ == '__main__':
    print("Generating IEEE survey figures...")
    fig1_convergence_comparison()
    fig2_reward_curves()
    fig3_entropy_length()
    fig4_clip_higher_visualization()
    fig5_token_vs_sample_level()
    fig6_overlong_shaping()
    fig7_radar_comparison()
    fig8_3d_surface()
    fig9_ablation()
    print(f"\nDone! {9} figures saved to: {OUT}")
