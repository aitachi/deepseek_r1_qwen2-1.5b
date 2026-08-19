# -*- coding: utf-8 -*-
"""
Generate additional images for 4-algorithm comparison document.
为4算法对比文档生成补充图片。
"""
import os, sys, warnings
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import matplotlib.patheffects as pe

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['svg.fonttype'] = 'none'  # Keep text as text in SVG

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(OUT, exist_ok=True)

BG = '#f5f7fa'

def save_dual(fig, name, dpi=180, facecolor=None):
    """Save figure as both PNG and SVG."""
    fig.savefig(f'{OUT}/{name}.png', dpi=dpi, bbox_inches='tight', facecolor=facecolor)
    fig.savefig(f'{OUT}/{name}.svg', bbox_inches='tight', facecolor=facecolor)
    print(f'[OK] {name}.png + .svg')


def algorithm_timeline():
    """Algorithm evolution timeline / 算法演进时间线"""
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.set_xlim(2016.5, 2026)
    ax.set_ylim(-0.5, 5.5)
    ax.axis('off')
    ax.set_facecolor(BG)
    fig.patch.set_facecolor(BG)
    ax.text(11.3, 5.2, 'RL for LLMs: Algorithm Evolution Timeline',
            ha='center', fontsize=18, fontweight='bold', color='#2c3e50')

    # Timeline axis
    ax.plot([2017, 2025.8], [2.5, 2.5], '-', color='#bdc3c7', lw=3, zorder=1)

    # Events
    events = [
        (2017.0, 3.5, 'PPO\n(Schulman et al.)', '#3498db',
         'Clipped surrogate\n+ Value network\n+ GAE'),
        (2019.0, 1.3, 'RLHF\n(OpenAI)', '#e67e22',
         'Reward model\n+ PPO fine-tune\n+ Human feedback'),
        (2023.0, 3.5, 'DPO\n(Rafailov et al.)', '#e74c3c',
         'Bradley-Terry\n+ No reward model\n+ Preference pairs'),
        (2025.0, 1.3, 'GRPO\n(DeepSeek-R1)', '#27ae60',
         'Group advantage\n+ No value net\n+ KL penalty'),
        (2025.5, 3.5, 'DAPO\n(ByteDance)', '#9b59b6',
         'Dynamic sampling\n+ Token-level loss\n+ Overlong filter'),
    ]

    for year, y, title, color, desc in events:
        # Dot on timeline
        ax.plot(year, 2.5, 'o', color=color, markersize=16, zorder=3)
        ax.text(year, 2.5, str(int(year)) if year == int(year) else f'{year:.1f}',
                ha='center', va='center', fontsize=7, color='white', fontweight='bold', zorder=4)

        # Box
        bw, bh = 2.0, 1.8
        bx = year - bw/2
        by = y - bh/2 if y > 2.5 else y - bh/2
        box = FancyBboxPatch((bx, by), bw, bh,
                             boxstyle="round,pad=0.1",
                             facecolor=color, edgecolor='white',
                             linewidth=1.5, alpha=0.9)
        ax.add_patch(box)
        ax.text(year, by + bh*0.7, title, ha='center', va='center',
                fontsize=11, fontweight='bold', color='white', linespacing=1.2)
        ax.text(year, by + bh*0.25, desc, ha='center', va='center',
                fontsize=7.5, color='white', alpha=0.9, linespacing=1.3)

        # Connector
        if y > 2.5:
            ax.annotate('', xy=(year, 2.5+0.15), xytext=(year, by),
                       arrowprops=dict(arrowstyle='->', color=color, lw=1.5))
        else:
            ax.annotate('', xy=(year, 2.5-0.15), xytext=(year, by+bh),
                       arrowprops=dict(arrowstyle='->', color=color, lw=1.5))

    # Arrows showing influence
    influences = [
        (2017.0, 2019.0, '#bdc3c7'),
        (2019.0, 2023.0, '#bdc3c7'),
        (2017.0, 2025.0, '#bdc3c7'),
        (2025.0, 2025.5, '#bdc3c7'),
    ]
    for x1, x2, c in influences:
        ax.annotate('', xy=(x2-0.2, 2.5), xytext=(x1+0.2, 2.5),
                   arrowprops=dict(arrowstyle='->', color=c, lw=1, linestyle='--'))

    plt.tight_layout()
    save_dual(plt.gcf(), 'algorithm_timeline', facecolor=BG)
    plt.close()


def algorithm_taxonomy():
    """Algorithm taxonomy/relationship diagram / 算法分类关系图"""
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 12)
    ax.axis('off')
    ax.set_facecolor(BG)
    fig.patch.set_facecolor(BG)
    ax.text(8, 11.5, 'RL Algorithms for LLM Training: Taxonomy',
            ha='center', fontsize=18, fontweight='bold', color='#2c3e50')

    def box(ax, cx, cy, w, h, txt, color, fs=11):
        b = FancyBboxPatch((cx-w/2, cy-h/2), w, h,
                          boxstyle="round,pad=0.05",
                          facecolor=color, edgecolor='white', lw=2, alpha=0.9)
        ax.add_patch(b)
        ax.text(cx, cy, txt, ha='center', va='center', fontsize=fs,
                fontweight='bold', color='white', linespacing=1.3)

    # Root
    box(ax, 8, 10.5, 4.5, 0.9, 'RL for LLM Training\nLLM 强化学习训练', '#2c3e50', 13)

    # Level 2: Two branches
    box(ax, 4, 8.8, 3.5, 0.85, 'Policy Gradient\n策略梯度方法', '#2980b9', 11)
    box(ax, 12, 8.8, 3.5, 0.85, 'Preference-based\n基于偏好方法', '#e74c3c', 11)

    ax.annotate('', xy=(4, 10.05), xytext=(6.5, 10.05),
               arrowprops=dict(arrowstyle='->', color='#7f8c8d', lw=2))
    ax.annotate('', xy=(12, 10.05), xytext=(9.5, 10.05),
               arrowprops=dict(arrowstyle='->', color='#7f8c8d', lw=2))

    # Level 3: PPO branch -> GRPO -> DAPO
    box(ax, 2.5, 7.0, 2.8, 0.85, 'PPO\nActor-Critic', '#3498db', 10)
    box(ax, 5.8, 7.0, 2.8, 0.85, 'GRPO\nGroup-based', '#27ae60', 10)
    box(ax, 9.2, 7.0, 2.8, 0.85, 'DAPO\nDynamic+Token', '#9b59b6', 10)

    ax.annotate('', xy=(2.5, 8.35), xytext=(3.5, 8.35),
               arrowprops=dict(arrowstyle='->', color='#3498db', lw=1.5))
    ax.annotate('', xy=(5.8, 8.35), xytext=(4.5, 8.35),
               arrowprops=dict(arrowstyle='->', color='#27ae60', lw=1.5))
    ax.annotate('', xy=(9.2, 8.35), xytext=(6.5, 8.35),
               arrowprops=dict(arrowstyle='->', color='#9b59b6', lw=1.5))

    # DPO branch
    box(ax, 12.5, 7.0, 2.8, 0.85, 'DPO\nBradley-Terry', '#e74c3c', 10)
    ax.annotate('', xy=(12.5, 8.35), xytext=(12, 8.35),
               arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=1.5))

    # Level 4: Key characteristics
    chars = [
        (2.5, 5.2, '#3498db', 'Value Network\nGAE Advantage\nEntropy Bonus\nMulti-epoch Update'),
        (5.8, 5.2, '#27ae60', 'No Value Network\nGroup Advantage\nKL Penalty\nFixed Group Size'),
        (9.2, 5.2, '#9b59b6', 'Dynamic Group\nOverlong Filter\nToken-level Loss\nReward Shaping'),
        (12.5, 5.2, '#e74c3c', 'No Reward Model\nPreference Pairs\nImplicit Reward\nSimple & Stable'),
    ]
    for cx, cy, color, desc in chars:
        rect = plt.Rectangle((cx-1.5, cy-1.1), 3, 2.2, fill=True,
                             facecolor=color, edgecolor=color, lw=1.5,
                             linestyle='--', alpha=0.1)
        ax.add_patch(rect)
        ax.text(cx, cy, desc, ha='center', va='center', fontsize=8.5,
                color=color, fontweight='bold', linespacing=1.5)
        ax.annotate('', xy=(cx, cy+1.1), xytext=(cx, 6.55),
                   arrowprops=dict(arrowstyle='->', color=color, lw=1.2, linestyle=':'))

    # Evolution arrows at bottom
    ax.annotate('', xy=(4.2, 7.0), xytext=(3.9, 7.0),
               arrowprops=dict(arrowstyle='->', color='#95a5a6', lw=2.5))
    ax.text(4.0, 7.5, 'evolves', ha='center', fontsize=8, color='#95a5a6', fontstyle='italic')

    ax.annotate('', xy=(7.5, 7.0), xytext=(7.2, 7.0),
               arrowprops=dict(arrowstyle='->', color='#95a5a6', lw=2.5))
    ax.text(7.4, 7.5, 'extends', ha='center', fontsize=8, color='#95a5a6', fontstyle='italic')

    # Memory efficiency scale
    box(ax, 3, 2.5, 2.5, 0.7, 'Memory: High', '#e74c3c', 9)
    box(ax, 7, 2.5, 2.5, 0.7, 'Memory: Medium', '#f39c12', 9)
    box(ax, 11, 2.5, 2.5, 0.7, 'Memory: Low', '#27ae60', 9)
    ax.text(8, 1.5, 'PPO (High) -> GRPO/DAPO (Medium) -> DPO (Low)',
            ha='center', fontsize=9, color='#7f8c8d')

    plt.tight_layout()
    save_dual(plt.gcf(), 'algorithm_taxonomy', facecolor=BG)
    plt.close()


def resource_heatmap():
    """Resource comparison heatmap / 资源消耗对比热力图"""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_facecolor(BG)
    fig.patch.set_facecolor(BG)

    metrics = ['GPU Memory', 'Training Time', 'Sample Efficiency',
               'Implementation Complexity', 'Training Stability',
               'Final Performance', 'Memory per Sample', 'Scalability']
    algorithms = ['PPO', 'DPO', 'GRPO', 'DAPO']

    # Scores (1-10, higher = better for performance metrics, lower = better for cost)
    data = np.array([
        [2, 8, 7, 8, 6, 7, 3, 7],   # PPO: high memory, good perf
        [8, 9, 6, 9, 9, 6, 8, 6],   # DPO: low memory, very stable
        [6, 7, 9, 7, 8, 9, 6, 8],   # GRPO: balanced, high perf
        [5, 6, 8, 6, 7, 10, 5, 9],  # DAPO: best perf, good efficiency
    ])

    im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=1, vmax=10)

    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(metrics, rotation=30, ha='right', fontsize=10)
    ax.set_yticks(range(len(algorithms)))
    ax.set_yticklabels(algorithms, fontsize=12, fontweight='bold')

    for i in range(len(algorithms)):
        for j in range(len(metrics)):
            val = data[i, j]
            color = 'white' if val < 4 or val > 7 else 'black'
            ax.text(j, i, f'{val}', ha='center', va='center',
                    fontsize=12, fontweight='bold', color=color)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Score (1=Poor, 10=Excellent)', fontsize=10)

    ax.set_title('Algorithm Resource & Performance Comparison Heatmap\n'
                 '算法资源与性能对比热力图',
                 fontsize=14, fontweight='bold', color='#2c3e50', pad=15)

    plt.tight_layout()
    save_dual(plt.gcf(), 'resource_heatmap', facecolor=BG)
    plt.close()


def loss_function_comparison():
    """Loss function formula comparison / 损失函数公式对比图"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.patch.set_facecolor(BG)

    colors = ['#3498db', '#e74c3c', '#27ae60', '#9b59b6']
    titles = ['PPO Loss', 'DPO Loss', 'GRPO Loss', 'DAPO Loss']
    formulas = [
        r'$L^{PPO}(\theta) = -\mathbb{E}_t\left[\min\left(r_t(\theta)\hat{A}_t,\ \mathrm{clip}(r_t, 1-\varepsilon, 1+\varepsilon)\hat{A}_t\right)\right] + c_1 L^{VF} - c_2 S[\pi_\theta]$',
        r'$L^{DPO}(\theta) = -\mathbb{E}_{(x,y_w,y_l)}\left[\log\sigma\left(\beta\left(\log\frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \log\frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\right)\right)\right]$',
        r'$L^{GRPO}(\theta) = -\frac{1}{G}\sum_{i=1}^{G}\min\left(\frac{\pi_\theta(o_i|q)}{\pi_{old}(o_i|q)}\hat{A}_i,\ \mathrm{clip}\left(\frac{\pi_\theta}{\pi_{old}}, 1-\varepsilon, 1+\varepsilon\right)\hat{A}_i\right) + \beta D_{KL}$',
        r'$L^{DAPO}(\theta) = -\frac{1}{\sum |o_i|}\sum_{i:valid}\frac{\mathbb{1}[|o_i|\leq L_{max}]}{|o_i|}\sum_t\min\left(r_t\hat{A}_i,\ \mathrm{clip}(r_t)\hat{A}_i\right) + \beta D_{KL}^{token}$',
    ]
    descriptions = [
        'Components: Clipped Surrogate + Value Loss + Entropy Bonus\nRequires: Policy Network + Value Network + Reward',
        'Components: Bradley-Terry Preference Loss\nRequires: Policy Network + Reference Network\nKey: No explicit reward model needed',
        'Components: Group Clipped Surrogate + KL Penalty\nRequires: Policy Network + Reference Network\nKey: Group advantage replaces value network',
        'Components: Token-level Clipped + Token KL + Dynamic G\nRequires: Policy Network + Reference Network\nKey: Per-token normalization, dynamic sampling',
    ]

    for idx, (ax, color, title, formula, desc) in enumerate(
            zip(axes.flat, colors, titles, formulas, descriptions)):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.set_facecolor(BG)
        ax.axis('off')

        # Background box
        rect = plt.Rectangle((0.3, 0.3), 9.4, 9.4, fill=True,
                             facecolor=color, edgecolor=color, lw=3,
                             alpha=0.08, zorder=0)
        ax.add_patch(rect)
        border = plt.Rectangle((0.3, 0.3), 9.4, 9.4, fill=False,
                               edgecolor=color, lw=2.5, alpha=0.6, zorder=1)
        ax.add_patch(border)

        # Title
        ax.text(5, 9.0, title, ha='center', va='center',
                fontsize=16, fontweight='bold', color=color)

        # Formula
        ax.text(5, 5.5, formula, ha='center', va='center',
                fontsize=11, color='#2c3e50',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                         edgecolor=color, alpha=0.9))

        # Description
        ax.text(5, 2.0, desc, ha='center', va='center',
                fontsize=9, color='#555', linespacing=1.6)

    fig.suptitle('Loss Function Comparison: PPO vs DPO vs GRPO vs DAPO\n'
                 '损失函数对比', fontsize=16, fontweight='bold',
                 color='#2c3e50', y=1.02)
    plt.tight_layout()
    save_dual(plt.gcf(), 'loss_function_comparison', facecolor=BG)
    plt.close()


def convergence_comparison_4():
    """4-algorithm convergence comparison / 4算法收敛对比"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.patch.set_facecolor(BG)

    np.random.seed(42)
    steps = np.arange(0, 100)

    configs = [
        ('PPO', '#3498db', 2.5, 0.35, 0.08),
        ('DPO', '#e74c3c', 2.2, 0.30, 0.10),
        ('GRPO', '#27ae60', 2.0, 0.25, 0.08),
        ('DAPO', '#9b59b6', 1.8, 0.20, 0.06),
    ]

    for ax, (name, color, init_loss, decay, noise) in zip(
            axes.flat, configs):
        ax.set_facecolor('white')
        loss = init_loss * np.exp(-decay * steps) + noise * np.random.randn(len(steps))
        loss = np.maximum(loss, 0.02)
        ax.plot(steps, loss, '-', color=color, lw=2, label=f'{name} Loss')
        ax.fill_between(steps, loss - noise, loss + noise, alpha=0.15, color=color)
        ax.set_title(f'{name} Training Loss', fontsize=13, fontweight='bold', color=color)
        ax.set_xlabel('Training Step', fontsize=10)
        ax.set_ylabel('Loss', fontsize=10)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, init_loss * 1.2)

    fig.suptitle('Training Loss Convergence Comparison\n训练损失收敛对比',
                 fontsize=15, fontweight='bold', color='#2c3e50')
    plt.tight_layout()
    save_dual(plt.gcf(), 'convergence_4algo', facecolor=BG)
    plt.close()


def performance_radar_detailed():
    """Detailed radar chart / 详细雷达图"""
    categories = ['GPU Memory\nEfficiency', 'Training\nSpeed', 'Sample\nEfficiency',
                  'Implementation\nSimplicity', 'Training\nStability',
                  'Final\nPerformance', 'Scalability', 'Adaptability']

    N = len(categories)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    data = {
        'PPO':   [3, 4, 6, 4, 6, 7, 7, 5],
        'DPO':   [8, 8, 6, 9, 9, 6, 6, 5],
        'GRPO':  [7, 7, 9, 7, 8, 9, 8, 7],
        'DAPO':  [6, 6, 8, 5, 7, 10, 9, 9],
    }
    colors = ['#3498db', '#e74c3c', '#27ae60', '#9b59b6']

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor('white')

    for (name, vals), color in zip(data.items(), colors):
        values = vals + vals[:1]
        ax.plot(angles, values, 'o-', linewidth=2.5, label=name, color=color, markersize=6)
        ax.fill(angles, values, alpha=0.1, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=9)
    ax.set_ylim(0, 10)
    ax.set_yticks([2, 4, 6, 8, 10])
    ax.set_yticklabels(['2', '4', '6', '8', '10'], fontsize=8)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)
    ax.set_title('Algorithm Performance Radar (8 Dimensions)\n'
                 '算法性能雷达图 (8维度)', fontsize=14, fontweight='bold',
                 color='#2c3e50', pad=20)

    plt.tight_layout()
    save_dual(plt.gcf(), 'radar_8dim', facecolor=BG)
    plt.close()


if __name__ == '__main__':
    print("Generating additional comparison images...")
    algorithm_timeline()
    algorithm_taxonomy()
    resource_heatmap()
    loss_function_comparison()
    convergence_comparison_4()
    performance_radar_detailed()
    print(f"\nDone! Saved to: {OUT}")
