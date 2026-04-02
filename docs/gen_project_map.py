# -*- coding: utf-8 -*-
"""
Generate a project structure map image for the README.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(1, 1, figsize=(16, 10))
ax.set_xlim(0, 16)
ax.set_ylim(0, 10)
ax.axis('off')

# Colors
C_ROOT = '#2563EB'
C_ALGO = '#059669'
C_DOCS = '#7C3AED'
C_DATA = '#D97706'
C_SRC  = '#DC2626'
C_FIG  = '#0891B2'

def draw_box(ax, x, y, w, h, text, color, fontsize=9, text_color='white'):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                          facecolor=color, edgecolor='white', linewidth=1.5, alpha=0.9)
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center',
            fontsize=fontsize, color=text_color, fontweight='bold', wrap=True)

# Title
ax.text(8, 9.6, 'Project Structure Map', ha='center', va='center',
        fontsize=18, fontweight='bold', color='#1F2937')
ax.text(8, 9.2, 'PPO / DPO / GRPO / DAPO — RL for LLM Training', ha='center', va='center',
        fontsize=11, color='#6B7280')

# Root box
draw_box(ax, 6.5, 8.2, 3, 0.7, 'deepseek_r1_qwen2-1.5b/\n(Project Root)', C_ROOT, fontsize=10)

# Level 1 boxes
draw_box(ax, 0.3, 6.2, 3, 0.7, 'algorithms/\n4 Algorithm Trainers', C_ALGO, fontsize=9)
draw_box(ax, 4.0, 6.2, 3.5, 0.7, 'docs/\nPapers, Docs, Figures', C_DOCS, fontsize=9)
draw_box(ax, 8.2, 6.2, 2.5, 0.7, 'data/\nTraining Data', C_DATA, fontsize=9)
draw_box(ax, 11.2, 6.2, 2.5, 0.7, 'src/\nSource Code', C_SRC, fontsize=9)

# Root -> Level 1 lines
for x_end in [1.8, 5.75, 9.45, 12.45]:
    ax.plot([8, x_end], [8.2, 6.9], color='#9CA3AF', linewidth=1.2, zorder=0)

# Algorithms sub-items
algo_items = [
    (0.3, 5.3, 'ppo_trainer.py'),
    (0.3, 4.7, 'dpo_trainer.py'),
    (0.3, 4.1, 'grpo_trainer.py'),
    (0.3, 3.5, 'dapo_trainer.py'),
]
for i, (x, y, text) in enumerate(algo_items):
    colors = ['#10B981', '#34D399', '#6EE7B7', '#A7F3D0']
    text_colors = ['white', 'white', '#065F46', '#065F46']
    draw_box(ax, x, y, 3, 0.45, text, colors[i], fontsize=8, text_color=text_colors[i])
ax.plot([1.8, 1.8], [6.2, 5.75], color='#9CA3AF', linewidth=1)

# Docs sub-items
docs_items = [
    (4.0, 5.3, 1.6, 0.45, 'ieee_en/\nEN LaTeX', '#8B5CF6'),
    (5.7, 5.3, 1.7, 0.45, 'ieee_cn/\nCN LaTeX', '#A78BFA'),
    (4.0, 4.7, 3.4, 0.45, 'figures/ (40+ images)', C_FIG),
    (4.0, 4.1, 1.6, 0.45, '*.pdf\nEN + CN', '#F59E0B'),
    (5.7, 4.1, 1.7, 0.45, '*.md\n4 Algo Docs', '#FBBF24'),
]
for x, y, w, h, text, color in docs_items:
    tc = 'white' if color not in ['#FBBF24', '#F59E0B', '#A78BFA'] else '#1F2937'
    draw_box(ax, x, y, w, h, text, color, fontsize=7, text_color=tc)
ax.plot([5.75, 5.75], [6.2, 5.75], color='#9CA3AF', linewidth=1)

# Data sub-items
draw_box(ax, 8.2, 5.3, 2.5, 0.45, 'sample_reasoning_data.json', C_DATA, fontsize=7.5)
draw_box(ax, 8.2, 4.7, 2.5, 0.45, 'preference_pairs.json', '#FCD34D', fontsize=7.5, text_color='#1F2937')
ax.plot([9.45, 9.45], [6.2, 5.75], color='#9CA3AF', linewidth=1)

# Src sub-items
src_items = [
    (11.2, 5.3, 'models/'),
    (11.2, 4.7, 'training/'),
    (11.2, 4.1, 'utils/'),
]
for x, y, text in src_items:
    draw_box(ax, x, y, 2.5, 0.45, text, '#F87171', fontsize=8)
ax.plot([12.45, 12.45], [6.2, 5.75], color='#9CA3AF', linewidth=1)

# Bottom: run files
draw_box(ax, 3.0, 2.5, 4.5, 0.5, 'run_comparison.py', '#374151', fontsize=9)
draw_box(ax, 8.0, 2.5, 4.5, 0.5, 'requirements.txt', '#374151', fontsize=9)
ax.plot([8, 5.25], [8.2, 3.0], color='#9CA3AF', linewidth=0.8, linestyle='--', zorder=0)
ax.plot([8, 10.25], [8.2, 3.0], color='#9CA3AF', linewidth=0.8, linestyle='--', zorder=0)

# Legend at bottom
legend_items = [
    (1, 1.2, C_ALGO, 'Algorithm Implementations'),
    (4.5, 1.2, C_DOCS, 'Documentation & Papers'),
    (8, 1.2, C_DATA, 'Training Data'),
    (11, 1.2, C_SRC, 'Source Code'),
]
for x, y, color, label in legend_items:
    ax.add_patch(FancyBboxPatch((x, y), 0.3, 0.3, boxstyle="round,pad=0.02",
                                 facecolor=color, edgecolor='none', alpha=0.9))
    ax.text(x + 0.45, y + 0.15, label, ha='left', va='center', fontsize=9, color='#374151')

# Pipeline flow at very bottom
ax.text(8, 0.3, 'Pipeline: SFT → Reward Model → RL Optimization (PPO / DPO / GRPO / DAPO)',
        ha='center', va='center', fontsize=10, color='#6B7280', style='italic')

plt.tight_layout()
outpath = os.path.join(os.path.dirname(__file__), 'figures', 'project_map.png')
fig.savefig(outpath, dpi=150, bbox_inches='tight', facecolor='white')
print(f'[OK] {outpath}')
plt.close()
