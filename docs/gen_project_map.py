# -*- coding: utf-8 -*-
"""
Generate a project structure map image for the README.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['svg.fonttype'] = 'none'  # Keep text as text in SVG

fig, ax = plt.subplots(1, 1, figsize=(20, 12))
ax.set_xlim(0, 20)
ax.set_ylim(0, 12)
ax.axis('off')

# Colors
C_ROOT = '#1E40AF'
C_ALGO = '#047857'
C_DOCS = '#6D28D9'
C_DATA = '#B45309'
C_SRC  = '#B91C1C'
C_FIG  = '#0E7490'

def draw_box(ax, x, y, w, h, text, color, fontsize=14, text_color='white'):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                          facecolor=color, edgecolor='white', linewidth=2, alpha=0.92)
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center',
            fontsize=fontsize, color=text_color, fontweight='bold')

# Title
ax.text(10, 11.4, 'Project Structure Map', ha='center', va='center',
        fontsize=28, fontweight='bold', color='#1F2937')
ax.text(10, 10.7, 'PPO / DPO / GRPO / DAPO  —  RL Algorithms for LLM Training', ha='center', va='center',
        fontsize=16, color='#6B7280')

# Root box
draw_box(ax, 7.5, 9.5, 5, 0.9, 'PPOvDPOvGRPOvDAPO/\n(Project Root)', C_ROOT, fontsize=16)

# Level 1 boxes - wider and taller
draw_box(ax, 0.5, 7.2, 4.2, 0.9, 'algorithms/\n4 Algorithm Trainers', C_ALGO, fontsize=14)
draw_box(ax, 5.2, 7.2, 4.5, 0.9, 'docs/\nPapers, Docs, Figures', C_DOCS, fontsize=14)
draw_box(ax, 10.2, 7.2, 3.5, 0.9, 'data/\nTraining Data', C_DATA, fontsize=14)
draw_box(ax, 14.2, 7.2, 3.5, 0.9, 'src/\nSource Code', C_SRC, fontsize=14)

# Root -> Level 1 lines
for x_end in [2.6, 7.45, 11.95, 15.95]:
    ax.plot([10, x_end], [9.5, 8.1], color='#9CA3AF', linewidth=1.5, zorder=0)

# Algorithms sub-items
algo_items = [
    (0.5, 5.9, 'ppo_trainer.py'),
    (0.5, 5.1, 'dpo_trainer.py'),
    (0.5, 4.3, 'grpo_trainer.py'),
    (0.5, 3.5, 'dapo_trainer.py'),
]
algo_colors = ['#10B981', '#34D399', '#6EE7B7', '#A7F3D0']
algo_tc =     ['white',   'white',   '#065F46', '#065F46']
for i, (x, y, text) in enumerate(algo_items):
    draw_box(ax, x, y, 4.2, 0.6, text, algo_colors[i], fontsize=13, text_color=algo_tc[i])
ax.plot([2.6, 2.6], [7.2, 6.5], color='#9CA3AF', linewidth=1.5)

# Docs sub-items
docs_items = [
    (5.2, 5.9, 2.1, 0.6, 'ieee_en/\nEN LaTeX', '#7C3AED'),
    (7.5, 5.9, 2.2, 0.6, 'ieee_cn/\nCN LaTeX', '#8B5CF6'),
    (5.2, 5.1, 4.5, 0.6, 'figures/ (40+ images)', C_FIG),
    (5.2, 4.3, 2.1, 0.6, '*.pdf\nEN + CN', '#F59E0B'),
    (7.5, 4.3, 2.2, 0.6, '*.md\n4 Algo Docs', '#FBBF24'),
]
for x, y, w, h, text, color in docs_items:
    tc = 'white' if color not in ['#FBBF24', '#F59E0B', '#8B5CF6'] else '#1F2937'
    draw_box(ax, x, y, w, h, text, color, fontsize=11, text_color=tc)
ax.plot([7.45, 7.45], [7.2, 6.5], color='#9CA3AF', linewidth=1.5)

# Data sub-items
draw_box(ax, 10.2, 5.9, 3.5, 0.6, 'sample_reasoning_data.json', C_DATA, fontsize=12)
draw_box(ax, 10.2, 5.1, 3.5, 0.6, 'preference_pairs.json', '#FCD34D', fontsize=12, text_color='#1F2937')
ax.plot([11.95, 11.95], [7.2, 6.5], color='#9CA3AF', linewidth=1.5)

# Src sub-items
for y, text in [(5.9, 'models/'), (5.1, 'training/'), (4.3, 'utils/')]:
    draw_box(ax, 14.2, y, 3.5, 0.6, text, '#F87171', fontsize=13)
ax.plot([15.95, 15.95], [7.2, 6.5], color='#9CA3AF', linewidth=1.5)

# Bottom: run files
draw_box(ax, 4.5, 2.2, 5, 0.6, 'run_comparison.py', '#374151', fontsize=14)
draw_box(ax, 10.5, 2.2, 5, 0.6, 'requirements.txt', '#374151', fontsize=14)
ax.plot([10, 7], [9.5, 2.8], color='#9CA3AF', linewidth=1, linestyle='--', zorder=0)
ax.plot([10, 13], [9.5, 2.8], color='#9CA3AF', linewidth=1, linestyle='--', zorder=0)

# Legend
legend_items = [
    (2, 1.0, C_ALGO, 'Algorithm Implementations'),
    (6.5, 1.0, C_DOCS, 'Documentation & Papers'),
    (11, 1.0, C_DATA, 'Training Data'),
    (15, 1.0, C_SRC, 'Source Code'),
]
for x, y, color, label in legend_items:
    ax.add_patch(FancyBboxPatch((x, y), 0.4, 0.4, boxstyle="round,pad=0.03",
                                 facecolor=color, edgecolor='none', alpha=0.9))
    ax.text(x + 0.6, y + 0.2, label, ha='left', va='center', fontsize=14, color='#374151')

# Pipeline
ax.text(10, 0.2, 'Pipeline:  SFT  →  Reward Model  →  RL Optimization (PPO / DPO / GRPO / DAPO)',
        ha='center', va='center', fontsize=15, color='#6B7280', style='italic')

plt.tight_layout()
outpath_png = os.path.join(os.path.dirname(__file__), 'figures', 'project_map.png')
outpath_svg = os.path.join(os.path.dirname(__file__), 'figures', 'project_map.svg')
fig.savefig(outpath_png, dpi=150, bbox_inches='tight', facecolor='white')
fig.savefig(outpath_svg, bbox_inches='tight', facecolor='white')
print(f'[OK] project_map.png + .svg')
plt.close()
