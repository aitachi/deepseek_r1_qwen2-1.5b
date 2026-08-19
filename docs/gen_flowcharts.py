# -*- coding: utf-8 -*-
"""
Regenerate 4 algorithm flowcharts with multi-branch vector style.
重新生成4张多分支矢量风格流程图: 小框、大字、并行分支、反馈环路。
"""
import os, sys, warnings
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Arc
import matplotlib.patheffects as pe

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['svg.fonttype'] = 'none'  # Keep text as text in SVG

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(OUT, exist_ok=True)

def save_dual(fig, name, dpi=180, facecolor=None):
    """Save figure as both PNG and SVG."""
    fig.savefig(f'{OUT}/{name}.png', dpi=dpi, bbox_inches='tight', facecolor=facecolor)
    fig.savefig(f'{OUT}/{name}.svg', bbox_inches='tight', facecolor=facecolor)
    print(f'[OK] {name}.png + .svg')

BG = '#f5f7fa'
C = {
    'input': '#34495e', 'gen': '#2980b9', 'reward': '#e67e22',
    'value': '#8e44ad', 'advantage': '#16a085', 'loss': '#c0392b',
    'update': '#2c3e50', 'done': '#7f8c8d', 'grpo': '#27ae60',
    'dapo': '#9b59b6', 'filter': '#d35400', 'dpo': '#e74c3c',
    'ref': '#1abc9c', 'fork': '#f39c12',
}

def _box(ax, cx, cy, w, h, txt, color, fs=11):
    """Draw compact box with large text."""
    b = FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                       boxstyle="round,pad=0.03",
                       facecolor=color, edgecolor='white', linewidth=1.2, alpha=0.92)
    ax.add_patch(b)
    ax.text(cx, cy, txt, ha='center', va='center', fontsize=fs,
            fontweight='bold', color='white', linespacing=1.3,
            path_effects=[pe.withStroke(linewidth=0, foreground='black')])

def _arrow(ax, x1, y1, x2, y2, color='#555', lw=1.8, style='->', curved=0):
    """Draw arrow with optional curve."""
    conn = f'arc3,rad={curved}' if curved else 'arc3,rad=0'
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw,
                                connectionstyle=conn))

def _label(ax, x, y, txt, fs=9, color='#7f8c8d', **kw):
    ax.text(x, y, txt, ha='center', va='center', fontsize=fs,
            color=color, fontstyle='italic', **kw)

def _dashed_rect(ax, x, y, w, h, color, label='', label_pos='top-left'):
    """Draw dashed region box with label."""
    rect = plt.Rectangle((x, y), w, h, fill=True, facecolor=color,
                          edgecolor=color, linewidth=1.5, linestyle='--', alpha=0.08)
    ax.add_patch(rect)
    rect2 = plt.Rectangle((x, y), w, h, fill=False,
                           edgecolor=color, linewidth=1.8, linestyle='--', alpha=0.5)
    ax.add_patch(rect2)
    if label:
        lx, ly = x + 0.15, y + h - 0.15
        ax.text(lx, ly, label, fontsize=8, color=color, fontweight='bold',
                va='top', fontstyle='italic', alpha=0.8)


# ================================================================
# 1. PPO Flowchart - 双网络并行架构
# ================================================================
def ppo_flowchart():
    fig, ax = plt.subplots(figsize=(16, 18))
    ax.set_xlim(0, 16); ax.set_ylim(0, 18); ax.axis('off')
    ax.set_facecolor(BG); fig.patch.set_facecolor(BG)
    ax.text(8, 17.5, 'PPO Algorithm Flow / PPO 算法流程',
            ha='center', fontsize=18, fontweight='bold', color='#2c3e50')

    W, H, FS = 3.6, 0.85, 10.5  # box width, height, font

    # Row 1: Input
    _box(ax, 8, 16.5, 4, H, 'Input Question q\n输入问题 q', C['input'], 12)

    # Row 2: Fork into 3 parallel paths
    _arrow(ax, 8, 16.5-H/2, 4, 15.3+H/2)    # -> policy
    _arrow(ax, 8, 16.5-H/2, 12, 15.3+H/2)    # -> value
    _label(ax, 2.2, 15.9, 'Actor', color=C['gen'], fs=10)
    _label(ax, 14, 15.9, 'Critic', color=C['value'], fs=10)

    # Row 3: Three parallel paths
    _box(ax, 4, 15.3, W, H, 'Policy Net\nGenerate o~pi', C['gen'], FS)
    _box(ax, 12, 15.3, W, H, 'Value Net\nV_phi(s)', C['value'], FS)

    # Row 4: Reward + GAE (still parallel feeds)
    _arrow(ax, 4, 15.3-H/2, 4, 14.1+H/2)
    _arrow(ax, 12, 15.3-H/2, 12, 14.1+H/2)

    _box(ax, 4, 14.1, W, H, 'Reward\nr = R(q, o)', C['reward'], FS)
    _box(ax, 12, 14.1, W, H, 'GAE Advantage\nA_t = sum(g*l)^d', C['advantage'], FS)

    # Row 5: Merge - two feeds into clipped loss
    _arrow(ax, 4, 14.1-H/2, 6.5, 12.9+H/2)
    _arrow(ax, 12, 14.1-H/2, 9.5, 12.9+H/2)

    # Dashed region for PPO Core
    _dashed_rect(ax, 1.5, 12.3, 13, 4.2, '#3498db', 'PPO Core (parallel)')

    # Row 6: Clipped Surrogate (merge point)
    _box(ax, 8, 12.9, 5.5, H*1.2, 'Clipped Surrogate\nL = min(r*A, clip(r,1-e,1+e)*A)', C['loss'], FS)

    # Row 7: Three parallel loss branches
    _arrow(ax, 6.5, 12.9-H*0.6, 4, 11.4+H/2)   # -> policy loss
    _arrow(ax, 8, 12.9-H*0.6, 8, 11.4+H/2)      # -> value loss
    _arrow(ax, 9.5, 12.9-H*0.6, 12, 11.4+H/2)   # -> entropy

    _box(ax, 4, 11.4, 3.2, H, 'Policy Loss\nL_CLIP', '#3498db', FS-1)
    _box(ax, 8, 11.4, 3.2, H, 'Value Loss\nL_VF', '#8e44ad', FS-1)
    _box(ax, 12, 11.4, 3.2, H, 'Entropy Bonus\nS[pi]', '#1abc9c', FS-1)

    # Row 8: Merge total loss
    _arrow(ax, 4, 11.4-H/2, 6.5, 10.1+H/2)
    _arrow(ax, 8, 11.4-H/2, 8, 10.1+H/2)
    _arrow(ax, 12, 11.4-H/2, 9.5, 10.1+H/2)

    _dashed_rect(ax, 1.5, 9.7, 13, 2.3, '#c0392b', 'Loss Components (parallel)')

    _box(ax, 8, 10.1, 5.5, H, 'Total Loss = L_CLIP - c1*L_VF + c2*S', C['loss'], FS-0.5)

    # Row 9: Update
    _arrow(ax, 8, 10.1-H/2, 8, 8.9+H/2)
    _box(ax, 8, 8.9, 4, H, 'Update theta & phi', C['update'], 12)

    # Row 10: Loop
    _arrow(ax, 8, 8.9-H/2, 8, 7.7+H/2)
    _box(ax, 8, 7.7, 3.5, H*0.8, 'Next Question', C['done'], 10)

    # Feedback loop
    _arrow(ax, 14.5, 16.5, 14.5, 7.9, color='#95a5a6', lw=2.5, curved=0)
    _arrow(ax, 14.5, 7.9, 10, 7.7, color='#95a5a6', lw=2.5)
    _label(ax, 15.2, 12.2, 'Loop\n循环', color='#95a5a6', fs=11)

    plt.tight_layout()
    save_dual(plt.gcf(), 'ppo_flowchart', facecolor=BG)
    plt.close()


# ================================================================
# 2. DPO Flowchart - 双分支偏好对比架构
# ================================================================
def dpo_flowchart():
    fig, ax = plt.subplots(figsize=(16, 18))
    ax.set_xlim(0, 16); ax.set_ylim(0, 18); ax.axis('off')
    ax.set_facecolor(BG); fig.patch.set_facecolor(BG)
    ax.text(8, 17.5, 'DPO Algorithm Flow / DPO 算法流程',
            ha='center', fontsize=18, fontweight='bold', color='#2c3e50')

    W, H, FS = 3.4, 0.85, 10.5

    # Row 1: Input
    _box(ax, 8, 16.5, 4, H, 'Input Question q\n输入问题 q', C['input'], 12)

    # Row 2: Fork into dual generation
    _arrow(ax, 8, 16.5-H/2, 5, 15.3+H/2)
    _arrow(ax, 8, 16.5-H/2, 11, 15.3+H/2)
    _label(ax, 3, 15.85, 'Preferred Path', color='#27ae60', fs=10)
    _label(ax, 13.2, 15.85, 'Rejected Path', color='#c0392b', fs=10)

    # Row 3: Dual generation (parallel)
    _box(ax, 5, 15.3, W, H, 'Generate o1\n生成响应1', C['dpo'], FS)
    _box(ax, 11, 15.3, W, H, 'Generate o2\n生成响应2', C['dpo'], FS)

    # Row 4: Dual reward (parallel)
    _arrow(ax, 5, 15.3-H/2, 5, 14.1+H/2)
    _arrow(ax, 11, 15.3-H/2, 11, 14.1+H/2)

    _box(ax, 5, 14.1, W, H, 'Reward r1\n奖励 r1', C['reward'], FS)
    _box(ax, 11, 14.1, W, H, 'Reward r2\n奖励 r2', C['reward'], FS)

    # Dashed region for parallel generation
    _dashed_rect(ax, 2.5, 13.5, 11, 3.1, '#e74c3c', 'Dual Path (parallel)')

    # Row 5: Merge -> Preference
    _arrow(ax, 5, 14.1-H/2, 8, 12.6+H/2)
    _arrow(ax, 11, 14.1-H/2, 8, 12.6+H/2)

    _box(ax, 8, 12.6, 5.5, H*1.1, 'Create Pair\ny_w=argmax(r1,r2)\ny_l=argmin(r1,r2)', C['fork'], FS)

    # Row 6: Fork again -> dual log ratio (parallel)
    _arrow(ax, 6.5, 12.6-H*0.55, 5, 11.2+H/2)
    _arrow(ax, 9.5, 12.6-H*0.55, 11, 11.2+H/2)

    _box(ax, 5, 11.2, W+0.5, H, 'log(pi(y_w)/ref(y_w))\nLog Ratio Preferred', '#2980b9', FS-0.5)
    _box(ax, 11, 11.2, W+0.5, H, 'log(pi(y_l)/ref(y_l))\nLog Ratio Rejected', '#2980b9', FS-0.5)

    # Row 7: Ref model (side input)
    _box(ax, 15, 13.0, 1.8, 3.5, 'Reference\nModel\npi_ref\n(冻结)', C['ref'], 8)
    _arrow(ax, 14.1, 13.0, 12.8, 11.2, color=C['ref'], lw=1.2, style='->', curved=0.15)
    _arrow(ax, 14.1, 13.0, 6.8, 11.2, color=C['ref'], lw=1.2, style='->', curved=-0.15)

    _dashed_rect(ax, 2.8, 10.6, 11.4, 1.5, '#2980b9', 'Log Ratios (parallel)')

    # Row 8: Merge -> Bradley-Terry Loss
    _arrow(ax, 5, 11.2-H/2, 8, 9.7+H/2)
    _arrow(ax, 11, 11.2-H/2, 8, 9.7+H/2)

    _box(ax, 8, 9.7, 6, H*1.2, 'Bradley-Terry Loss\nL = -log sigma(beta*(log_rw - log_rl))', C['loss'], FS)

    # Row 9: Update
    _arrow(ax, 8, 9.7-H*0.6, 8, 8.3+H/2)
    _box(ax, 8, 8.3, 4, H, 'Update pi_theta', C['update'], 12)

    # Row 10: Next
    _arrow(ax, 8, 8.3-H/2, 8, 7.2+H/2)
    _box(ax, 8, 7.2, 3.5, H*0.8, 'Next Question', C['done'], 10)

    # Loop
    _arrow(ax, 14.5, 16.5, 14.5, 7.4, color='#95a5a6', lw=2.5)
    _arrow(ax, 14.5, 7.4, 9.8, 7.2, color='#95a5a6', lw=2.5)
    _label(ax, 15.2, 12, 'Loop\n循环', color='#95a5a6', fs=11)

    plt.tight_layout()
    save_dual(plt.gcf(), 'dpo_flowchart', facecolor=BG)
    plt.close()


# ================================================================
# 3. GRPO Flowchart - 组并行 + 双模型架构
# ================================================================
def grpo_flowchart():
    fig, ax = plt.subplots(figsize=(16, 18))
    ax.set_xlim(0, 16); ax.set_ylim(0, 18); ax.axis('off')
    ax.set_facecolor(BG); fig.patch.set_facecolor(BG)
    ax.text(8, 17.5, 'GRPO Algorithm Flow / GRPO 算法流程',
            ha='center', fontsize=18, fontweight='bold', color='#2c3e50')

    W, H, FS = 3.2, 0.85, 10.5

    # Row 1: Input
    _box(ax, 8, 16.5, 4, H, 'Input Question q\n输入问题 q', C['input'], 12)

    # Row 2: Fork -> G parallel generations
    _arrow(ax, 8, 16.5-H/2, 4, 15.3+H/2)
    _arrow(ax, 8, 16.5-H/2, 12, 15.3+H/2)

    _box(ax, 4, 15.3, W, H, 'Policy pi_theta\nGenerate G Responses', C['grpo'], FS)
    _box(ax, 12, 15.3, W, H, 'Reference pi_ref\n(冻结)', C['ref'], FS)

    # Row 3: Show G parallel branches (fan-out visualization)
    _arrow(ax, 4, 15.3-H/2, 2.5, 13.9+H/2)
    _arrow(ax, 4, 15.3-H/2, 5.5, 13.9+H/2)
    _label(ax, 4, 14.55, '......', fs=14, color='#27ae60')
    _arrow(ax, 4, 15.3-H/2, 4, 14.0+H/2, color='#27ae60', lw=1)

    # G parallel reward boxes
    g_xs = [2, 4, 6]
    for gx in g_xs:
        _box(ax, gx, 13.9, 1.8, H*0.9, f'r_i = R(q,o_i)', C['reward'], FS-2)

    _dashed_rect(ax, 0.8, 13.3, 6.4, 1.3, '#e67e22', 'G Parallel Rewards')

    # Row 4: Group Advantage (merge)
    _arrow(ax, 2, 13.9-H*0.45, 4, 12.4+H/2)
    _arrow(ax, 4, 13.9-H*0.45, 4, 12.4+H/2)
    _arrow(ax, 6, 13.9-H*0.45, 4, 12.4+H/2)

    _box(ax, 4, 12.4, 3.8, H, 'Group Advantage\nA=(r-mean)/std', C['advantage'], FS-0.5)

    # Row 5: Ratio computation (parallel: policy + ref)
    _arrow(ax, 4, 12.4-H/2, 8, 11.2+H/2)
    _arrow(ax, 12, 15.3-H/2, 12, 11.2+H/2)
    # ref model feeds down

    _box(ax, 8, 11.2, W+0.5, H, 'Ratio r_t\npi_theta/pi_old', '#2980b9', FS)
    _box(ax, 12, 11.2, W, H, 'KL Divergence\npi_ref/pi_theta', C['ref'], FS)

    _dashed_rect(ax, 1, 10.6, 14, 2.5, '#27ae60', 'GRPO Core')

    # Row 6: Merge -> Clipped + KL
    _arrow(ax, 4, 12.4-H/2, 4, 9.8+H/2)
    _arrow(ax, 8, 11.2-H/2, 8, 9.8+H/2)
    _arrow(ax, 12, 11.2-H/2, 8, 9.8+H/2)

    _box(ax, 4, 9.8, 3.4, H, 'Avg over G\nG样本平均', '#8e44ad', FS-0.5)
    _box(ax, 8, 9.8, 4.5, H*1.1, 'Clipped + KL\nL = min(r*A) + beta*KL', C['loss'], FS-0.5)

    _arrow(ax, 4, 9.8-H/2, 8, 9.8-H*0.55)

    # Row 7: Update
    _arrow(ax, 8, 9.8-H*0.55, 8, 8.5+H/2)
    _box(ax, 8, 8.5, 4, H, 'Update pi_theta', C['update'], 12)

    _arrow(ax, 8, 8.5-H/2, 8, 7.3+H/2)
    _box(ax, 8, 7.3, 3.5, H*0.8, 'Next Question', C['done'], 10)

    # Loop
    _arrow(ax, 14.5, 16.5, 14.5, 7.5, color='#95a5a6', lw=2.5)
    _arrow(ax, 14.5, 7.5, 9.8, 7.3, color='#95a5a6', lw=2.5)
    _label(ax, 15.2, 12, 'Loop\n循环', color='#95a5a6', fs=11)

    plt.tight_layout()
    save_dual(plt.gcf(), 'grpo_flowchart', facecolor=BG)
    plt.close()


# ================================================================
# 4. DAPO Flowchart - 动态组 + 过滤 + Token级 + 反馈环路
# ================================================================
def dapo_flowchart():
    fig, ax = plt.subplots(figsize=(16, 20))
    ax.set_xlim(0, 16); ax.set_ylim(0, 20); ax.axis('off')
    ax.set_facecolor(BG); fig.patch.set_facecolor(BG)
    ax.text(8, 19.5, 'DAPO Algorithm Flow / DAPO 算法流程',
            ha='center', fontsize=18, fontweight='bold', color='#2c3e50')

    W, H, FS = 3.4, 0.85, 10.5

    # Row 1: Input + Dynamic G
    _box(ax, 5, 18.5, 4, H, 'Input Question q\n输入问题 q', C['input'], 12)
    _box(ax, 12, 18.5, 3.2, H, 'Dynamic G\n动态组大小', C['dapo'], FS)
    _arrow(ax, 12, 18.5-H/2, 12, 17.3+H/2)
    _dashed_rect(ax, 10, 17.8, 4, 1.6, '#9b59b6', 'Innovation 1')

    # Row 2: Dynamic sampling -> G parallel
    _arrow(ax, 5, 18.5-H/2, 5, 17.3+H/2)
    _box(ax, 5, 17.3, 4.5, H, 'Generate G Responses\n生成G个响应', C['dapo'], FS)
    _arrow(ax, 9.8, 18.5, 7.3, 17.3+H/2, color=C['dapo'], lw=1.5, style='->')

    # Row 3: Fan-out into G branches + filter
    g_xs = [2.5, 5, 7.5]
    for gx in g_xs:
        _arrow(ax, 5, 17.3-H/2, gx, 16.0+H/2)

    for gx in g_xs:
        _box(ax, gx, 16.0, 2.2, H*0.9, f'|o_i| tokens', C['gen'], FS-2.5)

    # Row 4: Overlong Filter (branching: valid vs filtered)
    for gx in g_xs:
        _arrow(ax, gx, 16.0-H*0.45, gx, 14.7+H/2)

    # Valid path
    _box(ax, 2.5, 14.7, 2.2, H, 'Valid\n|o|<=Lmax', '#27ae60', FS-1)
    _box(ax, 5, 14.7, 2.2, H, 'Valid\n|o|<=Lmax', '#27ae60', FS-1)
    _box(ax, 7.5, 14.7, 2.2, H, 'Filtered\n|o|>Lmax', '#e74c3c', FS-1)

    _dashed_rect(ax, 1, 14.1, 9, 3.4, '#d35400', 'Innovation 2: Overlong Filter')

    # Row 5: Rewards (only valid)
    _arrow(ax, 2.5, 14.7-H/2, 4, 13.4+H/2)
    _arrow(ax, 5, 14.7-H/2, 4, 13.4+H/2)

    _box(ax, 4, 13.4, 3.2, H, 'Reward r_i\n(valid only)', C['reward'], FS)
    _box(ax, 8.5, 13.4, 3.2, H, 'Reward Shaping\nr_t=base*decay^(T-t)', C['filter'], FS-0.5)
    _arrow(ax, 5.7, 13.4, 6.9, 13.4, color=C['filter'])

    # Row 6: Dynamic Advantage
    _arrow(ax, 4, 13.4-H/2, 6, 12.1+H/2)
    _arrow(ax, 8.5, 13.4-H/2, 6, 12.1+H/2)

    _box(ax, 6, 12.1, 4.5, H, 'Dynamic Advantage\nA=(r-mu_valid)/sigma_valid', C['advantage'], FS-0.5)

    # Row 7: Reference model (side) + Token-level loss
    _box(ax, 13, 13.5, 2.5, 3.0, 'Reference\npi_ref\n(冻结)', C['ref'], 9)

    _arrow(ax, 6, 12.1-H/2, 8, 10.7+H/2)
    _arrow(ax, 13, 11.0, 10, 10.7+H/2, color=C['ref'], curved=-0.1)

    _box(ax, 8, 10.7, 4.5, H*1.2, 'Token-Level Loss\nL=(1/|o|)*sum clip(r*A)\n+ beta*KL_token', C['loss'], FS-1)
    _dashed_rect(ax, 5, 10.0, 6, 2.3, '#c0392b', 'Innovation 3')

    # Row 8: Update
    _arrow(ax, 8, 10.7-H*0.6, 8, 9.2+H/2)
    _box(ax, 8, 9.2, 4, H, 'Update pi_theta', C['update'], 12)

    # Row 9: Dynamic G feedback (loop back)
    _arrow(ax, 8, 9.2-H/2, 8, 8.0+H/2)
    _box(ax, 8, 8.0, 4.5, H, 'Adjust G\nVar(r)>2t: G+2 | Var(r)<t/2: G-2', C['dapo'], FS-1)
    _arrow(ax, 10.3, 8.0, 12, 17.3-H/2, color=C['dapo'], lw=2.5, curved=-0.15)
    _label(ax, 13.5, 12.5, 'G Feedback\n组大小反馈', color=C['dapo'], fs=10)

    # Row 10: Next
    _arrow(ax, 8, 8.0-H/2, 8, 6.8+H/2)
    _box(ax, 8, 6.8, 3.5, H*0.8, 'Next Question', C['done'], 10)

    # Main loop
    _arrow(ax, 14.5, 18.5, 14.5, 7.0, color='#95a5a6', lw=2.5)
    _arrow(ax, 14.5, 7.0, 9.8, 6.8, color='#95a5a6', lw=2.5)
    _label(ax, 15.2, 12.5, 'Main\nLoop', color='#95a5a6', fs=11)

    # Filtered discard annotation
    _arrow(ax, 7.5, 14.7-H/2, 10.5, 13.8, color='#e74c3c', style='->', lw=1.5)
    _label(ax, 11.5, 13.6, 'Discard\nA=0', color='#e74c3c', fs=9)

    plt.tight_layout()
    save_dual(plt.gcf(), 'dapo_flowchart', facecolor=BG)
    plt.close()


# ================================================================
if __name__ == '__main__':
    print("Regenerating 4 flowcharts with multi-branch vector style...")
    ppo_flowchart()
    dpo_flowchart()
    grpo_flowchart()
    dapo_flowchart()
    print(f"\nDone! Saved to: {OUT}")
