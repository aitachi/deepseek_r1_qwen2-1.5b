# -*- coding: utf-8 -*-
"""Generate all visualization images for PPO/DPO/GRPO/DAPO documentation."""
import os, sys, warnings
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from mpl_toolkits.mplot3d import Axes3D

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['svg.fonttype'] = 'none'  # Keep text as text in SVG

COLORS = {'PPO':'#3498db','DPO':'#e74c3c','GRPO':'#2ecc71','DAPO':'#9b59b6','bg':'#f8f9fa','grid':'#dee2e6','text':'#2c3e50'}
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(OUT, exist_ok=True)

def smooth(d, w=15):
    return np.convolve(d, np.ones(w)/w, mode='same')

def save_dual(fig, name, dpi=200, facecolor=None):
    """Save figure as both PNG and SVG."""
    png_path = f'{OUT}/{name}.png'
    svg_path = f'{OUT}/{name}.svg'
    fig.savefig(png_path, dpi=dpi, bbox_inches='tight', facecolor=facecolor)
    fig.savefig(svg_path, bbox_inches='tight', facecolor=facecolor)
    print(f'[OK] {name}.png + .svg')

def box(ax, x, y, w, h, text, color, fs=9):
    b = FancyBboxPatch((x-w/2, y-h/2), w, h, boxstyle="round,pad=0.02",
                       facecolor=color, edgecolor='#2c3e50', linewidth=1.5, alpha=0.9)
    ax.add_patch(b)
    ax.text(x, y, text, ha='center', va='center', fontsize=fs, fontweight='bold', color='white')

def arrow(ax, x1, y1, x2, y2):
    ax.annotate('', xy=(x2,y2), xytext=(x1,y1), arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=2))

# ---- FLOWCHARTS ----
def ppo_flow():
    fig, ax = plt.subplots(figsize=(14,20)); ax.set_xlim(0,10); ax.set_ylim(0,22); ax.axis('off'); ax.set_facecolor(COLORS['bg']); fig.patch.set_facecolor(COLORS['bg'])
    ax.text(5,21.3,'PPO Algorithm Flow / PPO 算法流程', ha='center', fontsize=18, fontweight='bold', color=COLORS['text'])
    S=[
      (5,20.0,'Input: Question q\n输入问题 q','#34495e',8,0.7),
      (5,18.5,'Policy Network pi_theta\n策略网络生成响应','#3498db',8,0.7),
      (5,17.0,'Generate Response o\n生成响应 o ~ pi_theta(.|q)','#2980b9',8,0.7),
      (5,15.5,'Compute Reward r\n计算奖励 r = R(q, o)','#e67e22',8,0.7),
      (5,14.0,'Value Network V(s)\n价值网络估计 V_phi(s)','#8e44ad',8,0.7),
      (5,12.3,'GAE Advantage\nGAE优势估计\ndelta_t = r_t + gamma*V(s_{t+1}) - V(s_t)\nA_t = sum (gamma*lambda)^l * delta_{t+l}','#16a085',8,1.2),
      (5,10.2,'Clipped Surrogate Loss\n裁剪代理损失\nL = min(r*A, clip(r,1-eps,1+eps)*A)','#c0392b',8,1.0),
      (5,8.5,'Total Loss\n总损失 L_PPO = L_CLIP - c1*L_VF + c2*S','#e74c3c',8,0.7),
      (5,7.1,'Update Policy theta & Value phi\n更新策略和价值网络参数','#2c3e50',8,0.7),
      (5,5.8,'Next Question / 下一个问题','#7f8c8d',6,0.5),
    ]
    for x,y,t,c,w,h in S: box(ax,x,y,w,h,t,c)
    for i in range(len(S)-1): arrow(ax,S[i][0],S[i][1]-S[i][5]/2,S[i+1][0],S[i+1][1]+S[i+1][5]/2)
    ax.annotate('',xy=(8.5,19.5),xytext=(8.5,6.1),arrowprops=dict(arrowstyle='->',color='#95a5a6',lw=2,connectionstyle='arc3,rad=0.3'))
    ax.text(9.2,12,'Repeat\n循环',ha='center',fontsize=10,color='#7f8c8d',fontstyle='italic')
    plt.tight_layout(); save_dual(plt.gcf(), 'ppo_flowchart', facecolor=COLORS['bg']); plt.close()

def dpo_flow():
    fig, ax = plt.subplots(figsize=(14,20)); ax.set_xlim(0,10); ax.set_ylim(0,22); ax.axis('off'); ax.set_facecolor(COLORS['bg']); fig.patch.set_facecolor(COLORS['bg'])
    ax.text(5,21.3,'DPO Algorithm Flow / DPO 算法流程', ha='center', fontsize=18, fontweight='bold', color=COLORS['text'])
    S=[
      (5,20.0,'Input: Question q\n输入问题 q','#34495e',8,0.7),
      (3,18.3,'Generate o1\n生成响应1','#e74c3c',3.5,0.7),
      (7,18.3,'Generate o2\n生成响应2','#e74c3c',3.5,0.7),
      (3,16.6,'Reward r1\n奖励 r1','#e67e22',3.5,0.6),
      (7,16.6,'Reward r2\n奖励 r2','#e67e22',3.5,0.6),
      (5,14.9,'Create Preference Pair\n创建偏好对\ny_w=argmax(r1,r2), y_l=argmin(r1,r2)','#8e44ad',8,1.0),
      (5,13.0,'Compute Log Ratios (preferred)\n计算对数比率(偏好)\nlog(pi_theta(y_w|q)/pi_ref(y_w|q))','#2980b9',8,0.9),
      (5,11.1,'Compute Log Ratios (rejected)\nlog(pi_theta(y_l|q)/pi_ref(y_l|q))','#2980b9',8,0.7),
      (5,9.3,'Bradley-Terry Loss\nBradley-Terry 损失\nL = -log sigma(beta*(log r_w - log r_l))','#c0392b',8,1.1),
      (5,7.3,'Backpropagation\n反向传播','#e74c3c',8,0.6),
      (5,5.8,'Update Policy theta\n更新策略参数','#2c3e50',8,0.6),
      (5,4.3,'Next Question / 下一个问题','#7f8c8d',6,0.5),
    ]
    for x,y,t,c,w,h in S: box(ax,x,y,w,h,t,c)
    arrow(ax,5,20.0-0.35,3,18.3+0.35); arrow(ax,5,20.0-0.35,7,18.3+0.35)
    arrow(ax,3,18.3-0.35,3,16.6+0.3); arrow(ax,7,18.3-0.35,7,16.6+0.3)
    arrow(ax,3,16.6-0.3,5,14.9+0.5); arrow(ax,7,16.6-0.3,5,14.9+0.5)
    for i in range(6,len(S)-1): arrow(ax,S[i][0],S[i][1]-S[i][5]/2,S[i+1][0],S[i+1][1]+S[i+1][5]/2)
    ax.annotate('',xy=(8.5,19.5),xytext=(8.5,4.6),arrowprops=dict(arrowstyle='->',color='#95a5a6',lw=2,connectionstyle='arc3,rad=0.3'))
    ax.text(9.2,12,'Repeat\n循环',ha='center',fontsize=10,color='#7f8c8d',fontstyle='italic')
    plt.tight_layout(); save_dual(plt.gcf(), 'dpo_flowchart', facecolor=COLORS['bg']); plt.close()

def grpo_flow():
    fig, ax = plt.subplots(figsize=(14,20)); ax.set_xlim(0,10); ax.set_ylim(0,22); ax.axis('off'); ax.set_facecolor(COLORS['bg']); fig.patch.set_facecolor(COLORS['bg'])
    ax.text(5,21.3,'GRPO Algorithm Flow / GRPO 算法流程', ha='center', fontsize=18, fontweight='bold', color=COLORS['text'])
    S=[
      (5,20.0,'Input: Question q\n输入问题 q','#34495e',8,0.7),
      (5,18.2,'Generate G Responses\n生成 G 个响应 {o1..oG}\n(Group Sampling / 组采样)','#27ae60',8,1.0),
      (5,16.2,'Compute Rewards\n计算奖励 r_i = R(q, o_i), i=1..G','#e67e22',8,0.7),
      (5,14.3,'Group Advantage Normalization\n组优势归一化\nA_i = (r_i - mean(r)) / std(r)','#16a085',8,1.1),
      (5,12.2,'Compute Ratio\n计算比率 r_t = pi_theta(o_i|q) / pi_old(o_i|q)','#2980b9',8,0.7),
      (5,10.3,'Clipped Surrogate + KL Penalty\n裁剪代理 + KL 惩罚\nL = -min(r*A, clip(r,1-e,1+e)*A) + beta*KL','#c0392b',8,1.2),
      (5,8.2,'Average over G samples\n对 G 个样本取平均','#8e44ad',8,0.6),
      (5,6.8,'Backpropagation\n反向传播','#e74c3c',6,0.5),
      (5,5.4,'Update Policy theta\n更新策略参数','#2c3e50',8,0.6),
      (5,4.0,'Next Question / 下一个问题','#7f8c8d',6,0.5),
    ]
    for x,y,t,c,w,h in S: box(ax,x,y,w,h,t,c)
    for i in range(len(S)-1): arrow(ax,S[i][0],S[i][1]-S[i][5]/2,S[i+1][0],S[i+1][1]+S[i+1][5]/2)
    rect = plt.Rectangle((0.5,15.4),9,3.8,fill=False,edgecolor='#27ae60',linewidth=2,linestyle='--'); ax.add_patch(rect)
    ax.text(0.8,19.0,'Group Operation\n组操作',fontsize=9,color='#27ae60',fontstyle='italic')
    ax.annotate('',xy=(8.5,19.5),xytext=(8.5,4.3),arrowprops=dict(arrowstyle='->',color='#95a5a6',lw=2,connectionstyle='arc3,rad=0.3'))
    ax.text(9.2,12,'Repeat\n循环',ha='center',fontsize=10,color='#7f8c8d',fontstyle='italic')
    plt.tight_layout(); save_dual(plt.gcf(), 'grpo_flowchart', facecolor=COLORS['bg']); plt.close()

def dapo_flow():
    fig, ax = plt.subplots(figsize=(14,22)); ax.set_xlim(0,10); ax.set_ylim(0,24); ax.axis('off'); ax.set_facecolor(COLORS['bg']); fig.patch.set_facecolor(COLORS['bg'])
    ax.text(5,23.3,'DAPO Algorithm Flow / DAPO 算法流程', ha='center', fontsize=18, fontweight='bold', color=COLORS['text'])
    S=[
      (5,22.0,'Input: Question q\n输入问题 q','#34495e',8,0.7),
      (5,20.3,'Dynamic Sampling: Generate G responses\n动态采样: 生成 G 个响应\nG adapts based on reward variance\nG 根据奖励方差自适应调整','#9b59b6',8,1.2),
      (5,18.2,'Overlong Filtering\n过长过滤\nFilter |o_i| > L_max before advantage\n在优势计算前过滤超长响应','#8e44ad',8,1.1),
      (5,16.0,'Compute Rewards r_i\n计算奖励 r_i = R(q, o_i)','#e67e22',8,0.7),
      (5,14.4,'Reward Shaping (per-token)\n奖励塑形 (逐token)\nr_t = r_base * decay^(T-t)','#d35400',8,1.0),
      (5,12.5,'Dynamic Advantage\n动态优势计算\nA_i = (r_i - mu_valid) / sigma_valid\nOnly valid samples participate\n仅有效样本参与','#16a085',8,1.2),
      (5,10.3,'Token-Level Loss\nToken级别损失\nL_i = -(1/|o_i|)*sum min(r_t*A_i, clip(r_t)*A_i)','#c0392b',8,1.1),
      (5,8.2,'KL Penalty + Total Loss\nKL惩罚 + 总损失\nL_DAPO = L_policy + beta*D_KL','#e74c3c',8,0.7),
      (5,6.5,'Adjust Group Size G\n调整组大小 G\nBased on reward variance\n根据奖励方差调整','#9b59b6',8,0.9),
      (5,4.8,'Update Policy theta\n更新策略参数','#2c3e50',8,0.6),
      (5,3.3,'Next Question / 下一个问题','#7f8c8d',6,0.5),
    ]
    for x,y,t,c,w,h in S: box(ax,x,y,w,h,t,c)
    for i in range(len(S)-1): arrow(ax,S[i][0],S[i][1]-S[i][5]/2,S[i+1][0],S[i+1][1]+S[i+1][5]/2)
    for yt,yb,lb in [(21.0,19.5,'Innovation 1:\nDynamic\nSampling'),(19.3,17.4,'Innovation 2:\nOverlong\nFiltering'),(11.0,9.5,'Innovation 3:\nToken-Level\nLoss')]:
        r=plt.Rectangle((0.3,yb),2.2,yt-yb,fill=True,facecolor='#f0e6ff',edgecolor='#9b59b6',linewidth=2,linestyle='--',alpha=0.7); ax.add_patch(r)
        ax.text(1.4,(yt+yb)/2,lb,ha='center',va='center',fontsize=8,color='#9b59b6',fontweight='bold')
    ax.annotate('',xy=(8.5,21.5),xytext=(8.5,3.6),arrowprops=dict(arrowstyle='->',color='#95a5a6',lw=2,connectionstyle='arc3,rad=0.3'))
    ax.text(9.2,12,'Repeat\n循环',ha='center',fontsize=10,color='#7f8c8d',fontstyle='italic')
    plt.tight_layout(); save_dual(plt.gcf(), 'dapo_flowchart', facecolor=COLORS['bg']); plt.close()

# ---- DATA FLOWS ----
def data_flow(name, fn, boxes):
    fig, ax = plt.subplots(figsize=(16,6)); ax.set_xlim(-0.5,len(boxes)*2.5); ax.set_ylim(-1,3); ax.axis('off'); ax.set_facecolor(COLORS['bg']); fig.patch.set_facecolor(COLORS['bg'])
    ax.text(len(boxes)*1.25-0.25,2.7,f'{name} Data Processing Pipeline / {name} 数据处理流程',ha='center',fontsize=14,fontweight='bold',color=COLORS['text'])
    for i,(lb,sd,cl) in enumerate(boxes):
        x=i*2.5+1.2
        b=FancyBboxPatch((x-1.0,0.3),2.0,1.5,boxstyle="round,pad=0.05",facecolor=cl,edgecolor='#2c3e50',linewidth=1.5,alpha=0.85); ax.add_patch(b)
        ax.text(x,1.05,lb,ha='center',va='center',fontsize=7.5,fontweight='bold',color='white')
        if sd: ax.text(x,-0.1,sd,ha='center',fontsize=6,color='#7f8c8d',fontstyle='italic')
        if i<len(boxes)-1: ax.annotate('',xy=(x+1.1,1.05),xytext=(x+1.0,1.05),arrowprops=dict(arrowstyle='->',color='#2c3e50',lw=2))
    plt.tight_layout(); name = fn.replace('.png',''); save_dual(plt.gcf(), name, facecolor=COLORS['bg']); plt.close()

def all_data_flows():
    data_flow('PPO','ppo_data_flow.png',[('Training\nData\n训练数据','List[Dict]','#34495e'),('Prompt\nSelection\n提示选择','q_i','#3498db'),('Response\nGeneration\n响应生成','o ~ pi','#2980b9'),('Reward\nComputation\n奖励计算','r=R(q,o)','#e67e22'),('GAE\nComputation\nGAE计算','A_t, R_t','#16a085'),('Policy\nUpdate\n策略更新','theta update','#c0392b')])
    data_flow('DPO','dpo_data_flow.png',[('Training\nData\n训练数据','List[Dict]','#34495e'),('Dual\nGeneration\n双重生成','o1, o2','#e74c3c'),('Reward\nRanking\n奖励排序','r1, r2','#e67e22'),('Preference\nPair\n偏好对','(y_w, y_l)','#8e44ad'),('Log Ratio\nCompute\n对数比率','log r_w-r_l','#2980b9'),('Loss\nCompute\n损失计算','L_DPO','#c0392b')])
    data_flow('GRPO','grpo_data_flow.png',[('Training\nData\n训练数据','List[Dict]','#34495e'),('Group\nGeneration\n组生成','{o1..oG}','#27ae60'),('Reward\nArray\n奖励数组','[r1..rG]','#e67e22'),('Group\nNormalize\n组归一化','A_i','#16a085'),('Clipped\nObjective\n裁剪目标','L_clip','#c0392b'),('KL\nPenalty\nKL惩罚','+beta*KL','#8e44ad'),('Policy\nUpdate\n策略更新','theta update','#2c3e50')])
    data_flow('DAPO','dapo_data_flow.png',[('Training\nData\n训练数据','List[Dict]','#34495e'),('Dynamic G\nResponses\n动态响应','{o1..oG}','#9b59b6'),('Length\nFilter\n长度过滤','valid[]','#8e44ad'),('Reward\nShaping\n奖励塑形','r_t/token','#d35400'),('Dynamic\nAdvantage\n动态优势','A_i','#16a085'),('Token\nLoss\nToken损失','L_DAPO','#c0392b'),('Group\nAdjust\n组调整','G +/- d','#9b59b6')])

# ---- CONVERGENCE ----
def conv(name, fn, cfgs):
    fig, axes = plt.subplots(2,2,figsize=(14,10)); fig.suptitle(f'{name} Training Convergence / {name} 训练收敛曲线',fontsize=16,fontweight='bold',color=COLORS['text'])
    np.random.seed(42); n=200
    for ax,(lb,cl,bf,ns,yl) in zip(axes.flat,cfgs):
        x=np.arange(n); base=bf(n); y=base+np.random.normal(0,ns(n),n); ys=smooth(y,15)
        ax.fill_between(x,y,alpha=0.15,color=cl); ax.plot(x,y,alpha=0.3,color=cl,linewidth=0.5); ax.plot(x,ys,color=cl,linewidth=2.5,label=f'{lb}')
        ax.set_xlabel('Training Step / 训练步数',fontsize=10); ax.set_ylabel(yl,fontsize=10); ax.set_title(lb,fontsize=12,fontweight='bold'); ax.legend(fontsize=9); ax.grid(True,alpha=0.3); ax.set_facecolor(COLORS['bg'])
    plt.tight_layout(); name = fn.replace('.png',''); save_dual(plt.gcf(), name); plt.close()

def all_conv():
    conv('PPO','ppo_convergence.png',[('Policy Loss / 策略损失',COLORS['PPO'],lambda n:2.5*np.exp(-np.arange(n)/80)+0.5,lambda n:0.15,'Loss'),('Value Loss / 价值损失','#e74c3c',lambda n:3.0*np.exp(-np.arange(n)/60)+0.3,lambda n:0.2,'Loss'),('Entropy / 熵','#27ae60',lambda n:2.0-0.5*(1-np.exp(-np.arange(n)/120)),lambda n:0.08,'Entropy'),('Reward / 奖励','#e67e22',lambda n:2.0+6.0*(1-np.exp(-np.arange(n)/70)),lambda n:0.5,'Reward')])
    conv('DPO','dpo_convergence.png',[('DPO Loss / DPO损失',COLORS['DPO'],lambda n:0.7*np.exp(-np.arange(n)/50)+0.05,lambda n:0.05,'Loss'),('Preference Accuracy / 偏好准确率','#27ae60',lambda n:0.55+0.40*(1-np.exp(-np.arange(n)/60)),lambda n:0.03,'Accuracy'),('Margin (log rw - log rl)','#8e44ad',lambda n:0.1+2.4*(1-np.exp(-np.arange(n)/80)),lambda n:0.15,'Margin'),('Reward / 奖励','#e67e22',lambda n:2.5+5.0*(1-np.exp(-np.arange(n)/65)),lambda n:0.4,'Reward')])
    conv('GRPO','grpo_convergence.png',[('Total Loss / 总损失',COLORS['GRPO'],lambda n:3.0*np.exp(-np.arange(n)/70)+0.2,lambda n:0.2,'Loss'),('KL Divergence / KL散度','#8e44ad',lambda n:0.15*np.exp(-np.arange(n)/100)+0.03,lambda n:0.008,'D_KL'),('Avg Reward / 平均奖励','#e67e22',lambda n:1.5+7.5*(1-np.exp(-np.arange(n)/60)),lambda n:0.6,'Reward'),('Max Group Reward / 组最大奖励','#c0392b',lambda n:3.0+7.0*(1-np.exp(-np.arange(n)/50)),lambda n:0.4,'Reward')])
    conv('DAPO','dapo_convergence.png',[('Token-Level Loss / Token级损失',COLORS['DAPO'],lambda n:2.0*np.exp(-np.arange(n)/60)+0.15,lambda n:0.12,'Loss'),('KL Divergence / KL散度','#8e44ad',lambda n:0.1*np.exp(-np.arange(n)/90)+0.025,lambda n:0.006,'D_KL'),('Reward / 奖励','#e67e22',lambda n:2.0+7.5*(1-np.exp(-np.arange(n)/55)),lambda n:0.5,'Reward'),('Dynamic Group Size / 动态组大小','#16a085',lambda n:16+4*np.sin(np.arange(n)/20)-0.02*np.arange(n),lambda n:1.5,'Group Size')])

# ---- 2D COMPARISONS ----
def loss_2d():
    fig,ax=plt.subplots(figsize=(12,7)); np.random.seed(42); n=200; x=np.arange(n)
    losses={'PPO':2.5*np.exp(-x/80)+0.5,'DPO':0.7*np.exp(-x/50)+0.05,'GRPO':3.0*np.exp(-x/70)+0.2,'DAPO':2.0*np.exp(-x/60)+0.15}
    for nm,l in losses.items(): ns=l+np.random.normal(0,0.1,n); ax.plot(x,smooth(ns,15),color=COLORS[nm],linewidth=2.5,label=nm); ax.fill_between(x,ns,alpha=0.08,color=COLORS[nm])
    ax.set_xlabel('Training Step / 训练步数',fontsize=12); ax.set_ylabel('Loss / 损失',fontsize=12); ax.set_title('Loss Curves Comparison / 损失曲线对比',fontsize=14,fontweight='bold'); ax.legend(fontsize=11); ax.grid(True,alpha=0.3); ax.set_facecolor(COLORS['bg']); fig.patch.set_facecolor(COLORS['bg'])
    plt.tight_layout(); save_dual(plt.gcf(), 'comparison_loss_curves_2d'); plt.close()

def reward_2d():
    fig,ax=plt.subplots(figsize=(12,7)); np.random.seed(123); n=200; x=np.arange(n)
    rewards={'PPO':2.0+6.0*(1-np.exp(-x/70)),'DPO':2.5+5.0*(1-np.exp(-x/65)),'GRPO':1.5+7.5*(1-np.exp(-x/60)),'DAPO':2.0+7.5*(1-np.exp(-x/55))}
    for nm,r in rewards.items(): ns=r+np.random.normal(0,0.3,n); ax.plot(x,smooth(ns,15),color=COLORS[nm],linewidth=2.5,label=nm); ax.fill_between(x,ns,alpha=0.08,color=COLORS[nm])
    ax.set_xlabel('Training Step / 训练步数',fontsize=12); ax.set_ylabel('Reward / 奖励',fontsize=12); ax.set_title('Reward Curves Comparison / 奖励曲线对比',fontsize=14,fontweight='bold'); ax.legend(fontsize=11,loc='lower right'); ax.grid(True,alpha=0.3); ax.set_facecolor(COLORS['bg']); fig.patch.set_facecolor(COLORS['bg'])
    plt.tight_layout(); save_dual(plt.gcf(), 'comparison_reward_curves_2d'); plt.close()

def radar():
    cats=['Sample\nEfficiency\n采样效率','Training\nSpeed\n训练速度','Memory\nEfficiency\n内存效率','Implementation\nSimplicity\n实现简单性','Final\nPerformance\n最终性能','Stability\n稳定性']
    scores={'PPO':[70,60,55,55,80,75],'DPO':[65,75,80,80,70,90],'GRPO':[85,80,85,70,85,80],'DAPO':[90,75,75,60,90,85]}
    N=len(cats); angles=np.linspace(0,2*np.pi,N,endpoint=False).tolist(); angles+=angles[:1]
    fig,ax=plt.subplots(figsize=(10,10),subplot_kw=dict(polar=True)); fig.patch.set_facecolor(COLORS['bg']); ax.set_facecolor(COLORS['bg'])
    for nm,sc in scores.items(): sp=sc+sc[:1]; ax.plot(angles,sp,'o-',linewidth=2.5,label=nm,color=COLORS[nm],markersize=8); ax.fill(angles,sp,alpha=0.1,color=COLORS[nm])
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(cats,size=10); ax.set_ylim(0,100); ax.set_yticks([20,40,60,80,100]); ax.set_yticklabels(['20','40','60','80','100'],size=8); ax.grid(True,alpha=0.3); ax.legend(loc='upper right',bbox_to_anchor=(1.25,1.1),fontsize=11); ax.set_title('Algorithm Radar / 算法雷达图',size=14,fontweight='bold',pad=20)
    plt.tight_layout(); save_dual(plt.gcf(), 'comparison_radar_chart'); plt.close()

def bar_2d():
    fig,ax=plt.subplots(figsize=(14,7)); metrics=['Training Time\n训练时间','Memory Usage\n内存使用','Sample Efficiency\n采样效率','Final Reward\n最终奖励']; x=np.arange(len(metrics)); w=0.18
    data={'PPO':[85,90,70,80],'DPO':[60,70,65,70],'GRPO':[70,60,85,85],'DAPO':[75,75,90,90]}
    for i,(nm,v) in enumerate(data.items()):
        off=(i-1.5)*w; bars=ax.bar(x+off,v,w,label=nm,color=COLORS[nm],edgecolor='white',alpha=0.85)
        for bar,val in zip(bars,v): ax.text(bar.get_x()+bar.get_width()/2.,bar.get_height()+1,f'{val}',ha='center',va='bottom',fontsize=8,fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(metrics,fontsize=10); ax.set_ylabel('Score / 分数',fontsize=12); ax.set_title('Algorithm Comparison / 算法对比',fontsize=14,fontweight='bold'); ax.legend(fontsize=11); ax.grid(axis='y',alpha=0.3); ax.set_facecolor(COLORS['bg']); fig.patch.set_facecolor(COLORS['bg'])
    plt.tight_layout(); save_dual(plt.gcf(), 'comparison_bar_chart_2d'); plt.close()

# ---- 3D ----
def surface_3d():
    fig=plt.figure(figsize=(16,14)); fig.patch.set_facecolor(COLORS['bg']); fig.suptitle('3D Loss Surface: LR x Clip Epsilon\n三维损失曲面: 学习率 x 裁剪参数',fontsize=16,fontweight='bold',color=COLORS['text'])
    lr=np.linspace(-6,-4,40); ep=np.linspace(0.05,0.4,40); LR,EP=np.meshgrid(lr,ep); np.random.seed(42)
    fns={'PPO':(1,'viridis',lambda l,e:0.5+1.5*(l+5)**2+2*(e-0.2)**2),'DPO':(2,'magma',lambda l,e:0.1+0.8*(l+5)**2+0.5*(e-0.15)**2),'GRPO':(3,'YlGn',lambda l,e:0.3+1.2*(l+5)**2+1.5*(e-0.2)**2),'DAPO':(4,'Purples',lambda l,e:0.2+1.0*(l+5)**2+1.8*(e-0.18)**2)}
    for nm,(pos,cm,fn) in fns.items():
        ax=fig.add_subplot(2,2,pos,projection='3d'); Z=fn(LR,EP); ax.plot_surface(LR,EP,Z,cmap=cm,alpha=0.8,edgecolor='none')
        ax.set_xlabel('log10(LR)',fontsize=9); ax.set_ylabel('Clip eps',fontsize=9); ax.set_zlabel('Loss',fontsize=9); ax.set_title(nm,fontsize=12,fontweight='bold',color=COLORS[nm]); ax.view_init(elev=25,azim=45)
    plt.tight_layout(); plt.savefig(f'{OUT}/comparison_3d_surface.png',dpi=200,bbox_inches='tight'); plt.close(); print('[OK] comparison_3d_surface.png (PNG only - 3D)')

def tradeoff_3d():
    fig=plt.figure(figsize=(14,10)); fig.patch.set_facecolor(COLORS['bg']); ax=fig.add_subplot(111,projection='3d'); np.random.seed(42)
    ad={'PPO':(60,80,85),'DPO':(70,70,65),'GRPO':(80,85,55),'DAPO':(75,90,70)}
    mk={'PPO':'o','DPO':'s','GRPO':'^','DAPO':'D'}
    for nm,(cx,cy,cz) in ad.items():
        x=cx+np.random.normal(0,3,20); y=cy+np.random.normal(0,3,20); z=cz+np.random.normal(0,3,20)
        ax.scatter(x,y,z,c=COLORS[nm],marker=mk[nm],s=100,alpha=0.7,label=nm,edgecolors='white',linewidth=0.5); ax.text(cx,cy,cz+5,nm,fontsize=11,fontweight='bold',color=COLORS[nm],ha='center')
    ax.set_xlabel('Training Efficiency',fontsize=11,labelpad=10); ax.set_ylabel('Final Performance',fontsize=11,labelpad=10); ax.set_zlabel('Memory Usage',fontsize=11,labelpad=10); ax.set_title('3D Performance Trade-offs / 三维性能权衡',fontsize=14,fontweight='bold',pad=20); ax.legend(fontsize=11,loc='upper left'); ax.view_init(elev=20,azim=135); ax.set_facecolor(COLORS['bg'])
    plt.tight_layout(); plt.savefig(f'{OUT}/comparison_3d_tradeoff.png',dpi=200,bbox_inches='tight'); plt.close(); print('[OK] comparison_3d_tradeoff.png (PNG only - 3D)')

# ---- MAIN ----
if __name__=='__main__':
    print("="*60)
    print("Generating all images...")
    print("="*60)
    print("\n[1/5] Flowcharts..."); ppo_flow(); dpo_flow(); grpo_flow(); dapo_flow()
    print("\n[2/5] Data Flows..."); all_data_flows()
    print("\n[3/5] Convergence..."); all_conv()
    print("\n[4/5] 2D Comparisons..."); loss_2d(); reward_2d(); radar(); bar_2d()
    print("\n[5/5] 3D Comparisons..."); surface_3d(); tradeoff_3d()
    print(f"\nDone! All images saved to: {OUT}")
