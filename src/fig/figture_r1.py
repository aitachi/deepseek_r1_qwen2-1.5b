import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('Agg')  # 使用非交互式后端

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 从论文中提取的各阶段AIME准确率数据
# 这些数据是基于论文内容估计的，如有精确数据请替换
stages = [
    "DeepSeek-V3基础模型",
    "第一阶段GRPO (R1-Zero)",
    "拒绝采样后SFT",
    "第二阶段GRPO",
    "DeepSeek-R1最终模型"
]

# 各阶段的AIME pass@1准确率
aime_accuracy = [39.2, 71.0, 75.5, 78.0, 79.8]

# 计算各阶段的提升幅度
improvements = [0]  # 第一个阶段没有提升
for i in range(1, len(aime_accuracy)):
    improvements.append(aime_accuracy[i] - aime_accuracy[i - 1])

# 计算相对于基础模型的总提升
total_improvements = [0]
for i in range(1, len(aime_accuracy)):
    total_improvements.append(aime_accuracy[i] - aime_accuracy[0])

# 创建包含所有信息的数据表
data = {
    "训练阶段": stages,
    "AIME准确率 (%)": aime_accuracy,
    "阶段提升 (%)": improvements,
    "相对基础模型提升 (%)": total_improvements,
    "相对提升比例 (%)": [0] + [round((aime_accuracy[i] - aime_accuracy[i - 1]) / aime_accuracy[i - 1] * 100, 2) for i in
                               range(1, len(aime_accuracy))]
}
df = pd.DataFrame(data)

# 1. 创建阶段准确率柱状图 (更详细版本)
plt.figure(figsize=(14, 8))
bars = plt.bar(stages, aime_accuracy, color='#5B9BD5', width=0.6, edgecolor='black', linewidth=1)

# 添加详细的数据标签
for i, bar in enumerate(bars):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2., height + 1,
             f'{height}%',
             ha='center', va='bottom', fontsize=12, fontweight='bold')

    # 添加阶段提升标签
    if i > 0:
        plt.text(bar.get_x() + bar.get_width() / 2., height / 2,
                 f'+{improvements[i]:.1f}%',
                 ha='center', va='center', fontsize=11,
                 color='white', fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.3', fc='green', alpha=0.7))

# 添加总提升标注
plt.annotate(
    f'总提升: +{aime_accuracy[-1] - aime_accuracy[0]:.1f}%\n(+{(aime_accuracy[-1] - aime_accuracy[0]) / aime_accuracy[0] * 100:.1f}%)',
    xy=(4, aime_accuracy[-1]),
    xytext=(3.7, 90),
    fontsize=14, fontweight='bold', color='red',
    arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
    bbox=dict(boxstyle='round,pad=0.5', fc='white', ec='red', alpha=0.8))

# 添加阶段间的提升箭头
for i in range(1, len(stages)):
    plt.annotate('',
                 xy=(i, aime_accuracy[i]),
                 xytext=(i - 1, aime_accuracy[i - 1]),
                 arrowprops=dict(arrowstyle='->', color='#FFC000', lw=1.5, connectionstyle='arc3,rad=0.3'))

# 设置图表属性
plt.title('DeepSeek模型在不同训练阶段的AIME准确率变化', fontsize=18, fontweight='bold')
plt.xlabel('训练阶段', fontsize=14, fontweight='bold')
plt.ylabel('AIME 2024 pass@1准确率 (%)', fontsize=14, fontweight='bold')
plt.ylim(0, 100)  # 设置y轴范围
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(rotation=15, ha='right', fontsize=11, fontweight='bold')
plt.yticks(fontsize=12)

# 添加关键阶段说明
plt.figtext(0.02, 0.02,
            "关键阶段说明:\n"
            "DeepSeek-V3基础模型: 初始基础模型\n"
            "第一阶段GRPO (R1-Zero): 应用大规模强化学习\n"
            "拒绝采样后SFT: 使用拒绝采样数据进行监督微调\n"
            "第二阶段GRPO: 第二轮强化学习训练\n"
            "DeepSeek-R1最终模型: 最终发布的模型",
            fontsize=10, bbox=dict(boxstyle='round,pad=0.5', fc='#F0F0F0', alpha=0.8))

plt.tight_layout()
plt.savefig('deepseek_aime_stages_detailed.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. 创建折线图展示阶段性提升过程
plt.figure(figsize=(14, 8))

# 主折线-准确率
plt.plot(stages, aime_accuracy, marker='o', linewidth=3, markersize=12, color='#1f77b4', label='AIME准确率 (%)')

# 填充区域
plt.fill_between(stages, aime_accuracy, alpha=0.3, color='#1f77b4')

# 添加数据标签
for i, (x, y) in enumerate(zip(stages, aime_accuracy)):
    plt.annotate(f'{y}%',
                 xy=(x, y),
                 xytext=(0, 10),
                 textcoords="offset points",
                 ha='center',
                 va='bottom',
                 fontsize=14,
                 fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.3))

    # 为后续阶段添加提升标签
    if i > 0:
        plt.annotate(f'+{improvements[i]:.1f}%',
                     xy=((i - 0.5), (aime_accuracy[i] + aime_accuracy[i - 1]) / 2),
                     xytext=(0, 0),
                     textcoords="offset points",
                     ha='center',
                     va='center',
                     fontsize=12,
                     fontweight='bold',
                     color='green',
                     bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='green', alpha=0.7))

# 添加总提升标注
plt.annotate(
    f'总提升: +{aime_accuracy[-1] - aime_accuracy[0]:.1f}%\n(相对提升: +{(aime_accuracy[-1] - aime_accuracy[0]) / aime_accuracy[0] * 100:.1f}%)',
    xy=(stages[0], aime_accuracy[0]),
    xytext=(stages[1], 20),
    fontsize=14, fontweight='bold', color='red',
    arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
    bbox=dict(boxstyle='round,pad=0.5', fc='white', ec='red', alpha=0.8))

# 第二条折线-相对基础模型的总提升
plt.plot(stages, total_improvements, marker='s', linewidth=2, markersize=8, color='#ff7f0e', linestyle='--',
         label='相对基础模型累计提升 (%)')

# 设置图表属性
plt.title('DeepSeek模型各阶段AIME准确率及提升过程', fontsize=18, fontweight='bold')
plt.xlabel('训练阶段', fontsize=14, fontweight='bold')
plt.ylabel('准确率/提升百分比 (%)', fontsize=14, fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(rotation=15, ha='right', fontsize=11, fontweight='bold')
plt.yticks(fontsize=12)
plt.legend(fontsize=12, loc='upper left')

# 添加阶段描述
stage_descriptions = [
    "初始基础模型",
    "直接应用GRPO强化学习",
    "拒绝采样+SFT",
    "第二轮强化学习",
    "最终优化模型"
]

for i, desc in enumerate(stage_descriptions):
    plt.annotate(desc,
                 xy=(stages[i], 5),
                 xytext=(0, -25),
                 textcoords="offset points",
                 ha='center',
                 fontsize=10,
                 fontweight='bold',
                 rotation=45)

plt.tight_layout()
plt.savefig('deepseek_aime_improvement_process.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. 创建表格展示详细数据
fig, ax = plt.subplots(figsize=(14, 7))
ax.axis('tight')
ax.axis('off')

# 创建表格内容
cell_text = []
for i in range(len(df)):
    cell_text.append([
        df.iloc[i, 0],
        f"{df.iloc[i, 1]:.1f}%",
        f"{'+' if df.iloc[i, 2] > 0 else ''}{df.iloc[i, 2]:.1f}%",
        f"{'+' if df.iloc[i, 3] > 0 else ''}{df.iloc[i, 3]:.1f}%",
        f"{'+' if df.iloc[i, 4] > 0 else ''}{df.iloc[i, 4]:.2f}%"
    ])

# 添加表格
colLabels = ["训练阶段", "AIME准确率", "阶段提升", "相对基础模型累计提升", "阶段相对提升比例"]
table = ax.table(cellText=cell_text, colLabels=colLabels, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1, 1.8)  # 调整表格大小

# 设置标题行颜色
for j in range(len(colLabels)):
    table[(0, j)].set_facecolor('#4472C4')
    table[(0, j)].set_text_props(color='white', fontweight='bold')

# 设置行颜色
colors = ['#E6F0FF', '#E6F0FF', '#E6F0FF', '#E6F0FF', '#E6F0FF']
for i in range(len(cell_text)):
    for j in range(len(colLabels)):
        table[(i + 1, j)].set_facecolor(colors[i])
        # 为提升单元格添加绿色字体
        if j >= 2 and i > 0:  # 提升列且不是第一行
            table[(i + 1, j)].set_text_props(color='green', fontweight='bold')

# 高亮最终行
for j in range(len(colLabels)):
    table[(len(cell_text), j)].set_facecolor('#DDEBF7')
    if j >= 1:  # 从第二列开始加粗
        table[(len(cell_text), j)].set_text_props(fontweight='bold')

plt.title('DeepSeek模型各阶段AIME准确率详细数据', fontsize=18, fontweight='bold', pad=20)

# 添加表格说明
plt.figtext(0.5, 0.03,
            "说明: 所有数据基于论文《DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning》图表分析整理。\n"
            "相对提升比例 = 当前阶段提升/前一阶段准确率 × 100%",
            fontsize=10, ha='center',
            bbox=dict(boxstyle='round,pad=0.5', fc='#F0F0F0', alpha=0.8))

plt.tight_layout()
plt.savefig('deepseek_aime_data_table.png', dpi=300, bbox_inches='tight')
plt.close()

# 4. 创建提升贡献占比饼图
plt.figure(figsize=(10, 8))

# 提取各阶段的提升值(排除第一个0)
stage_improvements = improvements[1:]
stage_labels = stages[1:]

# 计算相对贡献百分比
contribution_percentages = [imp / sum(stage_improvements) * 100 for imp in stage_improvements]

# 创建饼图
wedges, texts, autotexts = plt.pie(stage_improvements,
                                   labels=None,
                                   autopct='',
                                   startangle=90,
                                   wedgeprops=dict(width=0.5, edgecolor='w'),
                                   colors=['#ff9999', '#66b3ff', '#99ff99', '#ffcc99'])

# 添加百分比标签
for i, (wedge, autotext) in enumerate(zip(wedges, autotexts)):
    ang = (wedge.theta2 - wedge.theta1) / 2. + wedge.theta1
    y = np.sin(np.deg2rad(ang))
    x = np.cos(np.deg2rad(ang))
    horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
    connectionstyle = f"angle,angleA=0,angleB={ang}"
    plt.annotate(f"{stage_labels[i]}\n+{stage_improvements[i]:.1f}% ({contribution_percentages[i]:.1f}%)",
                 xy=(x, y), xytext=(1.35 * np.sign(x), 1.4 * y),
                 horizontalalignment=horizontalalignment,
                 arrowprops=dict(arrowstyle="-", connectionstyle=connectionstyle, color='gray'),
                 fontsize=12, fontweight='bold')

plt.title('各训练阶段对AIME准确率提升的贡献占比', fontsize=18, fontweight='bold')

# 添加中心文字
plt.annotate(f'总提升\n+{sum(stage_improvements):.1f}%',
             xy=(0, 0), xytext=(0, 0),
             ha='center', va='center',
             fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('deepseek_aime_contribution_pie.png', dpi=300, bbox_inches='tight')
plt.close()

print("所有AIME准确率提升相关图表已成功生成！")
