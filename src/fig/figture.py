import matplotlib.pyplot as plt
import numpy as np
import matplotlib

# 设置使用Agg后端，避免PyCharm的交互式后端问题
matplotlib.use('Agg')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 从论文中提取的数据
# 假设x轴是训练步骤，y轴是AIME 2024的pass@1准确率
# 这些数据点是根据图2估计的
training_steps = np.array([0, 2000, 4000, 6000, 8000, 10000, 12000, 14000])
aime_accuracy = np.array([15.6, 25.0, 38.0, 50.0, 58.0, 64.0, 68.0, 71.0])

# 创建图表并保存
plt.figure(figsize=(12, 7))
plt.plot(training_steps, aime_accuracy, marker='o', linewidth=2.5, markersize=10, color='#1f77b4')

# 添加标题和标签
plt.title('DeepSeek-R1-Zero 在训练过程中的AIME准确率变化', fontsize=18, fontweight='bold')
plt.xlabel('训练步骤', fontsize=14, fontweight='bold')
plt.ylabel('AIME 2024 平均pass@1准确率 (%)', fontsize=14, fontweight='bold')

# 添加网格线
plt.grid(True, linestyle='--', alpha=0.7)

# 添加详细的数据标注
for i, (x, y) in enumerate(zip(training_steps, aime_accuracy)):
    plt.annotate(f'{y}%',
                 xy=(x, y),
                 xytext=(0, 10),
                 textcoords="offset points",
                 ha='center',
                 va='bottom',
                 fontsize=12,
                 fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.3))

    # 为每个点添加一条垂直虚线到x轴
    plt.plot([x, x], [0, y], 'k--', alpha=0.3)

# 标记起始和最终点
plt.annotate('初始准确率: 15.6%', xy=(0, 15.6), xytext=(500, 5),
             arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
             fontsize=12, fontweight='bold', color='red')

plt.annotate('最终准确率: 71.0%', xy=(14000, 71.0), xytext=(11500, 80),
             arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
             fontsize=12, fontweight='bold', color='red')

# 添加辅助线标记最终性能
plt.axhline(y=71.0, color='r', linestyle='--', alpha=0.5)

# 美化图表
plt.fill_between(training_steps, aime_accuracy, alpha=0.3, color='#1f77b4')

# 添加百分比增长标注
percent_increase = ((71.0 - 15.6) / 15.6) * 100
plt.annotate(f'总体提升: +{percent_increase:.1f}%',
             xy=(7000, 40),
             xytext=(7000, 40),
             fontsize=14,
             fontweight='bold',
             color='green',
             bbox=dict(boxstyle='round,pad=0.5', fc='white', ec='green', alpha=0.8))

# 添加平均每2000步的提升率
avg_increase_per_step = (71.0 - 15.6) / 7  # 7个区间
plt.annotate(f'平均每2000步提升: +{avg_increase_per_step:.2f}%',
             xy=(7000, 30),
             xytext=(7000, 30),
             fontsize=12,
             fontweight='bold',
             color='blue',
             bbox=dict(boxstyle='round,pad=0.5', fc='white', ec='blue', alpha=0.8))

# 设置y轴范围，留出标注空间
plt.ylim(0, 90)
plt.xlim(-500, 15000)

# 设置刻度字体大小
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# 增加重要里程碑标注
milestones = [
    (4000, "突破40%准确率"),
    (8000, "达到60%准确率附近"),
    (12000, "接近70%准确率")
]

for step, desc in milestones:
    idx = np.where(training_steps == step)[0][0]
    acc = aime_accuracy[idx]
    plt.annotate(desc,
                 xy=(step, acc),
                 xytext=(step, acc - 15),
                 ha='center',
                 fontsize=10,
                 arrowprops=dict(arrowstyle='->', color='green'),
                 bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='green', alpha=0.7))

plt.tight_layout()

# 保存图表
plt.savefig('deepseek_r1_zero_aime_progress_detailed.png', dpi=300, bbox_inches='tight')
plt.close()

print("AIME准确率进展详细图表已成功生成！")
