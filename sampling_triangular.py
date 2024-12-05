import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pandas as pd

# 设置随机种子以确保结果可重现
np.random.seed(42)

# 定义参数
sample_sizes = [1, 2, 4, 8, 16, 32]  # 修改为包含n=1
n_simulations = 500  # 模拟次数
left = 0    # 三角分布的最小值
mode = 2    # 三角分布的众数
right = 4   # 三角分布的最大值

# 存储所有样本大小的统计数据
stats_data = []

# 为每个样本大小创建图形
for n_samples in sample_sizes:
    # 进行多次抽样并计算每次抽样的均值
    sample_means = []
    for _ in range(n_simulations):
        samples = np.random.triangular(left, mode, right, size=n_samples)
        sample_means.append(np.mean(samples))
    
    # 计算统计量
    mean = np.mean(sample_means)
    std = np.std(sample_means)
    skewness = stats.skew(sample_means)
    
    # 存储统计数据
    stats_data.append({
        'n': n_samples,
        'x̄': mean,
        's': std,
        'g₁': skewness
    })
    
    # 创建图形
    if n_samples == 32:
        # 为n=32创建包含直方图和正态概率图的两个子图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    else:
        # 为其他n值只创建直方图
        plt.figure(figsize=(8, 6))
        ax1 = plt.gca()
    
    # 绘制直方图和拟合曲线
    sns.histplot(sample_means, kde=False, ax=ax1)
    if n_samples == 32:  # 只为n=32添加拟合曲线
        x = np.linspace(min(sample_means), max(sample_means), 100)
        y = stats.norm.pdf(x, mean, std)
        ax1.plot(x, y * len(sample_means) * (max(sample_means) - min(sample_means)) / 30, 
                'k-', lw=2, label=f'N({mean:.3f}, {std:.3f}²)')
        ax1.legend()
    
    ax1.set_title(f'Histogram for simulated sample means, n = {n_samples}')
    ax1.set_xlabel('Sample Mean')
    ax1.set_ylabel('Frequency')
    
    if n_samples == 32:
        # 只为n=32绘制正态概率图
        stats.probplot(sample_means, dist="norm", plot=ax2)
        ax2.set_title('Normal probability plot for simulated sample\nmeans, n = 32')
    
    plt.tight_layout()
    plt.show()

# 创建统计汇总表
df_stats = pd.DataFrame(stats_data)
# 创建一个格式化的表格
fig, ax = plt.subplots(figsize=(10, 2))
ax.axis('tight')
ax.axis('off')
table = ax.table(cellText=df_stats.values.round(4),
                colLabels=df_stats.columns,
                cellLoc='center',
                loc='center')
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.2, 1.5)
plt.title("Table 1: Summary statistics for sampling distributions", pad=20)
plt.show()

# 打印理论值
theoretical_mean = (left + mode + right) / 3
theoretical_variance = (left**2 + mode**2 + right**2 - left*mode - left*right - mode*right) / 18
print(f'\nTheoretical Values:')
print(f'Population Mean: {theoretical_mean:.4f}')
print(f'Population Standard Deviation: {np.sqrt(theoretical_variance):.4f}')