import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 设置随机种子以确保结果可重现
np.random.seed(42)

# 定义参数
sample_sizes = [2, 4, 8, 16, 32]  # 不同的样本大小
n_simulations = 500  # 模拟次数
left = 0    # 三角分布的最小值
mode = 2    # 三角分布的众数
right = 4   # 三角分布的最大值

# 为每个样本大小创建单独的图形
for i, n_samples in enumerate(sample_sizes, 1):
    # 创建新的图形
    plt.figure(figsize=(8, 6))
    
    # 进行多次抽样并计算每次抽样的均值
    sample_means = []
    for _ in range(n_simulations):
        samples = np.random.triangular(left, mode, right, size=n_samples)
        sample_means.append(np.mean(samples))
    
    # 绘制抽样均值的分布
    sns.histplot(sample_means, kde=True)
    plt.title(f'Sample Size n = {n_samples}')
    plt.xlabel('Sample Mean')
    plt.ylabel('Frequency')
    
    # 计算并显示统计量
    mean = np.mean(sample_means)
    std = np.std(sample_means)
    skewness = np.mean(((sample_means - mean)/std)**3)
    
    # 在图上添加统计信息
    stats_text = f'Mean: {mean:.3f}\nStd Dev: {std:.3f}\nSkewness: {skewness:.3f}'
    plt.text(0.95, 0.95, stats_text,
             transform=plt.gca().transAxes,
             verticalalignment='top',
             horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # 添加总标题
    plt.suptitle(f'Sampling Distribution\nTriangular Distribution (min=0, mode=2, max=4)', 
                 y=1.02, fontsize=12)
    
    plt.tight_layout()
    plt.show()

# 打印理论值
theoretical_mean = (left + mode + right) / 3
theoretical_variance = (left**2 + mode**2 + right**2 - left*mode - left*right - mode*right) / 18
print(f'\nTheoretical Values:')
print(f'Population Mean: {theoretical_mean:.4f}')
print(f'Population Standard Deviation: {np.sqrt(theoretical_variance):.4f}')

# 创建比较表格
print('\nComparison of Sample Sizes:')
print('Sample Size\tMean\t\tStd Dev\t\tSkewness')
print('-' * 60)
for n_samples in sample_sizes:
    sample_means = [np.mean(np.random.triangular(left, mode, right, size=n_samples)) 
                   for _ in range(n_simulations)]
    mean = np.mean(sample_means)
    std = np.std(sample_means)
    skewness = np.mean(((sample_means - mean)/std)**3)
    print(f'{n_samples}\t{mean:.4f}\t\t{std:.4f}\t\t{skewness:.4f}') 