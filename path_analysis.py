import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import numpy as np

# 读取数据
data = pd.read_csv('./pursuer_strategies/PRM/results/paths/path_evaluation_summary.csv')

# 设置颜色方案（与analysis文件一致的颜色）
colors = {
    'delta': '#1f77b4',   # 蓝色
    'beam': '#ff7f0e',    # 橙色
    'spars': '#2ca02c'    # 绿色
}

# 设置图形样式
plt.rcParams.update({'font.size': 10})
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# 指标列表和标题
metrics = ['prm_gen_time', 'avg_path_length', 'avg_search_time']
titles = ['PRM Generation Time', 'Average Path Length', 'Average Search Time']
y_labels = ['Time (s)', 'Path Length', 'Time (s)']

# 环境和算法列表
environments = ['maze', 'indoor', 'random']
algorithms = ['delta', 'beam', 'spars']

# 按算法和环境分组，计算各指标的平均值
grouped_data = data.groupby(['algorithm', 'environment']).agg({
    'prm_gen_time': 'mean',
    'avg_path_length': 'mean',
    'avg_search_time': 'mean',
    'success_rate': 'mean'
}).reset_index()

print("=== Grouped Data Summary ===")
print(grouped_data.round(4))

# 在设置x轴标签的部分，添加环境名称映射
environment_name_mapping = {
    'maze': 'Maze',
    'indoor': 'Indoor', 
    'random': 'Cluttered'  # 将random映射为Cluttered
}

# 为每个指标画图
for idx, (metric, title, ylabel) in enumerate(zip(metrics, titles, y_labels)):
    ax = axes[idx]
    
    # 准备数据
    env_data = []
    for env in environments:
        env_subset = grouped_data[grouped_data['environment'] == env]
        metric_values = []
        for algo in algorithms:
            algo_data = env_subset[env_subset['algorithm'] == algo][metric]
            if not algo_data.empty:
                metric_values.append(algo_data.iloc[0])
            else:
                metric_values.append(0)  # 如果没有数据，填充0
        env_data.append(metric_values)
    
    # 设置柱状图位置
    x = np.arange(len(environments))
    width = 0.25
    
    # 画柱状图
    for i, algo in enumerate(algorithms):
        values = [env_data[j][i] for j in range(len(environments))]
        bars = ax.bar(x + i * width, values, width, 
                     label=algo.upper(), color=colors[algo], alpha=0.8,
                     edgecolor='black', linewidth=0.5)
        
        # 在柱子上添加数值标签
        for bar in bars:
            height = bar.get_height()
            if height > 0:  # 只有当高度大于0时才添加标签
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    # 设置图形属性
    ax.set_xlabel('Environment', fontweight='bold')
    ax.set_ylabel(ylabel, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # 调整x轴标签位置
    if len(algorithms) == 4:
        ax.set_xticks(x + 1.5 * width)
    else:
        ax.set_xticks(x + width)
    
    # 修改这一行，使用映射而不是直接capitalize
    ax.set_xticklabels([environment_name_mapping.get(env, env.capitalize()) for env in environments])
    ax.legend(frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # 设置y轴从0开始
    ax.set_ylim(bottom=0)

# 调整布局
plt.tight_layout()
plt.savefig('pursuer_strategies/PRM/results/paths/path_metrics_analysis.pdf', 
            dpi=300, bbox_inches='tight')
plt.show()

# 打印详细统计摘要
print("\n=== Detailed Statistics by Algorithm and Environment ===")
for env in environments:
    print(f"\n{env.upper()} Environment:")
    env_data = grouped_data[grouped_data['environment'] == env]
    for _, row in env_data.iterrows():
        print(f"  {row['algorithm'].upper()}:")
        print(f"    PRM Gen Time: {row['prm_gen_time']:.4f}s")
        print(f"    Avg Path Length: {row['avg_path_length']:.4f}")
        print(f"    Avg Search Time: {row['avg_search_time']:.4f}s")
        print(f"    Success Rate: {row['success_rate']:.2%}")

# 计算各指标的最佳表现
print("\n=== Best Performance by Metric ===")
for metric, title in zip(metrics, titles):
    print(f"\n{title}:")
    for env in environments:
        env_subset = grouped_data[grouped_data['environment'] == env]
        if 'time' in metric:  # 对于时间指标，越小越好
            best_row = env_subset.loc[env_subset[metric].idxmin()]
        else:  # 对于路径长度，也是越小越好
            best_row = env_subset.loc[env_subset[metric].idxmin()]
        print(f"  {env.capitalize()}: {best_row['algorithm'].upper()} ({best_row[metric]:.4f})")

# 计算各环境下算法的综合排名（基于标准化得分）
print("\n=== Algorithm Ranking by Environment (Lower is Better) ===")
for env in environments:
    env_subset = grouped_data[grouped_data['environment'] == env].copy()
    
    # 标准化各指标（转换为0-1范围，越小越好）
    for metric in metrics:
        max_val = env_subset[metric].max()
        min_val = env_subset[metric].min()
        if max_val != min_val:
            env_subset[f'{metric}_norm'] = (env_subset[metric] - min_val) / (max_val - min_val)
        else:
            env_subset[f'{metric}_norm'] = 0
    
    # 计算综合得分
    env_subset['composite_score'] = (env_subset['prm_gen_time_norm'] + 
                                   env_subset['avg_path_length_norm'] + 
                                   env_subset['avg_search_time_norm']) / 3
    
    env_subset_sorted = env_subset.sort_values('composite_score')
    
    print(f"\n{env.capitalize()} Environment:")
    for i, (_, row) in enumerate(env_subset_sorted.iterrows(), 1):
        print(f"  {i}. {row['algorithm'].upper()} (Score: {row['composite_score']:.4f})")