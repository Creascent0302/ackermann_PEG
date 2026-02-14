import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import numpy as np
import os

# 读取数据
data = pd.read_csv('./pursuer_strategies/PRM/results/evaluation_summary.csv')

# 设置颜色方案（与path_analysis文件一致的颜色）
colors = {
    'delta': '#1f77b4',   # 蓝色
    'beam': '#ff7f0e',    # 橙色
    'spars': '#2ca02c',    # 绿色
    'beam-medial': '#9467bd'  # 紫色
}

# 设置图形样式
plt.rcParams.update({'font.size': 10})
fig, axes = plt.subplots(2, 4, figsize=(24, 12))  # 2行4列布局

# 指标列表和标题
metrics = [
    ('nodes_count', 'Number of Nodes', 'Node Count'),
    ('edges_count', 'Number of Edges', 'Edge Count'),
    ('dispersion', 'Dispersion Score', 'Dispersion'),
    ('discrepancy', 'Discrepancy Score', 'Discrepancy'),
    ('generation_time', 'Generation Time', 'Time (s)'),
    ('node_utilization', 'Node Utilization', 'Utilization (%)'),
    ('avg_nodes_per_path', 'Avg Nodes Per Path', 'Nodes')
]
# 前四个指标加beam-medial

other_metrics = []

# 环境和算法列表
environments = ['maze', 'indoor', 'random']
algorithms = ['delta', 'beam', 'spars']
algorithms_with_medial = ['delta', 'beam', 'beam-medial', 'spars']

# 处理beam-medial数据 - 修复这里的逻辑
beam_medial_rows = []
for env in environments:
    # 获取该环境下beam算法的所有行
    beam_env_data = data[(data['algorithm'] == 'beam') & (data['environment'] == env)]
    
    if not beam_env_data.empty:
        # 对于每一行beam数据，创建对应的beam-medial行
        for _, row in beam_env_data.iterrows():
            beam_medial_rows.append({
                'algorithm': 'beam-medial',
                'environment': env,
                'seed': row['seed'],  # 保持原始seed值（包括"N/A"）
                'nodes_count': row['medial_axis_nodes_count'],
                'edges_count': row['medial_axis_edges_count'],
                'dispersion': row['medial_axis_dispersion'],
                'discrepancy': row['medial_axis_discrepancy']
            })

beam_medial_df = pd.DataFrame(beam_medial_rows)

print("=== Beam-Medial Data ===")
print(beam_medial_df)

# 合并beam-medial到原始数据
plot_data = pd.concat([
    data[['algorithm', 'environment', 'seed', 'nodes_count', 'edges_count', 'dispersion', 'discrepancy', 'generation_time', 'node_utilization', 'avg_nodes_per_path']],
    beam_medial_df
], ignore_index=True)

# 按算法、环境分组取均值
grouped_data = plot_data.groupby(['algorithm', 'environment']).agg({
    'nodes_count': 'mean',
    'edges_count': 'mean',
    'dispersion': 'mean',
    'discrepancy': 'mean',
    'generation_time': 'mean',
    'node_utilization': 'mean',
    'avg_nodes_per_path': 'mean'
}).reset_index()

print("\n=== Grouped Data Summary ===")
print(grouped_data.round(4))

charts_dir = './pursuer_strategies/PRM/results/charts'
os.makedirs(charts_dir, exist_ok=True)

# 画所有指标
for idx, (metric, title, ylabel) in enumerate(metrics):
    if idx >= 8:  # 最多8个子图
        break
    row = idx // 4
    col = idx % 4
    ax = axes[row, col]
    
    # 前4个指标使用 beam-medial，后面的指标不使用
    use_medial = idx < 4
    current_algos = algorithms_with_medial if use_medial else algorithms
    
    env_data = []
    for env in environments:
        env_subset = grouped_data[grouped_data['environment'] == env]
        metric_values = []
        for algo in current_algos:
            algo_data = env_subset[env_subset['algorithm'] == algo][metric]
            if not algo_data.empty:
                value = algo_data.iloc[0]
                # 节点利用率转换为百分比
                if metric == 'node_utilization':
                    value = value * 100
                metric_values.append(value)
            else:
                metric_values.append(0)
        env_data.append(metric_values)
    
    x = np.arange(len(environments))
    width = 0.18 if use_medial else 0.25
    
    for i, algo in enumerate(current_algos):
        values = [env_data[j][i] for j in range(len(environments))]
        bars = ax.bar(x + i * width, values, width,
                      label=algo.upper(), color=colors.get(algo, '#888888'), alpha=0.8,
                      edgecolor='black', linewidth=0.5)
        
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                if metric in ['dispersion', 'discrepancy']:
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                            f'{height:.4f}', ha='center', va='bottom', fontsize=8)
                elif metric in ['node_utilization', 'avg_nodes_per_path']:
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                            f'{height:.2f}', ha='center', va='bottom', fontsize=8)
                else:
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                            f'{int(height)}', ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('Environment', fontweight='bold')
    ax.set_ylabel(ylabel, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    offset = 1.5 * width if use_medial else width
    ax.set_xticks(x + offset)
    ax.set_xticklabels([env.capitalize() for env in environments])
    ax.legend(frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim(bottom=0)

# 隐藏未使用的子图（如果有的话）
if len(metrics) < 8:
    for idx in range(len(metrics), 8):
        row = idx // 4
        col = idx % 4
        axes[row, col].axis('off')

# 调整布局
plt.tight_layout()
plt.savefig(os.path.join(charts_dir, 'prm_metrics_analysis.png'), 
            dpi=300, bbox_inches='tight')
plt.show()

# 打印详细统计摘要
print("\n=== Detailed Statistics by Algorithm and Environment ===")
for env in environments:
    print(f"\n{env.upper()} Environment:")
    env_data = grouped_data[grouped_data['environment'] == env]
    for _, row in env_data.iterrows():
        print(f"  {row['algorithm'].upper()}:")
        print(f"    Nodes Count: {int(row['nodes_count'])}")
        print(f"    Edges Count: {int(row['edges_count'])}")
        print(f"    Dispersion: {row['dispersion']:.4f}")
        print(f"    Discrepancy: {row['discrepancy']:.4f}")
        if 'generation_time' in row:
            print(f"    Generation Time: {row['generation_time']:.4f}s")
        if 'node_utilization' in row:
            print(f"    Node Utilization: {row['node_utilization']*100:.2f}%")
        if 'avg_nodes_per_path' in row:
            print(f"    Avg Nodes Per Path: {row['avg_nodes_per_path']:.2f}")

print(f"\n图像已保存到: {charts_dir}/prm_metrics_analysis.png")