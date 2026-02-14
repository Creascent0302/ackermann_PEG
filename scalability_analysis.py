"""
可扩展性分析脚本 - 绘制折线图展示算法随顶点数变化的性能
支持多seed的均值和标准差可视化
"""

import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import numpy as np
import os

# 读取数据
data = pd.read_csv('./pursuer_strategies/PRM/results/scalability_evaluation.csv')

# 设置颜色方案
colors = {
    'delta': '#1f77b4',   # 蓝色
    'beam': '#ff7f0e',    # 橙色
    'spars': '#2ca02c'    # 绿色
}

# 设置图形样式
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12

# 创建输出目录
charts_dir = './pursuer_strategies/PRM/results/charts'
os.makedirs(charts_dir, exist_ok=True)

# 定义要绘制的指标（共8个，2×4布局）
metrics_config = [
    {
        'column': 'generation_time',
        'title': 'PRM Generation Time vs Number of Nodes',
        'ylabel': 'Generation Time (s)',
        'filename': 'scalability_generation_time.png'
    },
    {
        'column': 'search_time',
        'title': 'Path Search Time vs Number of Nodes',
        'ylabel': 'Search Time (s)',
        'filename': 'scalability_search_time.png',
        'filter_success': True  # 只统计有路径的数据
    },
    {
        'column': 'path_length',
        'title': 'Average Path Length vs Number of Nodes',
        'ylabel': 'Path Length',
        'filename': 'scalability_path_length.png',
        'filter_success': True
    },
    {
        'column': 'path_nodes_count',
        'title': 'Path Node Count vs Number of Nodes',
        'ylabel': 'Number of Nodes in Path',
        'filename': 'scalability_path_nodes.png',
        'filter_success': True
    },
    {
        'column': 'edges_count',
        'title': 'Number of Edges vs Number of Nodes',
        'ylabel': 'Number of Edges',
        'filename': 'scalability_edges_count.png'
    },
    {
        'column': 'path_success',
        'title': 'Path Planning Success Rate vs Number of Nodes',
        'ylabel': 'Success Rate (%)',
        'filename': 'scalability_success_rate.png',
        'is_rate': True  # path_success 现在直接存的是百分比
    },
    {
        'column': 'dispersion',
        'title': 'Dispersion vs Number of Nodes',
        'ylabel': 'Dispersion',
        'filename': 'scalability_dispersion.png'
    },
    {
        'column': 'node_utilization',
        'title': 'Node Utilization vs Number of Nodes',
        'ylabel': 'Node Utilization',
        'filename': 'scalability_node_utilization.png',
        'is_ratio': True  # 0-1 的比例值
    }
]

# 算法和环境列表
algorithms = ['delta', 'beam', 'spars']
environments = data['environment'].unique()

print("开始生成可扩展性分析图表...")
print(f"检测到的环境: {list(environments)}")
print(f"检测到的算法: {list(data['algorithm'].unique())}")

# 为每个环境创建一组图表
for env in environments:
    print(f"\n正在处理环境: {env}")
    env_data = data[data['environment'] == env]
    
    # 创建图形 - 2行4列
    fig, axes = plt.subplots(2, 4, figsize=(24, 12))
    fig.suptitle(f'Algorithm Scalability Analysis - {env.capitalize()} Environment', 
                 fontsize=20, fontweight='bold', y=0.995)
    
    for idx, metric_config in enumerate(metrics_config):
        row = idx // 4
        col = idx % 4
        ax = axes[row, col]
        
        metric_col = metric_config['column']
        title = metric_config['title']
        ylabel = metric_config['ylabel']
        filter_success = metric_config.get('filter_success', False)
        is_rate = metric_config.get('is_rate', False)
        
        # 对每个算法绘制曲线
        for algo in algorithms:
            algo_env_data = env_data[env_data['algorithm'] == algo]
            
            if filter_success:
                # 只统计有成功路径的数据（path_success > 0表示至少有路径成功）
                algo_env_data = algo_env_data[algo_env_data['path_success'] > 0]
            
            if len(algo_env_data) == 0:
                print(f"  警告: {algo} 在 {env} 环境的 {metric_col} 没有数据")
                continue
            
            # 按顶点数分组，计算均值和标准差
            grouped = algo_env_data.groupby('num_nodes').agg({
                metric_col: ['mean', 'std', 'count']
            }).reset_index()
            grouped.columns = ['num_nodes', 'mean', 'std', 'count']
            # std为NaN时（只有1个数据点）填充为0
            grouped['std'] = grouped['std'].fillna(0)
            
            # 提取数据
            x = grouped['num_nodes'].values
            y = grouped['mean'].values
            std = grouped['std'].values
            
            # 绘制主曲线
            line = ax.plot(x, y, marker='o', linewidth=2, markersize=6,
                          label=algo.upper(), color=colors[algo])
            
            # 绘制标准差阴影 - 只在有多个数据点时绘制，并裁剪到合理范围
            is_ratio = metric_config.get('is_ratio', False)
            if is_rate:
                y_lower = np.clip(y - std, 0, 100)
                y_upper = np.clip(y + std, 0, 100)
            elif is_ratio:
                y_lower = np.clip(y - std, 0, 1)
                y_upper = np.clip(y + std, 0, 1)
            else:
                y_lower = np.maximum(y - std, 0)  # 不低于0
                y_upper = y + std
            ax.fill_between(x, y_lower, y_upper, 
                           color=colors[algo], alpha=0.2)
        
        # 设置标签和标题
        ax.set_xlabel('Number of Nodes', fontweight='bold')
        ax.set_ylabel(ylabel, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='best', frameon=True, fancybox=True, shadow=True)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # 设置合理的y轴范围
        is_ratio = metric_config.get('is_ratio', False)
        if is_rate:
            ax.set_ylim(0, 105)  # 成功率0-100%
        elif is_ratio:
            ax.set_ylim(bottom=0)  # 利用率 0-1
        else:
            ax.set_ylim(bottom=0)
    
    # 调整布局并保存
    output_file = os.path.join(charts_dir, f'scalability_analysis_{env}.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  已保存: {output_file}")
    plt.close()

# 创建综合对比图 - 所有环境在一起
print("\n正在生成综合对比图...")

for metric_config in metrics_config:
    metric_col = metric_config['column']
    title = metric_config['title']
    ylabel = metric_config['ylabel']
    filename = metric_config['filename']
    filter_success = metric_config.get('filter_success', False)
    is_rate = metric_config.get('is_rate', False)
    
    # 创建图形 - 1行3列，每列一个环境
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(title, fontsize=18, fontweight='bold', y=1.02)
    
    for env_idx, env in enumerate(environments):
        ax = axes[env_idx]
        env_data = data[data['environment'] == env]
        
        # 对每个算法绘制曲线
        for algo in algorithms:
            algo_env_data = env_data[env_data['algorithm'] == algo]
            
            if filter_success:
                algo_env_data = algo_env_data[algo_env_data['path_success'] > 0]
            
            if len(algo_env_data) == 0:
                continue
            
            # 按顶点数分组，计算均值和标准差
            grouped = algo_env_data.groupby('num_nodes').agg({
                metric_col: ['mean', 'std', 'count']
            }).reset_index()
            grouped.columns = ['num_nodes', 'mean', 'std', 'count']
            grouped['std'] = grouped['std'].fillna(0)
            
            x = grouped['num_nodes'].values
            y = grouped['mean'].values
            std = grouped['std'].values
            
            # 绘制主曲线
            ax.plot(x, y, marker='o', linewidth=2, markersize=6,
                   label=algo.upper(), color=colors[algo])
            
            # 绘制标准差阴影 - 裁剪到合理范围
            is_ratio = metric_config.get('is_ratio', False)
            if is_rate:
                y_lower = np.clip(y - std, 0, 100)
                y_upper = np.clip(y + std, 0, 100)
            elif is_ratio:
                y_lower = np.clip(y - std, 0, 1)
                y_upper = np.clip(y + std, 0, 1)
            else:
                y_lower = np.maximum(y - std, 0)
                y_upper = y + std
            ax.fill_between(x, y_lower, y_upper,
                           color=colors[algo], alpha=0.2)
        
        # 设置标签和标题
        ax.set_xlabel('Number of Nodes', fontweight='bold')
        if env_idx == 0:
            ax.set_ylabel(ylabel, fontweight='bold')
        ax.set_title(f'{env.capitalize()} Environment', fontsize=14, fontweight='bold')
        ax.legend(loc='best', frameon=True, fancybox=True, shadow=True)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        if is_rate:
            ax.set_ylim(0, 105)
        elif is_ratio:
            ax.set_ylim(bottom=0)
        else:
            ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    output_file = os.path.join(charts_dir, filename)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"已保存: {output_file}")
    plt.close()

# 打印统计摘要
print("\n" + "="*60)
print("统计摘要")
print("="*60)

for env in environments:
    print(f"\n{env.upper()} Environment:")
    env_data = data[data['environment'] == env]
    
    for algo in algorithms:
        algo_data = env_data[env_data['algorithm'] == algo]
        if len(algo_data) == 0:
            continue
        
        print(f"\n  {algo.upper()}:")
        print(f"    测试次数: {len(algo_data)}")
        print(f"    平均生成时间: {algo_data['generation_time'].mean():.3f}s ± {algo_data['generation_time'].std():.3f}s")
        print(f"    平均边数: {algo_data['edges_count'].mean():.1f} ± {algo_data['edges_count'].std():.1f}")
        
        success_data = algo_data[algo_data['path_success'] == True]
        if len(success_data) > 0:
            success_rate = len(success_data) / len(algo_data) * 100
            print(f"    路径成功率: {success_rate:.1f}%")
            print(f"    平均路径长度: {success_data['path_length'].mean():.3f} ± {success_data['path_length'].std():.3f}")
            print(f"    平均搜索时间: {success_data['search_time'].mean():.4f}s ± {success_data['search_time'].std():.4f}s")

print("\n" + "="*60)
print("所有图表已保存到:", charts_dir)
print("="*60)
