import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# 设置输出目录
output_dir = "./results/charts"
os.makedirs(output_dir, exist_ok=True)

# 要分析的指标
metrics = ['generation_time', 'nodes_count', 'edges_count', 'beam_connectivity_score']

# 要处理的CSV文件列表
files = [
    "./pursuer_strategies/PRM/results/metrics_indoor.csv",
    "./pursuer_strategies/PRM/results/metrics_random.csv",
    "./pursuer_strategies/PRM/results/metrics_maze.csv"
]

# 指标显示名称映射
metric_display = {
    'generation_time': 'Generation time(s)',
    'nodes_count': 'Number of nodes',
    'edges_count': 'Number of edges',
    'beam_connectivity_score': 'Connectivity score'
}

# 算法名称映射
algorithm_display = {
    'classical': 'Classical PRM',
    'star': 'PRM*',
    'beam': 'Beam-PRM',
    'spars': 'SPARS'
}

# 设置颜色
colors = ['#4E79A7', '#F28E2B', '#E15759', '#76B7B2']

# 处理每个文件
for file_path in files:
    # 从文件路径中提取环境名称
    env_name = os.path.basename(file_path).split('_')[1].split('.')[0]
    
    # 读取CSV文件
    df = pd.read_csv(file_path)
    
    # 获取不同的target_nodes值（采样次数）
    target_nodes = sorted(df['target_nodes'].unique())
    
    # 获取所有算法
    algorithms = ['classical', 'star', 'beam', 'spars']
    
    # 为每个指标创建一个柱状图
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        
        # 设置条形图的宽度和位置
        bar_width = 0.2
        x = np.arange(len(target_nodes))
        
        # 为每种算法绘制条形
        for i, algorithm in enumerate(algorithms):
            # 收集该算法在不同target_nodes下的指标值
            values = []
            for node in target_nodes:
                # 计算该算法在特定target_nodes下的平均指标值
                subset = df[(df['target_nodes'] == node) & (df['algorithm'] == algorithm)]
                if not subset.empty:
                    values.append(subset[metric].mean())
                else:
                    values.append(0)
            
            # 绘制条形
            plt.bar(x + i*bar_width, values, bar_width, 
                    label=algorithm_display[algorithm], 
                    color=colors[i],
                    edgecolor='black',
                    linewidth=0.5)
        
        # 添加数值标签（仅对部分重要指标）
        if metric in ['generation_time', 'beam_connectivity_score']:
            for i, algorithm in enumerate(algorithms):
                values = []
                for node in target_nodes:
                    subset = df[(df['target_nodes'] == node) & (df['algorithm'] == algorithm)]
                    if not subset.empty:
                        values.append(subset[metric].mean())
                    else:
                        values.append(0)
                
                for j, v in enumerate(values):
                    if v > 0:  # 只标注非零值
                        if metric == 'generation_time':
                            # 时间值保留2位小数
                            plt.text(x[j] + i*bar_width, v + 0.1, f"{v:.2f}", 
                                    ha='center', va='bottom', fontsize=8, rotation=0)
                        elif metric == 'beam_connectivity_score':
                            # 评分保留2位小数
                            plt.text(x[j] + i*bar_width, v + 0.01, f"{v:.2f}", 
                                    ha='center', va='bottom', fontsize=8, rotation=0)
        
        # 设置图表标题和标签
        env_display = {"indoor": "Indoor", "random": "Random", "maze": "Maze"}
        plt.title(f'{env_display.get(env_name, env_name)} - {metric_display[metric]}', fontsize=14)
        plt.xlabel('Number of samples', fontsize=12)
        plt.ylabel(metric_display[metric], fontsize=12)
        
        # 设置x轴刻度
        plt.xticks(x + bar_width*1.5, target_nodes)
        
        # 添加图例
        plt.legend(loc='best')
        
        # 添加网格线便于阅读
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图表
        output_file = os.path.join(output_dir, f'{env_name}_{metric}.pdf')
        plt.savefig(output_file, dpi=300)
        print(f"保存图表: {output_file}")
        plt.close()

print(f"\n所有图表已保存到: {output_dir}")