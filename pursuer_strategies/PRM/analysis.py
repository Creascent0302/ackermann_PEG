import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# 设置输出目录
output_dir = "./pursuer_strategies/PRM/results/charts"
os.makedirs(output_dir, exist_ok=True)

# 要分析的指标
metrics = ['generation_time', 'nodes_count', 'edges_count', 'dispersion', 'discrepancy']

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
    'dispersion': 'Dispersion score',
    'discrepancy': 'Discrepancy score'
}

# 算法名称映射
algorithm_display = {
    'classical': 'Classical PRM',
    'star': 'PRM*',
    'beam': 'Beam-PRM',
    'spars': 'SPARS'
}

# 设置颜色
colors = ['#E38691', '#F5C326', '#BACBA9', "#B7CAD9"]

# 处理每个文件
for file_path in files:
    # 从文件路径中提取环境名称
    env_name = os.path.basename(file_path).split('_')[1].split('.')[0]
    df = pd.read_csv(file_path)
    
    if df.empty:
        print(f"警告: {file_path} 为空，跳过")
        continue
    
    print(f"处理 {env_name} 环境")
    
    # 获取所有算法
    algorithms = ['classical', 'star', 'beam', 'spars']
    
    # 环境显示名称
    env_display = {"indoor": "Indoor", "random": "Random", "maze": "Maze"}
    
    # 为每个指标创建一个横向柱状图
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        
        # 计算每个算法的平均指标值
        values = []
        for algorithm in algorithms:
            # 筛选该算法的所有数据
            subset = df[df['algorithm'] == algorithm]
            
            if not subset.empty:
                # 计算平均值
                avg_value = subset[metric].mean()
                values.append(avg_value)
                print(f"  {env_name}, {algorithm}, {metric}平均值: {avg_value:.4f} (来自{len(subset)}个样本)")
            else:
                print(f"  警告: {env_name}, {algorithm} 没有数据")
                values.append(0)
        
        # 设置y轴位置
        y_pos = np.arange(len(algorithms))
        
        # 绘制横向条形图
        bars = plt.barh(y_pos, values, color=colors[:len(algorithms)])
        
        # 设置x轴范围，根据指标值稍微扩展一些
        x_max = max(values) * 1.2  # 扩展20%的空间用于显示标签
        plt.xlim(0, x_max)
        
        # 添加数值标签
        for i, v in enumerate(values):
            if v > 0:  # 只标注非零值
                label_offset = x_max * 0.01  # 根据x轴范围调整标签偏移
                if metric == 'generation_time':
                    # 时间值保留2位小数
                    plt.text(v + label_offset, i, f"{v:.2f}", 
                            va='center', fontsize=9)
                elif metric in ['dispersion', 'discrepancy']:
                    # 评分保留4位小数
                    plt.text(v + label_offset, i, f"{v:.4f}", 
                            va='center', fontsize=9)
                elif metric in ['nodes_count', 'edges_count']:
                    plt.text(v + label_offset, i, f"{int(v)}", 
                            va='center', fontsize=9)
        
        # 设置图表标题和标签
        plt.title(f'{env_display.get(env_name, env_name)} - {metric_display[metric]}', fontsize=14)
        plt.xlabel(metric_display[metric], fontsize=12)
        
        # 设置y轴刻度
        plt.yticks(y_pos, [algorithm_display[alg] for alg in algorithms])
        
        # 添加网格线便于阅读
        plt.grid(axis='x', linestyle=':', alpha=0.7)
        
        # 去除上边框和右边框
        ax = plt.gca()  # 获取当前坐标轴
        ax.spines['top'].set_visible(False)  # 隐藏上边框
        ax.spines['right'].set_visible(False)  # 隐藏右边框
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图表
        output_file = os.path.join(output_dir, f'{env_name}_{metric}.pdf')
        plt.savefig(output_file, dpi=300)
        print(f"保存图表: {output_file}")
        plt.close()

print(f"\n所有图表已保存到: {output_dir}")