import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# 设置字体为Times New Roman并增大字体大小
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 26          
plt.rcParams['axes.labelsize'] = 26     
plt.rcParams['axes.titlesize'] = 26     
plt.rcParams['xtick.labelsize'] = 26    
plt.rcParams['ytick.labelsize'] = 26    
plt.rcParams['legend.fontsize'] = 22    

# 读取两个数据文件
prm_data = pd.read_csv('./pursuer_strategies/PRM/results/evaluation_summary.csv')
path_data = pd.read_csv('./pursuer_strategies/PRM/results/paths/path_evaluation_summary.csv')

# 设置颜色方案
colors = ['#E38691', '#F5C326', '#BACBA9', "#B7CAD9"]

# 设置图形样式 - 减小高度，使图更扁
fig, axes = plt.subplots(2, 3, figsize=(24, 12))  # 从(24, 16)改为(24, 12)，减小高度

# 定义所有指标的信息 - 按新的顺序排列
metrics_info = [
    # 第一行：从prm_data来的指标
    ('generation_time', 'Generation Time', 'Time (s)', 'prm'),
    ('nodes_count', 'Number of Nodes', 'Node Count', 'prm'),
    ('edges_count', 'Number of Edges', 'Edge Count', 'prm'),
    # 第二行：discrepancy和path_data指标
    ('discrepancy', 'Discrepancy Score', 'Discrepancy', 'prm'),
    ('avg_search_time', 'Average Search Time', 'Time (s)', 'path'),
    ('avg_path_length', 'Average Path Length', 'Path Length', 'path')
]

# 环境和算法列表
environments = ['maze', 'indoor', 'random']
algorithms = ['delta', 'beam', 'spars']

# 预处理prm_data: 创建beam-medial数据
beam_medial_rows = []
for env in environments:
    beam_env_data = prm_data[(prm_data['algorithm'] == 'beam') & (prm_data['environment'] == env)]
    
    if not beam_env_data.empty:
        for _, row in beam_env_data.iterrows():
            beam_medial_rows.append({
                'algorithm': 'beam-medial',
                'environment': env,
                'seed': row['seed'],
                'nodes_count': row['medial_axis_nodes_count'],
                'edges_count': row['medial_axis_edges_count'],
                'discrepancy': row['medial_axis_discrepancy']
            })

beam_medial_df = pd.DataFrame(beam_medial_rows)

# 合并beam-medial到prm数据
enhanced_prm_data = pd.concat([
    prm_data[['algorithm', 'environment', 'seed', 'generation_time', 'nodes_count', 'edges_count', 'discrepancy']],
    beam_medial_df
], ignore_index=True)

# 按算法、环境分组取均值
grouped_prm_data = enhanced_prm_data.groupby(['algorithm', 'environment']).agg({
    'generation_time': 'mean',
    'nodes_count': 'mean',
    'edges_count': 'mean',
    'discrepancy': 'mean'
}).reset_index()

grouped_path_data = path_data.groupby(['algorithm', 'environment']).agg({
    'avg_path_length': 'mean',
    'avg_search_time': 'mean'
}).reset_index()

print("=== PRM Data Summary ===")
print(grouped_prm_data.round(4))
print("\n=== Path Data Summary ===")
print(grouped_path_data.round(4))

# 创建charts文件夹
charts_dir = './pursuer_strategies/PRM/results/charts'
os.makedirs(charts_dir, exist_ok=True)

# 定义算法名称映射
algorithm_name_mapping = {
    'beam': 'B-PRM',
    'beam-medial': 'BS-PRM', 
    'delta': 'δ-PRM',
    'spars': 'SPARS'
}

# 为generation_time单独定义算法名称映射
generation_time_algorithm_mapping = {
    'beam': 'B-PRM & BS-PRM',
    'delta': 'δ-PRM',
    'spars': 'SPARS'
}

# 添加环境名称映射
environment_name_mapping = {
    'maze': 'Maze',
    'indoor': 'Indoor',
    'random': 'Cluttered'
}

# 为每个指标画图
for idx, (metric, title, ylabel, data_source) in enumerate(metrics_info):
    row = idx // 3  
    col = idx % 3   
    ax = axes[row, col]
    
    # 生成子图标识 (a, b, c, d, e, f)
    subplot_label = chr(ord('a') + idx)
    
    # 选择数据源
    if data_source == 'prm':
        grouped_data = grouped_prm_data
        # 对于需要beam-medial的指标
        if metric in ['discrepancy', 'nodes_count', 'edges_count']:
            algorithms_to_use = ['beam', 'beam-medial', 'delta', 'spars']
            color_indices = [0, 1, 2, 3]  
        else:
            algorithms_to_use = ['beam', 'delta', 'spars']
            color_indices = [0, 2, 3] 
    else:  # path data 
        grouped_data = grouped_path_data
        algorithms_to_use = ['beam', 'delta', 'spars']
        color_indices = [1, 2, 3]
        
        # 为path data创建特殊的算法名称映射
        path_algorithm_mapping = {
            'beam': 'BS-PRM',
            'delta': 'δ-PRM',
            'spars': 'SPARS'
        }

    # 准备数据
    env_data = []
    for env in environments:
        env_subset = grouped_data[grouped_data['environment'] == env]
        metric_values = []
        for algo in algorithms_to_use:
            algo_data = env_subset[env_subset['algorithm'] == algo][metric]
            if not algo_data.empty:
                metric_values.append(algo_data.iloc[0])
            else:
                metric_values.append(0)
        env_data.append(metric_values)
    
    # 设置柱状图位置
    x = np.arange(len(environments))
    width = 0.18 if len(algorithms_to_use) == 4 else 0.25
    
    # 画柱状图
    for i, (algo, color_idx) in enumerate(zip(algorithms_to_use, color_indices)):
        values = [env_data[j][i] for j in range(len(environments))]
        
        # 根据指标和数据源选择不同的名称映射
        if metric == 'generation_time':
            display_name = generation_time_algorithm_mapping.get(algo, algo.upper())
        elif data_source == 'path':
            display_name = path_algorithm_mapping.get(algo, algo.upper())
        else:
            display_name = algorithm_name_mapping.get(algo, algo.upper())
            
        bars = ax.bar(x + i * width, values, width,
                      label=display_name, color=colors[color_idx], alpha=0.8,
                      edgecolor='white', linewidth=0.5)
        
    # 设置图形属性
    ax.set_xlabel(None, fontfamily='Times New Roman')
    ax.set_ylabel(ylabel, fontfamily='Times New Roman')
    ax.set_title(title, fontfamily='Times New Roman')
    
    # 调整x轴标签位置
    if len(algorithms_to_use) == 4:
        ax.set_xticks(x + 1.5 * width)
    else:
        ax.set_xticks(x + width)
    ax.set_xticklabels([environment_name_mapping.get(env, env.capitalize()) for env in environments])
    
    # 设置刻度标签字体
    for label in ax.get_xticklabels():
        label.set_fontfamily('Times New Roman')
    for label in ax.get_yticklabels():
        label.set_fontfamily('Times New Roman')

    # 调整图例位置 - 防止遮挡柱形图
    if metric in ['generation_time', 'nodes_count', 'edges_count', 'discrepancy']:
        # 对于时间类指标，图例放在左上角
        legend = ax.legend(frameon=True, fancybox=True, shadow=True, 
                          loc='upper left', bbox_to_anchor=(0.02, 0.98))
    else:
        # 对于其他指标，图例放在右上角
        legend = ax.legend(frameon=True, fancybox=True, shadow=True, 
                          loc='upper right', bbox_to_anchor=(0.98, 0.98))
    
    # 设置图例字体
    for text in legend.get_texts():
        text.set_fontfamily('Times New Roman')
        
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim(bottom=0)
    
    # 添加子图标识 (a), (b), (c), (d), (e), (f) 在图的正下方
    ax.text(0.5, -0.1, f'({subplot_label})', 
            transform=ax.transAxes, 
            fontsize=26,  
            fontfamily='Times New Roman',
            verticalalignment='top',
            horizontalalignment='center')

# 调整布局 - 减小间距使图更紧凑
plt.tight_layout(pad=1.0, h_pad=2.0, w_pad=1.0)  # 减小pad从2.0到1.0，减小h_pad
plt.subplots_adjust(bottom=0.08, hspace=0.3, wspace=0.25)  # 减小hspace行间距，减小bottom

plt.savefig(os.path.join(charts_dir, 'comprehensive_analysis.pdf'), 
            dpi=300, bbox_inches='tight')

print(f"\n图像已保存到: {charts_dir}/comprehensive_analysis.pdf")