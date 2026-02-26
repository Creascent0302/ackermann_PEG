"""
可扩展性分析脚本 - 绘制 6x4 综合对比 SVG 大图
展示算法随采样次数变化的性能 (支持均值和标准差可视化)
(仅保留 4 种核心算法: delta, beam, spars, gsrm)
"""

import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import numpy as np
import os

# ─────────────────────────────────────────────
# 1. 基础配置
# ─────────────────────────────────────────────
# ⚠️ 注意: 运行此脚本前，请将此路径修改为你实际跑完的 CSV 文件名
DATA_FILE = './pursuer_strategies/PRM/results/scalability_evaluation_202602230628.csv' 
data = pd.read_csv(DATA_FILE)

# 颜色方案与线型（4种算法）
colors = {
    'delta': '#1f77b4',   # 蓝色
    'beam':  '#ff7f0e',   # 橙色
    'spars': '#2ca02c',   # 绿色
    'gsrm':  '#d62728',   # 红色
}

linestyles = {
    'delta': '-',
    'beam':  '-',
    'spars': '--',
    'gsrm':  '-.',
}

markers = {
    'delta': 'o',
    'beam':  's',
    'spars': '^',
    'gsrm':  'D',
}

algorithms = ['delta', 'beam', 'spars', 'gsrm']

# 全局图形样式
plt.rcParams['font.family']      = 'Times New Roman'
plt.rcParams['font.size']        = 13
plt.rcParams['axes.labelsize']   = 15
plt.rcParams['axes.titlesize']   = 16
plt.rcParams['xtick.labelsize']  = 13
plt.rcParams['ytick.labelsize']  = 13
plt.rcParams['legend.fontsize']  = 14

charts_dir = './pursuer_strategies/PRM/results/charts'
os.makedirs(charts_dir, exist_ok=True)

# 图表列布局: Random, Maze, Indoor, All(Combined)
environments = ['random', 'maze', 'indoor', 'all'] 

# 精选 6 个最具有代表性的核心指标作为行
selected_metrics_config = [
    {'column': 'path_success', 'title': 'Path Success Rate (%)', 'is_rate': True},
    {'column': 'generation_time', 'title': 'Generation Time (s)', 'filter_success': False},
    {'column': 'path_length', 'title': 'Average Path Length', 'filter_success': True},
    {'column': 'search_time', 'title': 'Path Search Time (s)', 'filter_success': True},
    {'column': 'dispersion', 'title': 'Coverage Dispersion', 'filter_success': False},
    {'column': 'clearance', 'title': 'Path Clearance (Safety)', 'filter_success': True},
]

# ─────────────────────────────────────────────
# 2. 辅助绘图函数
# ─────────────────────────────────────────────
def plot_metric_on_ax(ax, algo_data, algo, metric_config):
    """在 ax 上绘制指定算法的折线 + 标准差阴影"""
    metric_col = metric_config['column']
    if metric_config.get('filter_success', False):
        algo_data = algo_data[algo_data['path_success'] > 0]
    
    if len(algo_data) == 0: return False

    # 按采样次数计算均值和标准差
    grouped = algo_data.groupby('num_samples')[metric_col].agg(['mean', 'std']).reset_index()
    grouped['std'] = grouped['std'].fillna(0)

    x, y, std = grouped['num_samples'].values, grouped['mean'].values, grouped['std'].values

    ax.plot(x, y, marker=markers[algo], linestyle=linestyles[algo], 
            linewidth=2.5, markersize=7, label=algo.upper(), color=colors[algo])

    if metric_config.get('is_rate', False):
        y_lower, y_upper = np.clip(y - std, 0, 100), np.clip(y + std, 0, 100)
    elif metric_config.get('is_ratio', False):
        y_lower, y_upper = np.clip(y - std, 0, 1), np.clip(y + std, 0, 1)
    else:
        y_lower, y_upper = np.maximum(y - std, 0), y + std

    ax.fill_between(x, y_lower, y_upper, color=colors[algo], alpha=0.15)
    return True

# ─────────────────────────────────────────────
# 3. 绘制 6x4 终极大图 (保存为 SVG)
# ─────────────────────────────────────────────
print("\n🚀 开始生成 6x4 综合对比 SVG 大图...")

# 创建 6 行 4 列的图布
fig, axes = plt.subplots(nrows=6, ncols=4, figsize=(24, 28))
fig.suptitle('Comprehensive Scalability Across Environments (6 Metrics x 4 Contexts)', 
             fontsize=26, fontweight='bold', y=0.995)

env_display_names = ['Random Env', 'Maze Env', 'Indoor Env', 'All Envs Combined']

for row_idx, metric_config in enumerate(selected_metrics_config):
    for col_idx, env_name in enumerate(environments):
        ax = axes[row_idx, col_idx]
        
        # 数据过滤 (All 表示不区分布局取总体平均)
        if env_name == 'all':
            env_data = data.copy()
        else:
            env_data = data[data['environment'] == env_name].copy()

        # 遍历算法绘制
        for algo in algorithms:
            algo_slice = env_data[env_data['algorithm'] == algo]
            plot_metric_on_ax(ax, algo_slice, algo, metric_config)

        # 样式设置
        ax.grid(True, alpha=0.3, linestyle='--')
        if metric_config.get('is_rate', False): ax.set_ylim(0, 105)
        else: ax.set_ylim(bottom=0)

        # 仅在第一行设置列标题 (环境名称)
        if row_idx == 0: 
            ax.set_title(env_display_names[col_idx], fontsize=18, fontweight='bold', pad=15)
        
        # 仅在第一列设置 Y 轴标签 (指标名称)
        if col_idx == 0: 
            ax.set_ylabel(metric_config['title'], fontweight='bold')
        else:
            ax.set_ylabel("")

        # 仅在最后一行设置 X 轴标签
        if row_idx == 5: 
            ax.set_xlabel('Sample Budget', fontweight='bold')
        else:
            ax.set_xlabel("")

        # 图例：为了干净，仅在第一行的最后一列（All Envs 右上角）显示一次图例
        if row_idx == 0 and col_idx == 3:
            ax.legend(loc='lower left', bbox_to_anchor=(1.02, 0), frameon=True, shadow=True)
        elif ax.get_legend():
            ax.get_legend().remove()

plt.tight_layout(rect=[0, 0, 0.92, 0.98])  # 留出右侧空间给图例

# 保存为高质量 SVG
out_path = os.path.join(charts_dir, 'scalability_ALL_6x4.svg')
plt.savefig(out_path, format='svg', bbox_inches='tight', dpi=300)
print(f"✅ 成功保存 6x4 大图至: {os.path.abspath(out_path)}")
plt.close()