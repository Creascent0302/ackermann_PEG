"""
可扩展性分析脚本 - 绘制 6x4 综合对比 SVG 大图
布局：按环境分组堆叠 (共3个环境，每个环境占2行4列，展示8个核心指标)
(剔除 path_nodes_count，保留 4 种核心算法: delta, beam, spars, gsrm)
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
# ⚠️ 注意: 运行此脚本前，请确保这里的 CSV 文件名和你刚跑完的一致
DATA_FILE = './pursuer_strategies/PRM/results/scalability_evaluation_202602230628.csv' 
data = pd.read_csv(DATA_FILE)

# 颜色方案与线型（4种算法）
colors = {
    'delta': '#1f77b4',   # 蓝色
    'beam':  '#ff7f0e',   # 橙色
    'spars': '#2ca02c',   # 绿色
    'gsrm':  '#d62728',   # 红色
}
linestyles = {'delta': '-', 'beam': '-', 'spars': '--', 'gsrm': '-.'}
markers    = {'delta': 'o', 'beam': 's', 'spars': '^', 'gsrm': 'D'}
algorithms = ['delta', 'beam', 'spars', 'gsrm']

# 全局图形样式
plt.rcParams['font.family']      = 'Times New Roman'
plt.rcParams['font.size']        = 13
plt.rcParams['axes.labelsize']   = 14
plt.rcParams['axes.titlesize']   = 15
plt.rcParams['xtick.labelsize']  = 12
plt.rcParams['ytick.labelsize']  = 12
plt.rcParams['legend.fontsize']  = 16

charts_dir = './pursuer_strategies/PRM/results/charts'
os.makedirs(charts_dir, exist_ok=True)

environments = ['random', 'maze', 'indoor']
env_display_names = {'random': 'Random Environment', 'maze': 'Maze Environment', 'indoor': 'Indoor Environment'}

# 精选 8 个核心指标 (每个环境 2行 x 4列)
# (已按要求删去 path_nodes_count，如果想替换其它指标可以直接在这里改 column 的名字)
selected_metrics_config = [
    # 第一行指标
    {'column': 'path_success',       'title': 'Path Success Rate (%)',   'is_rate': True},
    {'column': 'generation_time',    'title': 'Generation Time (s)',     'filter_success': False},
    {'column': 'actual_nodes_count', 'title': 'Actual Nodes Count',      'filter_success': False},
    {'column': 'edges_count',        'title': 'Graph Edges Count',       'filter_success': False},
    # 第二行指标
    {'column': 'path_length',        'title': 'Average Path Length',     'filter_success': True},
    {'column': 'search_time',        'title': 'Path Search Time (s)',    'filter_success': True},
    {'column': 'dispersion',         'title': 'Coverage Dispersion',     'filter_success': False},
    {'column': 'clearance',          'title': 'Path Clearance (Safety)', 'filter_success': True},
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

    grouped = algo_data.groupby('num_samples')[metric_col].agg(['mean', 'std']).reset_index()
    grouped['std'] = grouped['std'].fillna(0)

    x, y, std = grouped['num_samples'].values, grouped['mean'].values, grouped['std'].values

    ax.plot(x, y, marker=markers[algo], linestyle=linestyles[algo], 
            linewidth=2.5, markersize=7, label=algo.upper(), color=colors[algo])

    if metric_config.get('is_rate', False):
        y_lower, y_upper = np.clip(y - std, 0, 100), np.clip(y + std, 0, 100)
    else:
        y_lower, y_upper = np.maximum(y - std, 0), y + std

    ax.fill_between(x, y_lower, y_upper, color=colors[algo], alpha=0.15)
    return True

# ─────────────────────────────────────────────
# 3. 绘制 6x4 终极大图 (保存为 SVG)
# ─────────────────────────────────────────────
print("\n🚀 开始生成 6x4 (按环境堆叠) 综合对比 SVG 大图...")

# 创建 6 行 4 列的图布
fig, axes = plt.subplots(nrows=6, ncols=4, figsize=(24, 30))
fig.suptitle('Comprehensive Scalability Analysis Across Different Environments', 
             fontsize=28, fontweight='bold', y=0.995)

# 遍历每个环境 (每个环境占 2 行)
for env_idx, env_name in enumerate(environments):
    env_data = data[data['environment'] == env_name].copy()
    row_offset = env_idx * 2  # 当前环境的起始行号 (0, 2, 4)

    # 遍历 8 个指标
    for m_idx, metric_config in enumerate(selected_metrics_config):
        r = row_offset + (m_idx // 4)  # 计算在 6 行中的绝对行号
        c = m_idx % 4                  # 计算所在的列号
        ax = axes[r, c]

        # 遍历所有算法绘图
        has_data = False
        for algo in algorithms:
            algo_slice = env_data[env_data['algorithm'] == algo]
            if plot_metric_on_ax(ax, algo_slice, algo, metric_config):
                has_data = True

        ax.grid(True, alpha=0.3, linestyle='--')
        if metric_config.get('is_rate', False): ax.set_ylim(0, 105)
        else: ax.set_ylim(bottom=0)

        # 核心修改：明确标出环境与指标名称
        ax.set_title(f"[{env_display_names[env_name]}]\n{metric_config['title']}", 
                     fontsize=15, fontweight='bold', pad=10)
        ax.set_ylabel(metric_config['title'], fontsize=14)

        # 仅在最后一行 (第6行) 添加横轴标签
        if r == 5:
            ax.set_xlabel('Sample Budget (Nodes)', fontweight='bold', fontsize=14)
        else:
            ax.set_xlabel("")

# 提取图例，放置在整张图的最上方中心位置
handles, labels = axes[0, 0].get_legend_handles_labels()
if handles:
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.975), 
               ncol=4, frameon=True, shadow=True, fontsize=18)

# 调整子图间距，为顶部的主标题和图例留出充足空间
plt.tight_layout(rect=[0, 0, 1, 0.95], h_pad=2.0, w_pad=2.0)

# 保存为高质量 SVG
out_path = os.path.join(charts_dir, 'scalability_Envs_6x4.svg')
plt.savefig(out_path, format='svg', bbox_inches='tight', dpi=300)
print(f"✅ 成功保存 6x4 环境分组大图至: {os.path.abspath(out_path)}")
plt.close()