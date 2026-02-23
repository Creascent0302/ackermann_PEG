
"""
可扩展性分析脚本 - 绘制折线图展示算法随采样次数变化的性能
支持多seed的均值和标准差可视化
"""

import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import numpy as np
import os

# 读取数据
data = pd.read_csv('./pursuer_strategies/PRM/results/scalability_evaluation_202602230628.csv')

# ─────────────────────────────────────────────
# 颜色方案（五种算法）
# ─────────────────────────────────────────────
colors = {
    'delta': '#1f77b4',   # 蓝色
    'beam':  '#ff7f0e',   # 橙色
    'spars': '#2ca02c',   # 绿色
    'gsrm':  '#d62728',   # 红色
    'odrm':  '#9467bd',   # 紫色
}

# 线型方案（在黑白打印时也能区分）
linestyles = {
    'delta': '-',
    'beam':  '-',
    'spars': '--',
    'gsrm':  '-.',
    'odrm':  ':',
}

markers = {
    'delta': 'o',
    'beam':  's',
    'spars': '^',
    'gsrm':  'D',
    'odrm':  'v',
}

# ─────────────────────────────────────────────
# 全局图形样式
# ─────────────────────────────────────────────
plt.rcParams['font.family']      = 'Times New Roman'
plt.rcParams['font.size']        = 12
plt.rcParams['axes.labelsize']   = 14
plt.rcParams['axes.titlesize']   = 16
plt.rcParams['xtick.labelsize']  = 12
plt.rcParams['ytick.labelsize']  = 12
plt.rcParams['legend.fontsize']  = 11

# 创建输出目录
charts_dir = './pursuer_strategies/PRM/results/charts'
os.makedirs(charts_dir, exist_ok=True)

# ─────────────────────────────────────────────
# 指标配置（横坐标统一改为 num_samples）
# ─────────────────────────────────────────────
metrics_config = [
    {
        'column':   'generation_time',
        'title':    'PRM Generation Time vs Number of Samples',
        'ylabel':   'Generation Time (s)',
        'filename': 'scalability_generation_time.png'
    },
    {
        'column':         'search_time',
        'title':          'Path Search Time vs Number of Samples',
        'ylabel':         'Search Time (s)',
        'filename':       'scalability_search_time.png',
        'filter_success': True
    },
    {
        'column':         'path_length',
        'title':          'Average Path Length vs Number of Samples',
        'ylabel':         'Path Length',
        'filename':       'scalability_path_length.png',
        'filter_success': True
    },
    {
        'column':         'path_nodes_count',
        'title':          'Path Node Count vs Number of Samples',
        'ylabel':         'Number of Nodes in Path',
        'filename':       'scalability_path_nodes.png',
        'filter_success': True
    },
    {
        'column':   'edges_count',
        'title':    'Number of Edges vs Number of Samples',
        'ylabel':   'Number of Edges',
        'filename': 'scalability_edges_count.png'
    },
    {
        'column':   'path_success',
        'title':    'Path Planning Success Rate vs Number of Samples',
        'ylabel':   'Success Rate (%)',
        'filename': 'scalability_success_rate.png',
        'is_rate':  True
    },
    {
        'column':   'dispersion',
        'title':    'Dispersion vs Number of Samples',
        'ylabel':   'Dispersion',
        'filename': 'scalability_dispersion.png'
    },
    {
        'column':    'node_utilization',
        'title':     'Node Utilization vs Number of Samples',
        'ylabel':    'Node Utilization',
        'filename':  'scalability_node_utilization.png',
        'is_ratio':  True
    },
    # ── 新增：clearance 指标 ──────────────────
    {
        'column':         'clearance',
        'title':          'Path Clearance vs Number of Samples',
        'ylabel':         'Clearance',
        'filename':       'scalability_clearance.png',
        'filter_success': True   # 只统计路径规划成功的行
    },
]

# ─────────────────────────────────────────────
# 算法和环境列表
# ─────────────────────────────────────────────
algorithms   = ['delta', 'beam', 'spars', 'gsrm', 'odrm']
environments = data['environment'].unique()

print("开始生成可扩展性分析图表（横坐标：采样次数）...")
print(f"检测到的环境: {list(environments)}")
print(f"检测到的算法: {list(data['algorithm'].unique())}")


# ─────────────────────────────────────────────
# 辅助函数：在单个 Axes 上绘制一条指标曲线
# ─────────────────────────────────────────────
def plot_metric_on_ax(ax, env_data, algo, metric_config):
    """
    在 ax 上绘制指定算法的折线 + 标准差阴影。
    返回 True 表示成功绘制，False 表示无数据。
    """
    metric_col     = metric_config['column']
    filter_success = metric_config.get('filter_success', False)
    is_rate        = metric_config.get('is_rate', False)
    is_ratio       = metric_config.get('is_ratio', False)

    algo_data = env_data[env_data['algorithm'] == algo].copy()

    if filter_success:
        algo_data = algo_data[algo_data['path_success'] > 0]

    if len(algo_data) == 0:
        return False

    # 按采样次数分组，计算均值和标准差
    grouped = (
        algo_data
        .groupby('num_samples')[metric_col]          # ← 横坐标：num_samples
        .agg(['mean', 'std', 'count'])
        .reset_index()
    )
    grouped['std'] = grouped['std'].fillna(0)

    x   = grouped['num_samples'].values
    y   = grouped['mean'].values
    std = grouped['std'].values

    # 主曲线
    ax.plot(
        x, y,
        marker=markers[algo],
        linestyle=linestyles[algo],
        linewidth=2,
        markersize=6,
        label=algo.upper(),
        color=colors[algo],
    )

    # 标准差阴影（按类型裁剪到合理区间）
    if is_rate:
        y_lower = np.clip(y - std, 0, 100)
        y_upper = np.clip(y + std, 0, 100)
    elif is_ratio:
        y_lower = np.clip(y - std, 0, 1)
        y_upper = np.clip(y + std, 0, 1)
    else:
        y_lower = np.maximum(y - std, 0)
        y_upper = y + std

    ax.fill_between(x, y_lower, y_upper, color=colors[algo], alpha=0.15)
    return True


def style_ax(ax, metric_config, title_suffix='', show_ylabel=True):
    """统一设置坐标轴样式"""
    is_rate  = metric_config.get('is_rate', False)
    is_ratio = metric_config.get('is_ratio', False)

    ax.set_xlabel('Number of Samples', fontweight='bold')   # ← 横坐标标签
    if show_ylabel:
        ax.set_ylabel(metric_config['ylabel'], fontweight='bold')

    display_title = metric_config['title']
    if title_suffix:
        display_title = title_suffix
    ax.set_title(display_title, fontsize=14, fontweight='bold')

    ax.legend(loc='best', frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--')

    if is_rate:
        ax.set_ylim(0, 105)
    else:
        ax.set_ylim(bottom=0)


# ─────────────────────────────────────────────
# 1. 每个环境一张综合图（3×3 布局，9 个指标）
# ─────────────────────────────────────────────
for env in environments:
    print(f"\n正在处理环境: {env}")
    env_data = data[data['environment'] == env]

    fig, axes = plt.subplots(3, 3, figsize=(28, 18))   # ← 2×4 → 3×3
    fig.suptitle(
        f'Algorithm Scalability Analysis - {env.capitalize()} Environment',
        fontsize=20, fontweight='bold', y=1.002,
    )

    for idx, metric_config in enumerate(metrics_config):
        row, col = divmod(idx, 3)                       # ← 每行 4 → 3
        ax = axes[row, col]

        for algo in algorithms:
            ok = plot_metric_on_ax(ax, env_data, algo, metric_config)
            if not ok:
                print(f"  警告: {algo} 在 {env} 环境的 "
                      f"{metric_config['column']} 没有数据")

        style_ax(ax, metric_config)

    # 隐藏多余的子图（3×3=9 格，刚好放满，无需隐藏）
    plt.tight_layout()
    out = os.path.join(charts_dir, f'scalability_analysis_{env}.png')
    plt.savefig(out, dpi=300, bbox_inches='tight')
    print(f"  已保存: {out}")
    plt.close()


# ─────────────────────────────────────────────
# 2. 每个指标一张综合图（1×3 列，3 个环境）
# ─────────────────────────────────────────────
print("\n正在生成综合对比图（每指标×每环境）...")

for metric_config in metrics_config:
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    fig.suptitle(metric_config['title'], fontsize=18, fontweight='bold', y=1.02)

    for env_idx, env in enumerate(environments):
        ax       = axes[env_idx]
        env_data = data[data['environment'] == env]

        for algo in algorithms:
            plot_metric_on_ax(ax, env_data, algo, metric_config)

        style_ax(
            ax,
            metric_config,
            title_suffix=f'{env.capitalize()} Environment',
            show_ylabel=(env_idx == 0),
        )

    plt.tight_layout()
    out = os.path.join(charts_dir, metric_config['filename'])
    plt.savefig(out, dpi=300, bbox_inches='tight')
    print(f"  已保存: {out}")
    plt.close()


# ─────────────────────────────────────────────
# 3. 统计摘要
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("统计摘要")
print("=" * 60)

for env in environments:
    print(f"\n{env.upper()} Environment:")
    env_data = data[data['environment'] == env]

    for algo in algorithms:
        algo_data = env_data[env_data['algorithm'] == algo]
        if len(algo_data) == 0:
            continue

        print(f"\n  {algo.upper()}:")
        print(f"    测试次数:       {len(algo_data)}")
        print(f"    采样次数范围:   "
              f"{algo_data['num_samples'].min()} ~ "
              f"{algo_data['num_samples'].max()}")
        print(f"    平均生成时间:   "
              f"{algo_data['generation_time'].mean():.3f}s "
              f"± {algo_data['generation_time'].std():.3f}s")
        print(f"    平均实际节点数: "
              f"{algo_data['actual_nodes_count'].mean():.1f} "
              f"± {algo_data['actual_nodes_count'].std():.1f}")
        print(f"    平均边数:       "
              f"{algo_data['edges_count'].mean():.1f} "
              f"± {algo_data['edges_count'].std():.1f}")
        print(f"    平均节点利用率: "
              f"{algo_data['node_utilization'].mean()*100:.1f}% "
              f"± {algo_data['node_utilization'].std()*100:.1f}%")

        # 路径相关（只看有成功路径的行）
        success_data = algo_data[algo_data['path_success'] > 0]
        if len(success_data) > 0:
            overall_sr = algo_data['path_success'].mean()
            print(f"    平均路径成功率: {overall_sr:.1f}%")
            print(f"    平均路径长度:   "
                  f"{success_data['path_length'].mean():.3f} "
                  f"± {success_data['path_length'].std():.3f}")
            print(f"    平均搜索时间:   "
                  f"{success_data['search_time'].mean():.4f}s "
                  f"± {success_data['search_time'].std():.4f}s")
            print(f"    平均离散度:     "
                  f"{algo_data['dispersion'].mean():.4f} "
                  f"± {algo_data['dispersion'].std():.4f}")
            # ── 新增：clearance 统计摘要 ─────────
            print(f"    平均路径间隙:   "
                  f"{success_data['clearance'].mean():.4f} "
                  f"± {success_data['clearance'].std():.4f}")
        else:
            print(f"    路径成功率:     0.0%（无成功路径）")

print("\n" + "=" * 60)
print("所有图表已保存到:", charts_dir)
print("=" * 60)
