"""
可扩展性分析脚本 - 每个环境单独生成一张 3x2 SVG 对比图
布局：3行 × 2列，展示6个核心指标（已移除 dispersion / clearance）
支持算法：delta, beam, spars, gsrm
"""

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import os

# ─────────────────────────────────────────────────────────────────────
# 1. 数据加载
# ─────────────────────────────────────────────────────────────────────
DATA_FILE = './pursuer_strategies/PRM/results/scalability_evaluation_FAST_202602280234.csv'
data = pd.read_csv(DATA_FILE)

# 列名对齐检查（防御性输出，方便排查 CSV 与脚本不一致的问题）
print("CSV columns:", list(data.columns))
print("Algorithms  :", sorted(data['algorithm'].unique()))
print("Environments:", sorted(data['environment'].unique()))
print("Samples     :", sorted(data['num_samples'].unique()))
print()

# ─────────────────────────────────────────────────────────────────────
# 2. 样式配置
# ─────────────────────────────────────────────────────────────────────
ALGO_STYLE = {
    'delta': dict(color='#1f77b4', linestyle='-',  marker='o'),   # 蓝
    'beam':  dict(color='#ff7f0e', linestyle='-',  marker='s'),   # 橙
    'spars': dict(color='#2ca02c', linestyle='--', marker='^'),   # 绿
    'gsrm':  dict(color='#d62728', linestyle='-.', marker='D'),   # 红
}
ALGORITHMS = ['delta', 'beam', 'spars', 'gsrm']

plt.rcParams.update({
    'font.family':     'Times New Roman',
    'font.size':       12,
    'axes.labelsize':  13,
    'axes.titlesize':  14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 13,
    'axes.spines.top':    False,
    'axes.spines.right':  False,
})

# ─────────────────────────────────────────────────────────────────────
# 3. 指标配置
#    column          : CSV 中的列名
#    title           : 子图标题 & Y 轴标签
#    ylabel          : Y 轴单位（可与 title 不同，留空则用 title）
#    filter_success  : True → 只统计 path_success_rate > 0 的行
#    is_rate         : True → Y 轴限制在 [0, 105]，阴影 clip 到 [0,100]
#    ylim_bottom     : 自定义 Y 轴下界（None 则自动）
# ─────────────────────────────────────────────────────────────────────
METRICS_CONFIG = [
    # ── 第一行 ────────────────────────────────────────────────────────
    {
        'column':         'path_success_rate',
        'title':          'Path Success Rate',
        'ylabel':         'Success Rate (%)',
        'filter_success': False,
        'is_rate':        True,
        'ylim_bottom':    0,
    },
    {
        'column':         'generation_time',
        'title':          'Graph Generation Time',
        'ylabel':         'Time (s)',
        'filter_success': False,
        'is_rate':        False,
        'ylim_bottom':    0,
    },
    # ── 第二行 ────────────────────────────────────────────────────────
    {
        'column':         'spatial_coverage',
        'title':          'Spatial Coverage',
        'ylabel':         'Spatial Coverage (%)',
        'filter_success': False,
        'is_rate':        False,
        'ylim_bottom':    0,
    },
    {
        'column':         'edges_count',
        'title':          'Graph Edges Count',
        'ylabel':         'Edges',
        'filter_success': False,
        'is_rate':        False,
        'ylim_bottom':    0,
    },
    # ── 第三行 ────────────────────────────────────────────────────────
    {
        'column':         'path_length',
        'title':          'Average Path Length',
        'ylabel':         'Length (world units)',
        'filter_success': True,    # 只统计成功路径
        'is_rate':        False,
        'ylim_bottom':    0,
    },
    {
        'column':         'search_time',
        'title':          'Path Search Time',
        'ylabel':         'Time (s)',
        'filter_success': True,    # 只统计成功路径
        'is_rate':        False,
        'ylim_bottom':    0,
    },
]

ENV_DISPLAY = {
    'random': 'Random Environment',
    'maze':   'Maze Environment',
    'indoor': 'Indoor Environment',
}
ENVIRONMENTS = ['random', 'maze', 'indoor']

charts_dir = './pursuer_strategies/PRM/results/charts'
os.makedirs(charts_dir, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────
# 4. 单格绘图函数
# ─────────────────────────────────────────────────────────────────────
def _plot_single_metric(ax, env_data, metric_cfg, is_bottom_row: bool):
    """
    在 ax 上绘制一个指标的四条算法曲线（含标准差阴影）。

    参数
    ----
    ax            : matplotlib Axes
    env_data      : 已按环境过滤的 DataFrame
    metric_cfg    : METRICS_CONFIG 中的一项
    is_bottom_row : True → 添加 X 轴标签
    """
    col     = metric_cfg['column']
    is_rate = metric_cfg.get('is_rate', False)
    f_succ  = metric_cfg.get('filter_success', False)

    # 检查列是否存在（防止 CSV 版本不匹配时静默出错）
    if col not in env_data.columns:
        ax.text(0.5, 0.5, f'Column\n"{col}"\nnot found',
                ha='center', va='center', transform=ax.transAxes,
                color='red', fontsize=11)
        ax.set_title(metric_cfg['title'])
        return

    any_drawn = False
    for algo in ALGORITHMS:
        style    = ALGO_STYLE[algo]
        slice_df = env_data[env_data['algorithm'] == algo].copy()

        # 过滤：仅保留成功路径
        if f_succ:
            slice_df = slice_df[slice_df['path_success_rate'] > 0]

        if slice_df.empty:
            continue

        # 按 num_samples 聚合：均值 + 标准差
        grouped = (slice_df
                   .groupby('num_samples')[col]
                   .agg(['mean', 'std'])
                   .reset_index())
        grouped['std'] = grouped['std'].fillna(0)

        x   = grouped['num_samples'].values
        y   = grouped['mean'].values
        std = grouped['std'].values

        ax.plot(x, y,
                marker=style['marker'],
                linestyle=style['linestyle'],
                color=style['color'],
                linewidth=2.2,
                markersize=6,
                label=algo.upper(),
                zorder=3)

        # 标准差阴影
        if is_rate:
            lo = np.clip(y - std, 0, 100)
            hi = np.clip(y + std, 0, 100)
        else:
            lo = np.maximum(y - std, 0)
            hi = y + std

        ax.fill_between(x, lo, hi,
                         color=style['color'],
                         alpha=0.13,
                         zorder=2)
        any_drawn = True

    # ── 轴美化 ──────────────────────────────────────────────────────
    ax.set_title(metric_cfg['title'], fontweight='bold', pad=8)
    ax.set_ylabel(metric_cfg.get('ylabel') or metric_cfg['title'])
    ax.grid(True, alpha=0.30, linestyle='--', zorder=1)

    if is_rate:
        ax.set_ylim(0, 108)
        ax.yaxis.set_major_formatter(
            matplotlib.ticker.FormatStrFormatter('%.0f%%'))
    else:
        bot = metric_cfg.get('ylim_bottom', 0)
        ax.set_ylim(bottom=bot if bot is not None else None)

    if is_bottom_row:
        ax.set_xlabel('Sample Budget / Iterations', fontweight='bold')
    else:
        ax.set_xlabel('')

    # 若无数据则给提示
    if not any_drawn:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                transform=ax.transAxes, color='grey', fontsize=11)

# ─────────────────────────────────────────────────────────────────────
# 5. 主循环：每个环境生成一张独立 SVG
# ─────────────────────────────────────────────────────────────────────
import matplotlib.ticker   # 供 formatter 使用

for env_name in ENVIRONMENTS:
    env_data    = data[data['environment'] == env_name].copy()
    env_title   = ENV_DISPLAY.get(env_name, env_name)

    print(f"📊 绘制 [{env_title}] ...")

    # 创建 3行 × 2列 图布
    fig, axes = plt.subplots(
        nrows=3, ncols=2,
        figsize=(14, 16),           # 宽 14 英寸 × 高 16 英寸
        constrained_layout=False,   # 手动控制间距
    )

    # ── 主标题 ────────────────────────────────────────────────────────
    fig.suptitle(
        f'Scalability Analysis — {env_title}',
        fontsize=20, fontweight='bold', y=0.995,
    )

    # ── 逐格绘制 ──────────────────────────────────────────────────────
    for m_idx, metric_cfg in enumerate(METRICS_CONFIG):
        row = m_idx // 2          # 0, 0, 1, 1, 2, 2
        col = m_idx %  2          # 0, 1, 0, 1, 0, 1
        ax  = axes[row, col]
        is_bottom = (row == 2)    # 第三行（最后一行）才加 X 轴标签

        _plot_single_metric(ax, env_data, metric_cfg, is_bottom)

    # ── 全局图例（放在图顶部居中，与 suptitle 分层）──────────────────
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles, labels,
            loc='upper center',
            bbox_to_anchor=(0.5, 0.972),
            ncol=len(ALGORITHMS),
            frameon=True,
            framealpha=0.9,
            edgecolor='#cccccc',
            shadow=False,
            fontsize=13,
            handlelength=2.4,
            handletextpad=0.6,
            columnspacing=1.5,
        )

    # ── 子图间距 ──────────────────────────────────────────────────────
    # top 留给 suptitle + legend；bottom/left/right 留适量白边
    plt.subplots_adjust(
        top=0.93,
        bottom=0.06,
        left=0.09,
        right=0.97,
        hspace=0.45,   # 行间距
        wspace=0.32,   # 列间距
    )

    # ── 保存 SVG ──────────────────────────────────────────────────────
    out_path = os.path.join(charts_dir, f'scalability_{env_name}_3x2.svg')
    fig.savefig(out_path, format='svg', bbox_inches='tight')
    plt.close(fig)
    print(f"   ✅ 已保存: {os.path.abspath(out_path)}")

print("\n🎉 全部完成！共生成 3 张 SVG 图像：")
for env_name in ENVIRONMENTS:
    p = os.path.join(charts_dir, f'scalability_{env_name}_3x2.svg')
    print(f"   • {os.path.abspath(p)}")