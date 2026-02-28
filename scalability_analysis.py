
"""
可扩展性分析脚本 - 每个指标生成一张 2x2 SVG 对比图
布局：2行 × 2列，展示4个环境（random/maze/indoor/four_rooms）
每个子图展示4种算法（delta/beam/spars/gsrm）的折线图（均值+std阴影）
共生成 6 张 SVG（6个指标）

修改说明：
  1. 去掉大标题（suptitle）
  2. 原标题内容（metric title）改为每个子图的 Y 轴标签，原 ylabel 改为 Y 轴单位放在括号内追加
  3. 去掉所有网格线
  4. 所有文字 weight 均为 'normal'（不加粗）
"""

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import os

FONT_CONFIG = {

    # ── 子图小标题（每个环境的标题）─────────────────────────────────
    'subplot_title': {
        'family': 'Times New Roman',
        'size':   16,
        'weight': 'normal',      # ← 不加粗
        'style':  'normal',
        'color':  'black',
    },

    # ── 坐标轴标签（X轴/Y轴 文字）──────────────────────────────────
    'axis_label': {
        'family': 'Times New Roman',
        'size':   14,
        'weight': 'normal',      # ← 不加粗
        'style':  'normal',
        'color':  'black',
    },

    # ── 坐标轴刻度数字 ──────────────────────────────────────────────
    'tick_label': {
        'family': 'Times New Roman',
        'size':   11,
        'weight': 'normal',
        'style':  'normal',
        'color':  'black',
    },

    # ── 图例文字（算法名称）─────────────────────────────────────────
    'legend': {
        'family': 'Times New Roman',
        'size':   12,
        'weight': 'normal',
        'style':  'normal',
        'color':  'black',
    },
}
# ═════════════════════════════════════════════════════════════════════


# ─────────────────────────────────────────────────────────────────────
# 1. 数据加载
# ─────────────────────────────────────────────────────────────────────
DATA_FILE = './pursuer_strategies/PRM/results/scalability_evaluation_final.csv'
data = pd.read_csv(DATA_FILE)

print("CSV columns:", list(data.columns))
print("Algorithms  :", sorted(data['algorithm'].unique()))
print("Environments:", sorted(data['environment'].unique()))
print("Samples     :", sorted(data['num_samples'].unique()))
print()

# ─────────────────────────────────────────────────────────────────────
# 2. 算法样式配置
# ─────────────────────────────────────────────────────────────────────
ALGO_STYLE = {
    'delta': dict(color='#1f77b4', linestyle='-',  marker='o'),
    'beam':  dict(color='#ff7f0e', linestyle='-',  marker='s'),
    'spars': dict(color='#2ca02c', linestyle='--', marker='^'),
    'gsrm':  dict(color='#d62728', linestyle='-.', marker='D'),
}
ALGORITHMS = ['delta', 'beam', 'spars', 'gsrm']

plt.rcParams.update({
    'font.family':        FONT_CONFIG['tick_label']['family'],
    'axes.unicode_minus': False,
    'axes.spines.top':    False,
    'axes.spines.right':  False,
})

# ─────────────────────────────────────────────────────────────────────
# 3. 指标配置
#    ylabel_unit : 括号内的单位，追加到 title 后作为完整 Y 轴标签
#                  留空则 Y 轴标签直接等于 title
# ─────────────────────────────────────────────────────────────────────
METRICS_CONFIG = [
    {
        'column':         'path_success_rate',
        'title':          'Path Success Rate',
        'ylabel_unit':    '%',           # → Y轴显示 "Path Success Rate (%)"
        'filter_success': False,
        'is_rate':        True,
        'ylim_bottom':    0,
    },
    {
        'column':         'generation_time',
        'title':          'Graph Generation Time',
        'ylabel_unit':    's',           # → "Graph Generation Time (s)"
        'filter_success': False,
        'is_rate':        False,
        'ylim_bottom':    0,
    },
    {
        'column':         'spatial_coverage',
        'title':          'Spatial Coverage',
        'ylabel_unit':    '',            # → "Spatial Coverage"（无单位）
        'filter_success': False,
        'is_rate':        False,
        'ylim_bottom':    0,
    },
    {
        'column':         'edges_count',
        'title':          'Graph Edges Count',
        'ylabel_unit':    '',            # → "Graph Edges Count"
        'filter_success': False,
        'is_rate':        False,
        'ylim_bottom':    0,
    },
    {
        'column':         'path_length',
        'title':          'Average Path Length',
        'ylabel_unit':    'world units', # → "Average Path Length (world units)"
        'filter_success': True,
        'is_rate':        False,
        'ylim_bottom':    0,
    },
    {
        'column':         'search_time',
        'title':          'Path Search Time',
        'ylabel_unit':    's',           # → "Path Search Time (s)"
        'filter_success': True,
        'is_rate':        False,
        'ylim_bottom':    0,
    },
]

ENV_DISPLAY = {
    'random':     'Random Environment',
    'maze':       'Maze Environment',
    'indoor':     'Indoor Environment',
    'four_rooms': 'Four Rooms Environment',
}
ENVIRONMENTS = ['random', 'maze', 'indoor', 'four_rooms']

charts_dir = './pursuer_strategies/PRM/results/charts_by_metric'
os.makedirs(charts_dir, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────
# 4. 工具函数
# ─────────────────────────────────────────────────────────────────────
def _fp(key: str) -> dict:
    """将 FONT_CONFIG[key] 转换为 matplotlib fontdict"""
    cfg = FONT_CONFIG[key]
    return {
        'fontfamily': cfg['family'],
        'fontsize':   cfg['size'],
        'fontweight': cfg['weight'],
        'fontstyle':  cfg['style'],
        'color':      cfg['color'],
    }


def _ylabel_str(metric_cfg: dict) -> str:
    """
    构建 Y 轴标签字符串：
      有单位 → "Title (unit)"
      无单位 → "Title"
    """
    unit = metric_cfg.get('ylabel_unit', '').strip()
    title = metric_cfg['title']
    return f"{title} ({unit})" if unit else title


def apply_font_to_ax(ax: plt.Axes):
    """将 FONT_CONFIG 精确应用到 Axes 的所有文本元素"""
    # 刻度数字
    tk_cfg = FONT_CONFIG['tick_label']
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily(tk_cfg['family'])
        label.set_fontsize(tk_cfg['size'])
        label.set_fontweight(tk_cfg['weight'])
        label.set_fontstyle(tk_cfg['style'])
        label.set_color(tk_cfg['color'])

    # 坐标轴标签（二次加固）
    al_cfg = FONT_CONFIG['axis_label']
    for text_obj in [ax.xaxis.label, ax.yaxis.label]:
        text_obj.set_fontfamily(al_cfg['family'])
        text_obj.set_fontsize(al_cfg['size'])
        text_obj.set_fontweight(al_cfg['weight'])
        text_obj.set_fontstyle(al_cfg['style'])
        text_obj.set_color(al_cfg['color'])

    # 子图标题（二次加固）
    st_cfg = FONT_CONFIG['subplot_title']
    ax.title.set_fontfamily(st_cfg['family'])
    ax.title.set_fontsize(st_cfg['size'])
    ax.title.set_fontweight(st_cfg['weight'])
    ax.title.set_fontstyle(st_cfg['style'])
    ax.title.set_color(st_cfg['color'])


def apply_font_to_legend(leg):
    """将 FONT_CONFIG['legend'] 应用到图例文字"""
    lg_cfg = FONT_CONFIG['legend']
    for text in leg.get_texts():
        text.set_fontfamily(lg_cfg['family'])
        text.set_fontsize(lg_cfg['size'])
        text.set_fontweight(lg_cfg['weight'])
        text.set_fontstyle(lg_cfg['style'])
        text.set_color(lg_cfg['color'])


# ─────────────────────────────────────────────────────────────────────
# 5. 单个子图绘制
# ─────────────────────────────────────────────────────────────────────
def plot_env_metric(ax: plt.Axes, env_data: pd.DataFrame, metric_cfg: dict):
    col     = metric_cfg['column']
    is_rate = metric_cfg.get('is_rate', False)
    f_succ  = metric_cfg.get('filter_success', False)

    if col not in env_data.columns:
        ax.text(0.5, 0.5, f'Column "{col}"\nnot found',
                ha='center', va='center', transform=ax.transAxes,
                color='red', fontsize=11)
        return

    any_drawn = False
    for algo in ALGORITHMS:
        style    = ALGO_STYLE[algo]
        slice_df = env_data[env_data['algorithm'] == algo].copy()

        if f_succ:
            slice_df = slice_df[slice_df['path_success_rate'] > 0]
        if slice_df.empty:
            continue

        grouped = (slice_df
                   .groupby('num_samples')[col]
                   .agg(['mean', 'std'])
                   .reset_index())
        grouped['std'] = grouped['std'].fillna(0)

        x = grouped['num_samples'].to_numpy()
        y = grouped['mean'].to_numpy()
        s = grouped['std'].to_numpy()

        ax.plot(
            x, y,
            marker=style['marker'],
            linestyle=style['linestyle'],
            color=style['color'],
            linewidth=2.2,
            markersize=6,
            label=algo.upper(),
            zorder=3,
        )

        lo = np.clip(y - s, 0, 100) if is_rate else np.maximum(y - s, 0)
        hi = np.clip(y + s, 0, 100) if is_rate else y + s

        ax.fill_between(x, lo, hi, color=style['color'], alpha=0.13, zorder=2)
        any_drawn = True

    # ── 无网格线 ──────────────────────────────────────────────────────
    ax.grid(False)

    if is_rate:
        ax.set_ylim(0, 108)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f%%'))
    else:
        bot = metric_cfg.get('ylim_bottom', 0)
        ax.set_ylim(bottom=(bot if bot is not None else None))

    if not any_drawn:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                transform=ax.transAxes, color='grey', fontsize=11)


# ─────────────────────────────────────────────────────────────────────
# 6. 主循环：每个指标 → 一张 2×2 SVG
# ─────────────────────────────────────────────────────────────────────
for metric_cfg in METRICS_CONFIG:
    metric_col   = metric_cfg['column']
    metric_title = metric_cfg['title']
    y_label_str  = _ylabel_str(metric_cfg)   # 用 title 构建的 Y 轴标签

    print(f"📌 生成指标图: [{metric_title}] ...")

    fig, axes = plt.subplots(
        nrows=2, ncols=2,
        figsize=(14, 10),
        constrained_layout=False,
    )


    # ── 2×2 子图 ──────────────────────────────────────────────────────
    for idx, env_name in enumerate(ENVIRONMENTS):
        r, c      = divmod(idx, 2)
        ax        = axes[r, c]
        env_data  = data[data['environment'] == env_name].copy()
        env_title = ENV_DISPLAY.get(env_name, env_name)

        # 子图小标题（环境名称）
        ax.set_title(env_title, pad=8, fontdict=_fp('subplot_title'))

        # Y 轴：原 metric title + 单位；X 轴：固定说明
        ax.set_ylabel(y_label_str,                     fontdict=_fp('axis_label'))
        ax.set_xlabel('Sample Budget / Iterations',    fontdict=_fp('axis_label'))

        plot_env_metric(ax, env_data, metric_cfg)
        apply_font_to_ax(ax)

    # ── 全局图例（居中，位于四个子图上方）────────────────────────────
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        leg = fig.legend(
            handles, labels,
            loc='upper center',
            bbox_to_anchor=(0.5, 0.995),
            ncol=len(ALGORITHMS),
            frameon=True,
            framealpha=0.92,
            edgecolor='#cccccc',
            shadow=False,
            fontsize=FONT_CONFIG['legend']['size'],
            handlelength=2.4,
            handletextpad=0.6,
            columnspacing=1.5,
        )
        apply_font_to_legend(leg)

    plt.subplots_adjust(
        top=0.93,       # 顶部留给图例（无 suptitle，可更紧凑）
        bottom=0.09,
        left=0.09,
        right=0.98,
        hspace=0.26,
        wspace=0.10,
    )

    out_path = os.path.join(charts_dir, f'scalability_{metric_col}_2x2.svg')
    fig.savefig(out_path, format='svg', bbox_inches='tight')
    plt.close(fig)
    print(f"   ✅ 已保存: {os.path.abspath(out_path)}")

print("\n🎉 全部完成！共生成 6 张 2×2 SVG（每张=一个指标，含四环境对比）")
for metric_cfg in METRICS_CONFIG:
    p = os.path.join(charts_dir, f"scalability_{metric_cfg['column']}_2x2.svg")
    print("   •", os.path.abspath(p))
