
"""
可扩展性分析脚本 - 每个指标生成一张 2x2 PDF 对比图
同时生成一张 6行×4列 的汇总大图（6指标 × 4环境）
"""

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import os

# ═════════════════════════════════════════════════════════════════════
# 字体配置（小图用）
# ═════════════════════════════════════════════════════════════════════
FONT_CONFIG = {
    'subplot_title': {'family': 'Times New Roman', 'size': 52, 'weight': 'normal', 'style': 'normal', 'color': 'black'},
    'axis_label':    {'family': 'Times New Roman', 'size': 48, 'weight': 'normal', 'style': 'normal', 'color': 'black'},
    'tick_label':    {'family': 'Times New Roman', 'size': 48, 'weight': 'normal', 'style': 'normal', 'color': 'black'},
    'legend':        {'family': 'Times New Roman', 'size': 48, 'weight': 'normal', 'style': 'normal', 'color': 'black'},
}

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

LINE_WIDTH  = 3.5
MARKER_SIZE = 11

plt.rcParams.update({
    'font.family':        'Times New Roman',
    'axes.unicode_minus': False,
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    'axes.linewidth':     1.8,
    'xtick.major.width':  1.8,
    'ytick.major.width':  1.8,
    'xtick.major.size':   8,
    'ytick.major.size':   8,
})

# ─────────────────────────────────────────────────────────────────────
# 3. 指标配置
#    ⚠️ Average Path Length 的 ylabel_unit 已清空（去掉括号内容）
# ─────────────────────────────────────────────────────────────────────
METRICS_CONFIG = [
    {
        'column':         'path_success_rate',
        'title':          'Path Success Rate',
        'ylabel_unit':    '%',
        'filter_success': False,
        'is_rate':        True,
        'ylim_bottom':    0,
    },
    {
        'column':         'generation_time',
        'title':          'Graph Generation Time',
        'ylabel_unit':    's',
        'filter_success': False,
        'is_rate':        False,
        'ylim_bottom':    0,
    },
    {
        'column':         'spatial_coverage',
        'title':          'Spatial Coverage',
        'ylabel_unit':    '',
        'filter_success': False,
        'is_rate':        False,
        'ylim_bottom':    0,
    },
    {
        'column':         'edges_count',
        'title':          'Graph Edges Count',
        'ylabel_unit':    '',
        'filter_success': False,
        'is_rate':        False,
        'ylim_bottom':    0,
    },
    {
        'column':         'path_length',
        'title':          'Average Path Length',
        'ylabel_unit':    '',          # ← 括号内容已去掉
        'filter_success': True,
        'is_rate':        False,
        'ylim_bottom':    0,
    },
    {
        'column':         'search_time',
        'title':          'Path Search Time',
        'ylabel_unit':    's',
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
def _fp(key: str, font_cfg: dict = None) -> dict:
    cfg = (font_cfg or FONT_CONFIG)[key]
    return {
        'fontfamily': cfg['family'],
        'fontsize':   cfg['size'],
        'fontweight': cfg['weight'],
        'fontstyle':  cfg['style'],
        'color':      cfg['color'],
    }


def _ylabel_str(metric_cfg: dict) -> str:
    unit  = metric_cfg.get('ylabel_unit', '').strip()
    title = metric_cfg['title']
    return f"{title} ({unit})" if unit else title


def apply_font_to_ax(ax: plt.Axes, font_cfg: dict = None):
    fc = font_cfg or FONT_CONFIG
    tk_cfg = fc['tick_label']
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily(tk_cfg['family'])
        label.set_fontsize(tk_cfg['size'])
        label.set_fontweight(tk_cfg['weight'])
        label.set_fontstyle(tk_cfg['style'])
        label.set_color(tk_cfg['color'])

    al_cfg = fc['axis_label']
    for text_obj in [ax.xaxis.label, ax.yaxis.label]:
        text_obj.set_fontfamily(al_cfg['family'])
        text_obj.set_fontsize(al_cfg['size'])
        text_obj.set_fontweight(al_cfg['weight'])
        text_obj.set_fontstyle(al_cfg['style'])
        text_obj.set_color(al_cfg['color'])

    st_cfg = fc['subplot_title']
    ax.title.set_fontfamily(st_cfg['family'])
    ax.title.set_fontsize(st_cfg['size'])
    ax.title.set_fontweight(st_cfg['weight'])
    ax.title.set_fontstyle(st_cfg['style'])
    ax.title.set_color(st_cfg['color'])


def apply_font_to_legend(leg, font_cfg: dict = None):
    lg_cfg = (font_cfg or FONT_CONFIG)['legend']
    for text in leg.get_texts():
        text.set_fontfamily(lg_cfg['family'])
        text.set_fontsize(lg_cfg['size'])
        text.set_fontweight(lg_cfg['weight'])
        text.set_fontstyle(lg_cfg['style'])
        text.set_color(lg_cfg['color'])


# ─────────────────────────────────────────────────────────────────────
# 5. 单个子图绘制（复用于小图和大图）
# ─────────────────────────────────────────────────────────────────────
def plot_env_metric(ax: plt.Axes, env_data: pd.DataFrame, metric_cfg: dict):
    col     = metric_cfg['column']
    is_rate = metric_cfg.get('is_rate', False)
    f_succ  = metric_cfg.get('filter_success', False)

    if col not in env_data.columns:
        ax.text(0.5, 0.5, f'Column "{col}"\nnot found',
                ha='center', va='center', transform=ax.transAxes,
                color='red', fontsize=14)
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
            linewidth=LINE_WIDTH,
            markersize=MARKER_SIZE,
            label=algo.upper(),
            zorder=3,
        )

        lo = np.clip(y - s, 0, 100) if is_rate else np.maximum(y - s, 0)
        hi = np.clip(y + s, 0, 100) if is_rate else y + s
        ax.fill_between(x, lo, hi, color=style['color'], alpha=0.15, zorder=2)
        any_drawn = True

    ax.grid(False)

    if is_rate:
        ax.set_ylim(0, 108)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f%%'))
    else:
        bot = metric_cfg.get('ylim_bottom', 0)
        ax.set_ylim(bottom=(bot if bot is not None else None))

    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=4, integer=True))
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=4))

    if not any_drawn:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                transform=ax.transAxes, color='grey', fontsize=14)


# ═════════════════════════════════════════════════════════════════════
# 6. 生成 6 张小图（每个指标一张 2×2）
# ═════════════════════════════════════════════════════════════════════
for metric_cfg in METRICS_CONFIG:
    metric_col   = metric_cfg['column']
    metric_title = metric_cfg['title']
    y_label_str  = _ylabel_str(metric_cfg)

    print(f"📌 生成小图: [{metric_title}] ...")

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(30, 20), constrained_layout=False)

    for idx, env_name in enumerate(ENVIRONMENTS):
        r, c     = divmod(idx, 2)
        ax       = axes[r, c]
        env_data = data[data['environment'] == env_name].copy()

        ax.set_title(ENV_DISPLAY.get(env_name, env_name), pad=16, fontdict=_fp('subplot_title'))
        ax.set_ylabel(y_label_str, fontdict=_fp('axis_label'), labelpad=18)
        ax.set_xlabel('Sample Budget / Iterations', fontdict=_fp('axis_label'), labelpad=12)
        ax.tick_params(axis='both', which='major', pad=10, length=8)

        plot_env_metric(ax, env_data, metric_cfg)
        apply_font_to_ax(ax)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        leg = fig.legend(
            handles, labels,
            loc='upper center',
            bbox_to_anchor=(0.5, 0.995),
            ncol=len(ALGORITHMS),
            frameon=True, framealpha=0.92, edgecolor='#cccccc',
            fontsize=FONT_CONFIG['legend']['size'],
            handlelength=3.0, handletextpad=0.8, columnspacing=2.0,
        )
        apply_font_to_legend(leg)

    plt.subplots_adjust(top=0.88, bottom=0.12, left=0.16, right=0.97, hspace=0.55, wspace=0.45)

    out_path = os.path.join(charts_dir, f'scalability_{metric_col}_2x2.pdf')
    fig.savefig(out_path, format='pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"   ✅ 已保存: {os.path.abspath(out_path)}")


# ═════════════════════════════════════════════════════════════════════
# 7. 生成 1 张大图（6行 × 4列，6指标 × 4环境）
#
#    大图每个子图的物理尺寸与小图相同：
#      小图 figsize=(30,20) / 2列2行 → 每格 ≈ 15×10 英寸
#      大图 4列×6行 → figsize = (4×15, 6×10) = (60, 60)
#    因此字体 pt 值不变，在纸张上呈现比例与小图完全一致。
# ═════════════════════════════════════════════════════════════════════
print("\n📌 生成大图（6×4汇总图）...")

N_ROWS = len(METRICS_CONFIG)   # 6
N_COLS = len(ENVIRONMENTS)     # 4

# 每格与小图等尺寸：小图单格约 15×10，大图整体 60×60
CELL_W, CELL_H = 15, 10
fig_big, axes_big = plt.subplots(
    nrows=N_ROWS, ncols=N_COLS,
    figsize=(CELL_W * N_COLS, CELL_H * N_ROWS),
    constrained_layout=False,
)

for row_idx, metric_cfg in enumerate(METRICS_CONFIG):
    y_label_str = _ylabel_str(metric_cfg)

    for col_idx, env_name in enumerate(ENVIRONMENTS):
        ax       = axes_big[row_idx, col_idx]
        env_data = data[data['environment'] == env_name].copy()

        # ── 第一行：显示环境标题 ──────────────────────────────────────
        if row_idx == 0:
            ax.set_title(
                ENV_DISPLAY.get(env_name, env_name),
                pad=16,
                fontdict=_fp('subplot_title'),
            )

        # ── 第一列：显示 Y 轴标签 ────────────────────────────────────
        if col_idx == 0:
            ax.set_ylabel(y_label_str, fontdict=_fp('axis_label'), labelpad=18)
        else:
            ax.set_ylabel('')

        # ── 最后一行：显示 X 轴标签 ─────────────────────────────────
        if row_idx == N_ROWS - 1:
            ax.set_xlabel('Sample Budget / Iterations', fontdict=_fp('axis_label'), labelpad=12)
        else:
            ax.set_xlabel('')

        ax.tick_params(axis='both', which='major', pad=10, length=8)
        plot_env_metric(ax, env_data, metric_cfg)
        apply_font_to_ax(ax)

# ── 全局图例（只放一次，位于大图顶部）───────────────────────────────
handles, labels = axes_big[0, 0].get_legend_handles_labels()
if handles:
    leg_big = fig_big.legend(
        handles, labels,
        loc='upper center',
        bbox_to_anchor=(0.5, 0.998),
        ncol=len(ALGORITHMS),
        frameon=True, framealpha=0.92, edgecolor='#cccccc',
        fontsize=FONT_CONFIG['legend']['size'],
        handlelength=3.0, handletextpad=0.8, columnspacing=2.0,
    )
    apply_font_to_legend(leg_big)

# ── 边距：与小图比例保持一致 ──────────────────────────────────────
plt.subplots_adjust(
    top=0.955,     # 图例所需空间（整图变高，比例压缩）
    bottom=0.055,
    left=0.07,
    right=0.99,
    hspace=0.55,
    wspace=0.45,
)

out_path_big = os.path.join(charts_dir, 'scalability_all_metrics_6x4.pdf')
fig_big.savefig(out_path_big, format='pdf', bbox_inches='tight')
plt.close(fig_big)
print(f"   ✅ 大图已保存: {os.path.abspath(out_path_big)}")

# ─────────────────────────────────────────────────────────────────────
print("\n🎉 全部完成！")
print("  小图（6张）：")
for mc in METRICS_CONFIG:
    p = os.path.join(charts_dir, f"scalability_{mc['column']}_2x2.pdf")
    print(f"   • {os.path.abspath(p)}")
print(f"  大图（1张）：\n   • {os.path.abspath(out_path_big)}")
