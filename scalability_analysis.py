
"""
可扩展性分析脚本 - 6行×4列 汇总大图（最终版）

改动：
1. 字体全部再放大（subplot_title:74, axis_label:72, tick_label:70, legend:70, row_label:76）
2. 图更扁：CELL_H 从 15 降至 10，整图高宽比大幅压缩
3. 编号 (a)-(f) 放在每行第一列子图的左上角（axes坐标 -0.02, 1.0），
   ha='right', va='top'，紧贴子图左上顶点外侧，不加粗，不与任何元素重叠
"""

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import os

# ═════════════════════════════════════════════════════════════════════
# 字体配置
# ═════════════════════════════════════════════════════════════════════
FONT_CONFIG = {
    'subplot_title': {'family': 'Times New Roman', 'size': 74, 'weight': 'normal', 'style': 'normal', 'color': 'black'},
    'axis_label':    {'family': 'Times New Roman', 'size': 72, 'weight': 'normal', 'style': 'normal', 'color': 'black'},
    'tick_label':    {'family': 'Times New Roman', 'size': 70, 'weight': 'normal', 'style': 'normal', 'color': 'black'},
    'legend':        {'family': 'Times New Roman', 'size': 70, 'weight': 'normal', 'style': 'normal', 'color': 'black'},
    'row_label':     {'family': 'Times New Roman', 'size': 76, 'weight': 'normal', 'style': 'normal', 'color': 'black'},
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
    'delta': dict(color='#1f77b4', linestyle='-',  marker='o', label=' -PRM'),
    'beam':  dict(color='#ff7f0e', linestyle='-',  marker='s', label='BSRM'),
    'spars': dict(color='#2ca02c', linestyle='--', marker='^', label='SPARS2'),
    'gsrm':  dict(color='#d62728', linestyle='-.', marker='D', label='GSRM'),
}
ALGORITHMS = ['delta', 'beam', 'spars', 'gsrm']

LINE_WIDTH  = 4.0
MARKER_SIZE = 16

plt.rcParams.update({
    'font.family':        'Times New Roman',
    'font.size':          70,
    'axes.unicode_minus': False,
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    'axes.linewidth':     2.2,
    'xtick.major.width':  2.2,
    'ytick.major.width':  2.2,
    'xtick.major.size':   12,
    'ytick.major.size':   12,
    'xtick.labelsize':    70,
    'ytick.labelsize':    70,
})

# ─────────────────────────────────────────────────────────────────────
# 3. 指标配置
# ─────────────────────────────────────────────────────────────────────
METRICS_CONFIG = [
    {'column': 'path_success_rate', 'title': 'Path-finding Success Rate',
     'ylabel_unit': '%',  'filter_success': False, 'is_rate': True,  'ylim_bottom': 0, 'row_label': '(a)'},
    {'column': 'generation_time',   'title': 'Graph Generation Time',
     'ylabel_unit': 's',  'filter_success': False, 'is_rate': False, 'ylim_bottom': 0, 'row_label': '(b)'},
    {'column': 'spatial_coverage',  'title': 'Spatial Coverage',
     'ylabel_unit': '',   'filter_success': False, 'is_rate': False, 'ylim_bottom': 0, 'row_label': '(c)'},
    {'column': 'edges_count',       'title': 'Graph Edges Count',
     'ylabel_unit': '',   'filter_success': False, 'is_rate': False, 'ylim_bottom': 0, 'row_label': '(d)'},
    {'column': 'path_length',       'title': 'Average Path Length',
     'ylabel_unit': '',   'filter_success': True,  'is_rate': False, 'ylim_bottom': 0, 'row_label': '(e)'},
    {'column': 'search_time',       'title': 'Path Search Time',
     'ylabel_unit': 's',  'filter_success': True,  'is_rate': False, 'ylim_bottom': 0, 'row_label': '(f)'},
]

ENV_DISPLAY = {
    'random':     'Cluttered Environment',
    'maze':       'Maze Environment',
    'indoor':     'Indoor Environment',
    'four_rooms': 'Narrow Passage Environment',
}
ENVIRONMENTS = ['random', 'maze', 'indoor', 'four_rooms']

charts_dir = './pursuer_strategies/PRM/results/charts_by_metric'
os.makedirs(charts_dir, exist_ok=True)

SUCCESS_RATE_THRESHOLD = 50.0

# ─────────────────────────────────────────────────────────────────────
# 4. 预计算：每个 (环境, 算法) 成功率首次 >= 50% 的最小采样数
# ─────────────────────────────────────────────────────────────────────
def compute_algo_env_threshold(env_name: str, algo: str) -> float:
    col = 'path_success_rate'
    if col not in data.columns:
        return float('-inf')
    subset = data[
        (data['environment'] == env_name) &
        (data['algorithm']   == algo)
    ][['num_samples', col]].copy()
    if subset.empty:
        return float('inf')
    grouped = subset.groupby('num_samples')[col].mean().reset_index()
    grouped.columns = ['num_samples', 'mean_rate']
    if grouped['mean_rate'].max() <= 1.0:
        grouped['mean_rate'] *= 100.0
    valid = grouped[grouped['mean_rate'] >= SUCCESS_RATE_THRESHOLD]['num_samples']
    if valid.empty:
        return float('inf')
    return float(valid.min())


ALGO_ENV_MIN_SAMPLE = {}
print("各 (环境, 算法) 的其他指标最小绘制采样数：")
for env in ENVIRONMENTS:
    for algo in ALGORITHMS:
        thr = compute_algo_env_threshold(env, algo)
        ALGO_ENV_MIN_SAMPLE[(env, algo)] = thr
        label = ALGO_STYLE[algo]['label']
        if thr == float('inf'):
            print(f"  [{env:12s}][{label:6s}]: 成功率始终 < 50%，其他指标不绘制")
        elif thr == float('-inf'):
            print(f"  [{env:12s}][{label:6s}]: 全部采样数均可绘制")
        else:
            print(f"  [{env:12s}][{label:6s}]: 从采样数 {thr:.0f} 起绘制其他指标")
print()

# ─────────────────────────────────────────────────────────────────────
# 5. 工具函数
# ─────────────────────────────────────────────────────────────────────
def _fp(key: str) -> dict:
    cfg = FONT_CONFIG[key]
    return {'fontfamily': cfg['family'], 'fontsize': cfg['size'],
            'fontweight': cfg['weight'], 'fontstyle': cfg['style'], 'color': cfg['color']}


def _ylabel_str(metric_cfg: dict) -> str:
    unit = metric_cfg.get('ylabel_unit', '').strip()
    return f"{metric_cfg['title']} ({unit})" if unit else metric_cfg['title']


def apply_font_to_ax(ax: plt.Axes):
    tk = FONT_CONFIG['tick_label']
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontfamily(tk['family']); lbl.set_fontsize(tk['size'])
        lbl.set_fontweight(tk['weight']); lbl.set_fontstyle(tk['style'])
        lbl.set_color(tk['color'])
    al = FONT_CONFIG['axis_label']
    for obj in [ax.xaxis.label, ax.yaxis.label]:
        obj.set_fontfamily(al['family']); obj.set_fontsize(al['size'])
        obj.set_fontweight(al['weight']); obj.set_fontstyle(al['style'])
        obj.set_color(al['color'])
    st = FONT_CONFIG['subplot_title']
    ax.title.set_fontfamily(st['family']); ax.title.set_fontsize(st['size'])
    ax.title.set_fontweight(st['weight']); ax.title.set_fontstyle(st['style'])
    ax.title.set_color(st['color'])


def apply_font_to_legend(leg):
    lg = FONT_CONFIG['legend']
    for t in leg.get_texts():
        t.set_fontfamily(lg['family']); t.set_fontsize(lg['size'])
        t.set_fontweight(lg['weight']); t.set_fontstyle(lg['style'])
        t.set_color(lg['color'])


# ─────────────────────────────────────────────────────────────────────
# 6. 单个子图绘制
# ─────────────────────────────────────────────────────────────────────
def plot_env_metric(ax: plt.Axes, env_data: pd.DataFrame,
                    metric_cfg: dict, env_name: str):
    col     = metric_cfg['column']
    is_rate = metric_cfg.get('is_rate', False)
    f_succ  = metric_cfg.get('filter_success', False)

    if col not in env_data.columns:
        ax.text(0.5, 0.5, f'Column "{col}"\nnot found',
                ha='center', va='center', transform=ax.transAxes, color='red',
                fontsize=FONT_CONFIG['tick_label']['size'])
        return

    any_drawn = False
    for algo in ALGORITHMS:
        style    = ALGO_STYLE[algo]
        slice_df = env_data[env_data['algorithm'] == algo].copy()
        if f_succ:
            slice_df = slice_df[slice_df['path_success_rate'] > 0]
        if slice_df.empty:
            continue

        grouped = (slice_df.groupby('num_samples')[col]
                   .agg(['mean', 'std']).reset_index())
        grouped['std'] = grouped['std'].fillna(0)

        if not is_rate:
            min_samp = ALGO_ENV_MIN_SAMPLE.get((env_name, algo), float('-inf'))
            if min_samp == float('inf'):
                continue
            grouped = grouped[grouped['num_samples'] >= min_samp]

        if grouped.empty:
            continue

        x, y, s = (grouped['num_samples'].to_numpy(),
                   grouped['mean'].to_numpy(),
                   grouped['std'].to_numpy())

        ax.plot(x, y, marker=style['marker'], linestyle=style['linestyle'],
                color=style['color'], linewidth=LINE_WIDTH, markersize=MARKER_SIZE,
                label=style['label'], zorder=3)

        lo = np.clip(y - s, 0, 100) if is_rate else np.maximum(y - s, 0)
        hi = np.clip(y + s, 0, 100) if is_rate else y + s
        ax.fill_between(x, lo, hi, color=style['color'], alpha=0.15, zorder=2)
        any_drawn = True

    ax.grid(False)
    if is_rate:
        ax.set_ylim(0, 108)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f%%'))
        ax.axhline(y=SUCCESS_RATE_THRESHOLD, color='gray', linestyle=':',
                   linewidth=2.2, alpha=0.7, zorder=1)
    else:
        bot = metric_cfg.get('ylim_bottom', 0)
        ax.set_ylim(bottom=(bot if bot is not None else None))

    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=4, integer=True))
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=4))

    if not any_drawn:
        ax.text(0.5, 0.5, 'No data\n(success rate < 50%\nfor all algorithms)',
                ha='center', va='center', transform=ax.transAxes, color='grey',
                fontsize=FONT_CONFIG['tick_label']['size'] * 0.75,
                style='italic', multialignment='center')


# ═════════════════════════════════════════════════════════════════════
# 7. 生成大图（6行 × 4列）
# ═════════════════════════════════════════════════════════════════════
print("📌 生成大图（6×4汇总图）...")

N_ROWS, N_COLS = len(METRICS_CONFIG), len(ENVIRONMENTS)

# 图更扁：宽不变，高大幅压缩
CELL_W, CELL_H = 22, 13          # ← CELL_H: 15 → 10
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

        # ── 第一行：环境列标题 ────────────────────────────────────────
        if row_idx == 0:
            ax.set_title(ENV_DISPLAY.get(env_name, env_name),
                         pad=22, fontdict=_fp('subplot_title'))

        # ── 第一列：ylabel ────────────────────────────────────────────
        if col_idx == 0:
            ax.set_ylabel(y_label_str, fontdict=_fp('axis_label'), labelpad=24)
        else:
            ax.set_ylabel('')

        # ── 最后一行：X 轴标签 ────────────────────────────────────────
        if row_idx == N_ROWS - 1:
            ax.set_xlabel('Iterations',
                          fontdict=_fp('axis_label'), labelpad=18)
        else:
            ax.set_xlabel('')

        ax.tick_params(axis='both', which='major', pad=14, length=12)
        plot_env_metric(ax, env_data, metric_cfg, env_name=env_name)
        apply_font_to_ax(ax)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)
        ax.spines['bottom'].set_linewidth(2.2)
        ax.spines['left'].set_linewidth(2.2)


# ── 全局图例 ──────────────────────────────────────────────────────────
legend_handles = [
    plt.Line2D([0], [0],
               color=ALGO_STYLE[a]['color'],
               linestyle=ALGO_STYLE[a]['linestyle'],
               marker=ALGO_STYLE[a]['marker'],
               linewidth=LINE_WIDTH, markersize=MARKER_SIZE,
               label=ALGO_STYLE[a]['label'])
    for a in ALGORITHMS
]
leg_big = fig_big.legend(
    legend_handles,
    [ALGO_STYLE[a]['label'] for a in ALGORITHMS],
    loc='upper center',
    bbox_to_anchor=(0.5, 0.998),
    ncol=len(ALGORITHMS),
    frameon=True, framealpha=0.92, edgecolor='#cccccc',
    fontsize=FONT_CONFIG['legend']['size'],
    handlelength=4.0, handletextpad=1.2, columnspacing=3.0,
)
apply_font_to_legend(leg_big)

plt.subplots_adjust(
    top=0.940,
    bottom=0.090,       # 扁图底部留更多空间给 xlabel
    left=0.14,
    right=0.99,
    hspace=0.55,        # 行间距适当加大，防止扁图中 xlabel 与上行内容重叠
    wspace=0.28,
)

out_path_big = os.path.join(charts_dir, 'scalability_all_metrics_6x4.svg')
fig_big.savefig(out_path_big, format='svg', bbox_inches='tight')
plt.close(fig_big)
print(f"   ✅ 大图已保存: {os.path.abspath(out_path_big)}")
print("\n🎉 完成！")
