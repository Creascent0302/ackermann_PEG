"""
PRM 算法可扩展性分析：生成节点数 vs 迭代次数
支持通过命令行选择执行模式：测试采集数据 / 仅画图 / 测试并画图
"""

import argparse
import os
import time
import random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from datetime import datetime

# =====================================================================
# 根据你的实际项目路径导入算法和地图生成器
# =====================================================================
from map_generator import generate_maze_obstacles, generate_indoor_obstacles
from generator import DeltaPRM, BeamPRM, SPARS, GSRM

# =====================================================================
# 全局配置
# =====================================================================
SEED = 42
OUTPUT_DIR = "./pursuer_strategies/PRM/results/iter_vs_nodes"
CSV_FILE = os.path.join(OUTPUT_DIR, "iter_vs_nodes_data.csv")

# 环境与迭代次数配置 (扩充了 four_rooms)
ENVIRONMENTS = ['random', 'maze', 'indoor', 'four_rooms']
ITER_RANGE = {env: list(range(100, 2100, 100)) for env in ENVIRONMENTS}

ENV_DISPLAY = {
    'random':     'Cluttered Environment',
    'maze':       'Maze Environment',
    'indoor':     'Indoor Environment',
    'four_rooms': 'Narrow Passage Environment',
}

# 算法配置
ALGORITHMS = ['delta', 'beam', 'spars', 'gsrm']
ALGO_STYLE = {
    'delta': dict(color='#1f77b4', linestyle='-',  marker='o', label='Delta-PRM'),
    'beam':  dict(color='#ff7f0e', linestyle='-',  marker='s', label='Beam-PRM'),
    'spars': dict(color='#2ca02c', linestyle='--', marker='^', label='SPARS'),
    'gsrm':  dict(color='#d62728', linestyle='-.', marker='D', label='GSRM'),
}

# 字体配置 (严格参照要求)
FONT_CONFIG = {
    'subplot_title': {'family': 'Times New Roman', 'size': 74, 'weight': 'normal', 'style': 'normal', 'color': 'black'},
    'axis_label':    {'family': 'Times New Roman', 'size': 72, 'weight': 'normal', 'style': 'normal', 'color': 'black'},
    'tick_label':    {'family': 'Times New Roman', 'size': 70, 'weight': 'normal', 'style': 'normal', 'color': 'black'},
    'legend':        {'family': 'Times New Roman', 'size': 70, 'weight': 'normal', 'style': 'normal', 'color': 'black'},
    'row_label':     {'family': 'Times New Roman', 'size': 76, 'weight': 'normal', 'style': 'normal', 'color': 'black'},
}

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

# =====================================================================
# 1. 数据测试与采集逻辑
# =====================================================================
def build_environment(env_type, seed):
    np.random.seed(seed)
    random.seed(seed)

    if env_type == "maze":
        grid_width, grid_height = 49, 49
        obstacles = generate_maze_obstacles(grid_width, grid_height)
    elif env_type == "indoor":
        grid_width, grid_height = 51, 51
        obstacles = generate_indoor_obstacles(grid_width, grid_height)
    elif env_type == "random":
        grid_width, grid_height = 40, 40
        obstacles = []
        for i in range(grid_width):
            obstacles.extend([(i, 0), (i, grid_height - 1), (0, i), (grid_width - 1, i)])
        obstacle_set = set(obstacles)
        while len(obstacle_set) < int(grid_width * grid_height * 0.25):
            obstacle_set.add((np.random.randint(1, grid_width - 1), np.random.randint(1, grid_height - 1)))
        obstacles = list(obstacle_set)
    elif env_type == "four_rooms":
        # 简单模拟 Narrow Passage (Four Rooms)
        grid_width, grid_height = 40, 40
        obstacles = []
        for i in range(grid_width):
            obstacles.extend([(i, 0), (i, grid_height - 1), (0, i), (grid_width - 1, i)])
        for i in range(1, 39):
            if i not in [8, 9, 10, 29, 30, 31]:  # 留出门
                obstacles.append((20, i))
                obstacles.append((i, 20))
        obstacles = list(set(obstacles))
    else:
        raise ValueError(f"Unknown environment: {env_type}")
    
    return grid_width, grid_height, obstacles

def measure_once(algo, env, input_param, grid_width, grid_height, obstacles):
    # 此处省略具体参数映射字典，沿用之前的结构
    if algo == "delta":
        generator = DeltaPRM(grid_width, grid_height, obstacles, num_nodes=input_param, connection_radius=1.6, delta_radius=0.3)
    elif algo == "beam":
        generator = BeamPRM(grid_width, grid_height, obstacles, num_nodes=input_param, connection_radius=1.2, beam_angle_step_deg=25, beam_ray_step=0.2)
    elif algo == "spars":
        generator = SPARS(grid_width, grid_height, obstacles, num_nodes=input_param, max_failures=100, delta=0.2)
    elif algo == "gsrm":
        generator = GSRM(grid_width, grid_height, obstacles, iterations=input_param, upscale_factor=6)
    
    t0 = time.perf_counter()
    result = generator.generate_prm()
    elapsed = time.perf_counter() - t0
    
    nodes, edges = result[0], result[1]
    return len(nodes), len(edges), elapsed

def run_tests(repeat=3):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results = []
    
    for env in ENVIRONMENTS:
        actual_repeat = 1 if env in ["indoor", "four_rooms"] else repeat
        grid_w, grid_h, obs = build_environment(env, SEED)
        
        for algo in ALGORITHMS:
            print(f"\n[{env.upper()}] {algo.upper()} testing...")
            for ni in ITER_RANGE[env]:
                n_buf, e_buf, t_buf = [], [], []
                for r in range(actual_repeat):
                    run_seed = SEED + r * 1000 + ni
                    np.random.seed(run_seed)
                    random.seed(run_seed)
                    try:
                        n, e, t = measure_once(algo, env, ni, grid_w, grid_h, obs)
                    except Exception as ex:
                        print(f"  Error: {ex}")
                        n, e, t = 0, 0, 0.0
                    n_buf.append(n); e_buf.append(e); t_buf.append(t)
                
                results.append({
                    "environment": env, "algorithm": algo, "input_param": ni,
                    "nodes_mean": np.mean(n_buf), "nodes_std": np.std(n_buf),
                    "edges_mean": np.mean(e_buf), "edges_std": np.std(e_buf),
                    "time_mean": np.mean(t_buf), "time_std": np.std(t_buf)
                })
                print(f"  Param={ni:5d} -> Nodes={np.mean(n_buf):.1f}")
                
    df = pd.DataFrame(results)
    df.to_csv(CSV_FILE, index=False)
    print(f"\n✅ 测试数据已保存至: {CSV_FILE}")

# =====================================================================
# 2. 1x4 绘图逻辑
# =====================================================================
def _fp(key: str) -> dict:
    c = FONT_CONFIG[key]
    return {'fontfamily': c['family'], 'fontsize': c['size'], 'fontweight': c['weight'], 'fontstyle': c['style'], 'color': c['color']}

def apply_font(ax: plt.Axes):
    tk = FONT_CONFIG['tick_label']
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontfamily(tk['family']); lbl.set_fontsize(tk['size'])
    al = FONT_CONFIG['axis_label']
    for obj in [ax.xaxis.label, ax.yaxis.label]:
        obj.set_fontfamily(al['family']); obj.set_fontsize(al['size'])

def plot_1x4():
    if not os.path.exists(CSV_FILE):
        print(f"❌ 找不到数据文件 {CSV_FILE}，请先运行测试: python script.py --mode test")
        return
        
    df = pd.read_csv(CSV_FILE)
    
    # 图更扁：宽不变，高大幅压缩 (CELL_H 从 15 降至 10)
    CELL_W, CELL_H = 22, 10  
    N_COLS = len(ENVIRONMENTS)
    
    fig, axes = plt.subplots(1, N_COLS, figsize=(CELL_W * N_COLS, CELL_H), constrained_layout=False)
    
    for col_idx, env in enumerate(ENVIRONMENTS):
        ax = axes[col_idx]
        env_data = df[df['environment'] == env]
        
        for algo in ALGORITHMS:
            algo_data = env_data[env_data['algorithm'] == algo]
            if algo_data.empty: continue
            
            x = algo_data['input_param'].values
            y = algo_data['nodes_mean'].values
            s = algo_data['nodes_std'].values
            style = ALGO_STYLE[algo]
            
            ax.plot(x, y, marker=style['marker'], linestyle=style['linestyle'],
                    color=style['color'], linewidth=LINE_WIDTH, markersize=MARKER_SIZE,
                    label=style['label'], zorder=3)
            ax.fill_between(x, np.maximum(y - s, 0), y + s, color=style['color'], alpha=0.15, zorder=2)
            
        # ── 坐标轴与标题 ──────────────────────────────────────────
        ax.set_title(ENV_DISPLAY.get(env, env), pad=22, fontdict=_fp('subplot_title'))
        ax.set_xlabel('Iterations / Samples', fontdict=_fp('axis_label'), labelpad=18)
        
        if col_idx == 0:
            ax.set_ylabel('Generated Nodes Count', fontdict=_fp('axis_label'), labelpad=24)
        
        # ── 编号 (a)-(d) 控制在左上角外侧 ───────────────────────
        label_char = chr(ord('a') + col_idx)
        ax.text(-0.02, 1.0, f'({label_char})', transform=ax.transAxes,
                ha='right', va='top', fontdict=_fp('row_label'))
        
        # ── 样式调整 ──────────────────────────────────────────────
        ax.tick_params(axis='both', which='major', pad=14, length=12)
        ax.set_ylim(bottom=0)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=4, integer=True))
        ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=4))
        
        apply_font(ax)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_linewidth(2.2)
        ax.spines['left'].set_linewidth(2.2)

    # ── 全局图例配置 ───────────────────────────────────────────────
    legend_handles = [plt.Line2D([0], [0], color=ALGO_STYLE[a]['color'], 
                                 linestyle=ALGO_STYLE[a]['linestyle'], marker=ALGO_STYLE[a]['marker'],
                                 linewidth=LINE_WIDTH, markersize=MARKER_SIZE) for a in ALGORITHMS]
    
    leg = fig.legend(legend_handles, [ALGO_STYLE[a]['label'] for a in ALGORITHMS],
                     loc='upper center', bbox_to_anchor=(0.5, 1.18), ncol=len(ALGORITHMS),
                     frameon=False, fontsize=FONT_CONFIG['legend']['size'],
                     handlelength=4.0, handletextpad=1.2, columnspacing=3.0)
    
    lg_cfg = FONT_CONFIG['legend']
    for t in leg.get_texts():
        t.set_fontfamily(lg_cfg['family']); t.set_fontsize(lg_cfg['size'])

    # 预留空间给顶部的图例和底部的标签
    plt.subplots_adjust(top=0.82, bottom=0.20, left=0.06, right=0.98, wspace=0.25)
    
    # 保存 SVG 和 PDF
    svg_path = os.path.join(OUTPUT_DIR, "scalability_nodes_1x4.svg")
    pdf_path = os.path.join(OUTPUT_DIR, "scalability_nodes_1x4.pdf")
    
    fig.savefig(svg_path, format='svg', bbox_inches='tight')
    fig.savefig(pdf_path, format='pdf', bbox_inches='tight')
    plt.close(fig)
    
    print(f"✅ 图表已生成！\n ➜ {svg_path}\n ➜ {pdf_path}")

# =====================================================================
# 3. 主入口 (命令行控制)
# =====================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PRM Scalability Evaluation (Iterations vs Nodes)")
    parser.add_argument('--mode', type=str, choices=['test', 'plot', 'all'], default='all',
                        help="Select mode: 'test' to run algo, 'plot' to draw charts, 'all' for both.")
    args = parser.parse_args()
    
    print("=" * 60)
    print(f"🚀 开始执行模块，当前模式: {args.mode.upper()}")
    print("=" * 60)
    
    if args.mode in ['test', 'all']:
        run_tests()
        
    if args.mode in ['plot', 'all']:
        plot_1x4()