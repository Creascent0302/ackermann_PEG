"""
算法参数 vs 生成节点数 关系分析
测试 DeltaPRM / BeamPRM / SPARS (目标节点数) 与 GSRM (仿真迭代次数)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec
import time
import os
import random
from datetime import datetime
from map_generator import generate_maze_obstacles, generate_indoor_obstacles
# 导入所有算法
from generator import DeltaPRM, BeamPRM, SPARS, GSRM

# ============================================================
# 配置区
# ============================================================

SEED = 42
OUTPUT_DIR = "./pursuer_strategies/PRM/results/iter_vs_nodes"

# 测试的输入参数序列 (对于普通PRM是采样数，对于GSRM是迭代次数)
# 由于 GSRM 迭代 2000 次可能极其耗时，如果跑得太慢，你可以适当调小这里的上限
ITER_RANGE = {
    "random": list(range(100, 2100, 100)),
    "maze":   list(range(100, 2100, 100)),
    "indoor": list(range(100, 2100, 100)),
}

# 算法颜色和标记
ALGO_STYLE = {
    "delta": {"color": "#E63946", "marker": "o", "linestyle": "-",  "label": "DeltaPRM (Samples)"},
    "beam":  {"color": "#2196F3", "marker": "s", "linestyle": "--", "label": "BeamPRM (Samples)"},
    "spars": {"color": "#4CAF50", "marker": "^", "linestyle": "-.", "label": "SPARS (Samples)"},
    "gsrm":  {"color": "#9C27B0", "marker": "d", "linestyle": ":",  "label": "GSRM (Iterations)"},
}

ENV_TITLES = {
    "random": "Random Environment (40×40)",
    "maze":   "Maze Environment (49×49)",
    "indoor": "Indoor Environment (51×51)",
}

# 全局环境配置 (为 GSRM 提供必要的 cell_size 等参数)
ENV_CONFIG = {'cell_size': 1.0}

# ============================================================
# 环境构建
# ============================================================

def build_environment(environment_type, seed):
    np.random.seed(seed)
    random.seed(seed)

    if environment_type == "maze":
        grid_width, grid_height = 49, 49
        obstacles = generate_maze_obstacles(grid_width, grid_height)

    elif environment_type == "indoor":
        grid_width, grid_height = 51, 51
        obstacles = generate_indoor_obstacles(grid_width, grid_height)

    elif environment_type == "random":
        grid_width, grid_height = 40, 40
        total_cells = grid_width * grid_height
        num_obstacles = int(total_cells * 0.25)
        obstacles = []
        for i in range(grid_width):
            obstacles.append((i, 0))
            obstacles.append((i, grid_height - 1))
            obstacles.append((0, i))
            obstacles.append((grid_width - 1, i))
        obstacle_set = set(obstacles)
        while len(obstacle_set) < num_obstacles:
            x = np.random.randint(1, grid_width - 1)
            y = np.random.randint(1, grid_height - 1)
            obstacle_set.add((x, y))
        obstacles = list(obstacle_set)
    else:
        raise ValueError(f"Unknown environment type: {environment_type}")

    return grid_width, grid_height, obstacles


# ============================================================
# 单次测量
# ============================================================

def measure_once(algorithm_name, environment_type, input_param,
                 grid_width, grid_height, obstacles):
    """
    input_param: 对于 delta/beam/spars 是 num_nodes; 对于 gsrm 是 iterations
    """
    if algorithm_name == "delta":
        params = {
            "random": dict(connection_radius=1.6, max_failures=100, delta_radius=0.2),
            "maze":   dict(connection_radius=1.6, max_failures=100, delta_radius=0.3),
            "indoor": dict(connection_radius=1.4, max_failures=100, delta_radius=0.3),
        }[environment_type]
        generator = DeltaPRM(grid_width, grid_height, obstacles, num_nodes=input_param, **params)

    elif algorithm_name == "beam":
        params = {
            "random": dict(connection_radius=1.2, beam_angle_step_deg=2.6, beam_ray_step=0.08, min_connection_radius=0.3),
            "maze":   dict(connection_radius=1.2, beam_angle_step_deg=30, beam_ray_step=0.25, min_connection_radius=0.3),
            "indoor": dict(connection_radius=2.0, beam_angle_step_deg=25, beam_ray_step=0.2, min_connection_radius=0.4),
        }[environment_type]
        generator = BeamPRM(grid_width, grid_height, obstacles, num_nodes=input_param, **params)

    elif algorithm_name == "spars":
        params = {
            "random": dict(max_failures=100, delta=0.2),
            "maze":   dict(max_failures=100, delta=0.15, visibility_radius=0.8, connection_radius=0.6),
            "indoor": dict(max_failures=200, delta=0.2, visibility_radius=1.4, connection_radius=1.0),
        }[environment_type]
        generator = SPARS(grid_width, grid_height, obstacles, num_nodes=input_param, **params)
        
    elif algorithm_name == "gsrm":
        # GSRM 将 input_param 作为 iterations 传入，并且不传递 num_nodes
        params = {
            "random": dict(min_node_dist=0.5, peak_min_distance=8, upscale_factor=6),
            "maze":   dict(min_node_dist=0.5, peak_min_distance=8, upscale_factor=6),
            "indoor": dict(min_node_dist=0.5, peak_min_distance=8, upscale_factor=6),
        }[environment_type]
        generator = GSRM(grid_width, grid_height, obstacles, iterations=input_param, **params)

    else:
        raise ValueError(f"Unknown algorithm: {algorithm_name}")

    t0 = time.perf_counter()
    result = generator.generate_prm()
    elapsed = time.perf_counter() - t0

    if algorithm_name == "beam":
        nodes, edges = result[0], result[1]
    else:
        nodes, edges = result[0], result[1]

    return len(nodes), len(edges), elapsed


# ============================================================
# 核心循环
# ============================================================

def run_all_experiments(seed=SEED, repeat=3):
    results = {}

    for env in ["random", "maze", "indoor"]:
        results[env] = {}
        actual_repeat = 1 if env == "indoor" else repeat
        grid_width, grid_height, obstacles = build_environment(env, seed)

        for algo in ["delta", "beam", "spars", "gsrm"]:
            iter_list  = ITER_RANGE[env]
            nodes_runs, edges_runs, time_runs = [], [], []

            print(f"\n[{env.upper()}] {algo.upper()} — {len(iter_list)} points × {actual_repeat} repeats")

            for ni, param_val in enumerate(iter_list):
                n_buf, e_buf, t_buf = [], [], []

                for r in range(actual_repeat):
                    run_seed = seed + r * 1000 + ni
                    np.random.seed(run_seed)
                    random.seed(run_seed)

                    try:
                        # 对于 GSRM，这里通常会屏蔽内部的 print 输出以免刷屏
                        # sys.stdout = open(os.devnull, 'w') (如果需要可取消注释)
                        n, e, t = measure_once(algo, env, param_val, grid_width, grid_height, obstacles)
                    except Exception as ex:
                        print(f"  ⚠ Error at param={param_val}, repeat={r}: {ex}")
                        n, e, t = 0, 0, 0.0

                    n_buf.append(n)
                    e_buf.append(e)
                    t_buf.append(t)

                nodes_runs.append(n_buf)
                edges_runs.append(e_buf)
                time_runs.append(t_buf)

                print(f"  Param={param_val:5d}  →  nodes={np.mean(n_buf):.1f}  "
                      f"edges={np.mean(e_buf):.1f}  time={np.mean(t_buf):.3f}s")

            nodes_arr = np.array(nodes_runs)
            edges_arr = np.array(edges_runs)
            time_arr  = np.array(time_runs)

            results[env][algo] = {
                "iter_list":  iter_list,
                "nodes_mean": nodes_arr.mean(axis=1).tolist(),
                "nodes_std":  nodes_arr.std(axis=1).tolist(),
                "edges_mean": edges_arr.mean(axis=1).tolist(),
                "edges_std":  edges_arr.std(axis=1).tolist(),
                "time_mean":  time_arr.mean(axis=1).tolist(),
                "time_std":   time_arr.std(axis=1).tolist(),
            }
    return results


# ============================================================
# 绘图
# ============================================================

def plot_results(results, output_dir=OUTPUT_DIR):
    os.makedirs(output_dir, exist_ok=True)
    envs  = ["random", "maze", "indoor"]
    algos = ["delta", "beam", "spars", "gsrm"]

    # ── 图1：生成节点数趋势 ──────────
    fig1, axes1 = plt.subplots(1, 3, figsize=(18, 5))
    fig1.suptitle("Input Parameter vs. Generated Nodes", fontsize=14, fontweight="bold")

    for col, env in enumerate(envs):
        ax = axes1[col]
        for algo in algos:
            d  = results[env][algo]
            xs = d["iter_list"]
            ys = np.array(d["nodes_mean"])
            ye = np.array(d["nodes_std"])
            st = ALGO_STYLE[algo]

            ax.plot(xs, ys, color=st["color"], marker=st["marker"],
                    linestyle=st["linestyle"], linewidth=1.8, markersize=4, label=st["label"], zorder=3)
            ax.fill_between(xs, ys - ye, ys + ye, color=st["color"], alpha=0.15, zorder=2)

        # 仅针对采样类算法绘制理想对角线
        diag = np.array([min(xs), max(xs)])
        ax.plot(diag, diag, color="gray", linestyle=":", linewidth=1, label="y = x (Sampling Ideal)", zorder=1)

        ax.set_title(ENV_TITLES[env], fontsize=11)
        ax.set_xlabel("Input Parameter (Samples / Iters)", fontsize=10)
        ax.set_ylabel("Generated Nodes" if col == 0 else "", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig1.savefig(os.path.join(output_dir, "param_vs_nodes.pdf"), bbox_inches="tight", dpi=150)

    # ── 图2：综合对比矩阵 (4行3列) ──────────
    fig3 = plt.figure(figsize=(18, 18))
    gs   = GridSpec(4, 3, figure=fig3, hspace=0.45, wspace=0.35)
    fig3.suptitle("Performance Metrics per Algorithm × Environment", fontsize=15, fontweight="bold")

    row_labels = ["DeltaPRM", "BeamPRM", "SPARS", "GSRM"]
    metrics    = [
        ("nodes_mean", "nodes_std", "Nodes"),
        ("edges_mean", "edges_std", "Edges"),
        ("time_mean",  "time_std",  "Time (s)"),
    ]

    for row, algo in enumerate(algos):
        for col, env in enumerate(envs):
            ax  = fig3.add_subplot(gs[row, col])
            d   = results[env][algo]
            xs  = d["iter_list"]
            st  = ALGO_STYLE[algo]

            for mi, (mean_key, std_key, ylabel) in enumerate(metrics):
                ys = np.array(d[mean_key])
                ye = np.array(d[std_key])
                color = [st["color"], "#FF9800", "#9C27B0"][mi]

                ax.plot(xs, ys, color=color, linewidth=1.5, linestyle=["-", "--", ":"][mi], label=ylabel)
                ax.fill_between(xs, ys - ye, ys + ye, color=color, alpha=0.12)

            # 为 GSRM 单独修改 X 轴标签以避免歧义
            xlabel = "Iterations" if algo == "gsrm" else "Sample Target"
            ax.set_title(f"{row_labels[row]}  |  {env}", fontsize=10)
            ax.set_xlabel(xlabel, fontsize=9)
            if col == 0: ax.set_ylabel("Metrics", fontsize=9)
            ax.legend(fontsize=8, loc="upper left")
            ax.grid(True, alpha=0.25)

    fig3.savefig(os.path.join(output_dir, "full_matrix.pdf"), bbox_inches="tight", dpi=150)
    plt.show()

# ============================================================
# 数据持久化
# ============================================================

def save_csv(results, output_dir=OUTPUT_DIR):
    import csv
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "iter_vs_nodes_data.csv")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "environment", "algorithm", "input_param",
            "nodes_mean", "nodes_std", "edges_mean", "edges_std", "time_mean", "time_std"
        ])
        for env in results:
            for algo in results[env]:
                d  = results[env][algo]
                xs = d["iter_list"]
                for i, ni in enumerate(xs):
                    writer.writerow([
                        env, algo, ni,
                        round(d["nodes_mean"][i], 2), round(d["nodes_std"][i], 2),
                        round(d["edges_mean"][i], 2), round(d["edges_std"][i], 2),
                        round(d["time_mean"][i],  4), round(d["time_std"][i],  4)
                    ])

# ============================================================
# 主入口
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("算法参数 vs 生成节点数 实验 (含 GSRM)")
    print("=" * 60)
    
    results = run_all_experiments(seed=SEED, repeat=3)
    save_csv(results)
    plot_results(results)
    print("\n测试完成。数据与图表已保存至:", OUTPUT_DIR)