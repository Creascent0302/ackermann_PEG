"""
可扩展性测试与数据采集脚本
功能：执行跑测，保存 CSV 数据，并在特定参数下调用 PRMRenderer (无头模式) 留存测试地图快照
支持命令行快速验证模式：python test_script.py --fast
"""

import csv
import os
import argparse
from datetime import datetime
import time
import random
import numpy as np
import sys
import traceback

sys.path.append('.')
from config import ENV_CONFIG
from generator import *
from map_generator import generate_maze_obstacles, generate_indoor_obstacles

# =====================================================================
# 辅助函数：快照留存
# =====================================================================
def save_test_snapshot(grid_width, grid_height, nodes, edges, obstacles,
                       env_name, algo_name, num_samples, seed, path_nodes=None):
    snapshot_dir = './eval_snapshots'
    os.makedirs(snapshot_dir, exist_ok=True)
    filename = os.path.join(snapshot_dir,
                            f"snap_{env_name}_{algo_name}_n{num_samples}.pdf")
    try:
        renderer = PRMRenderer(grid_width, grid_height, cell_size=15, headless=True)
        highlight_paths = [path_nodes] if (path_nodes and len(path_nodes) > 1) else []
        renderer.save_image(
            nodes=nodes, edges=edges, obstacles=obstacles,
            filepath=filename, medial_axis_paths=highlight_paths,
            env=env_name, algorithm=algo_name
        )
    except Exception as e:
        print(f"\n[警告] 渲染快照 {filename} 失败: {e}")

def save_scalability_metrics_to_file(metrics, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    file_exists = os.path.exists(filepath)
    with open(filepath, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                'timestamp', 'algorithm', 'environment', 'num_samples', 'seed',
                'generation_time', 'search_time', 'path_length', 'edges_count',
                'path_success_rate',
                'spatial_coverage'          # ★ 改：node_utilization → spatial_coverage
            ])
        writer.writerow([
            metrics['timestamp'], metrics['algorithm'], metrics['environment'],
            metrics['num_samples'], metrics['seed'],
            metrics['generation_time'], metrics['search_time'], metrics['path_length'],
            metrics['edges_count'], metrics['path_success'],
            metrics['spatial_coverage']     # ★ 改：node_utilization → spatial_coverage
        ])

def log_failure(log_filepath, failure_type, algorithm, environment,
                num_samples, seed, start=None, goal=None,
                exception_msg=None, exception_tb=None):
    os.makedirs(os.path.dirname(log_filepath), exist_ok=True)
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    with open(log_filepath, 'a', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write(f"[{timestamp}] FAILURE_TYPE : {failure_type}\n")
        f.write(f"  algorithm   : {algorithm}\n")
        f.write(f"  environment : {environment}\n")
        f.write(f"  num_samples : {num_samples}\n")
        f.write(f"  seed        : {seed}\n")

        if start is not None:
            f.write(f"  start       : {start}\n")
        if goal is not None:
            f.write(f"  goal        : {goal}\n")

        if failure_type == 'NO_POINT_PAIRS':
            f.write("  reason      : generate_valid_point_pairs() returned empty list.\n")
            f.write("                可能原因: 节点不足 / 障碍物密度过高 / GSRM未收敛\n")
        elif failure_type == 'PATH_NOT_FOUND':
            f.write("  reason      : find_path() returned None — 图中起终点不连通\n")
            f.write("                可能原因: 图连通性不足 / 起终点孤立 / 节点数过少\n")
        elif failure_type == 'EXCEPTION':
            f.write(f"  exception   : {exception_msg}\n")
            if exception_tb:
                f.write("  traceback   :\n")
                for line in exception_tb.splitlines():
                    f.write(f"    {line}\n")

        f.write("=" * 70 + "\n\n")

# =====================================================================
# 环境生成
# =====================================================================
def generate_environment_obstacles(environment_type):
    ENV_OBSTACLE_SEED = 42
    np_state = np.random.get_state()
    py_state = random.getstate()

    np.random.seed(ENV_OBSTACLE_SEED)
    random.seed(ENV_OBSTACLE_SEED)

    if environment_type == "maze":
        grid_width, grid_height = 49, 49
        obstacles = generate_maze_obstacles(grid_width, grid_height)
    elif environment_type == "indoor":
        grid_width, grid_height = 51, 51
        obstacles = generate_indoor_obstacles(grid_width, grid_height)
    elif environment_type == "random":
        grid_width, grid_height = 40, 40
        num_obstacles = int(grid_width * grid_height * 0.25)
        obstacles = []
        for i in range(grid_width):
            obstacles.extend([(i, 0), (i, grid_height - 1),
                               (0, i), (grid_width - 1, i)])
        while len(obstacles) < num_obstacles:
            x = np.random.randint(1, grid_width - 1)
            y = np.random.randint(1, grid_height - 1)
            if (x, y) not in obstacles:
                obstacles.append((x, y))
    else:
        raise ValueError(f"Unknown environment type: {environment_type}")

    np.random.set_state(np_state)
    random.setstate(py_state)

    ENV_CONFIG['gridnum_width']  = grid_width
    ENV_CONFIG['gridnum_height'] = grid_height

    return grid_width, grid_height, obstacles

# =====================================================================
# 算法实例化
# =====================================================================
def _create_generator(algorithm_name, environment_type,
                      grid_width, grid_height, obstacles, num_samples):
    if algorithm_name == "delta":
        params = {
            "random": dict(connection_radius=1.6, max_failures=100, delta_radius=0.2),
            "maze":   dict(connection_radius=1.6, max_failures=100, delta_radius=0.3),
            "indoor": dict(connection_radius=1.4, max_failures=100, delta_radius=0.3)
        }
        return DeltaPRM(grid_width, grid_height, obstacles,
                        num_nodes=num_samples, **params[environment_type])

    elif algorithm_name == "beam":
        params = {
            "random": dict(connection_radius=1.2, beam_angle_step_deg=2.6,
                           beam_ray_step=0.08, min_connection_radius=0.3),
            "maze":   dict(connection_radius=1.2, beam_angle_step_deg=30,
                           beam_ray_step=0.25, min_connection_radius=0.3),
            "indoor": dict(connection_radius=2.0, beam_angle_step_deg=25,
                           beam_ray_step=0.2,  min_connection_radius=0.4)
        }
        return BeamPRM(grid_width, grid_height, obstacles,
                       num_nodes=num_samples, **params[environment_type])

    elif algorithm_name == "spars":
        params = {
            "random": dict(max_failures=100, delta=0.2),
            "maze":   dict(max_failures=100, delta=0.15,
                           visibility_radius=0.8, connection_radius=0.6),
            "indoor": dict(max_failures=200, delta=0.2,
                           visibility_radius=1.4, connection_radius=1.0)
        }
        return SPARS2(grid_width, grid_height, obstacles,
                      num_nodes=num_samples, **params[environment_type])

    elif algorithm_name == "gsrm":
        return GSRM(grid_width, grid_height, obstacles, iterations=num_samples)

    else:
        raise ValueError(f"Unknown algorithm: {algorithm_name}")

# =====================================================================
# 核心测试函数
# =====================================================================
def run_scalability_test(algorithm_name, environment_type,
                         num_samples, seed, num_path_tests, log_filepath):
    grid_width, grid_height, obstacles = generate_environment_obstacles(environment_type)
    np.random.seed(seed)
    random.seed(seed)

    # ── 建图（generation_time = 仿真/采样 + 建邻接表，不含寻路）──────
    t0 = time.time()
    generator = _create_generator(algorithm_name, environment_type,
                                  grid_width, grid_height, obstacles, num_samples)
    if algorithm_name == "beam":
        result = generator.generate_prm()
        nodes, edges = result[0], result[1]
    else:
        nodes, edges = generator.generate_prm()
    generation_time = time.time() - t0

    # ── 取点 ──────────────────────────────────────────────────────────
    point_pairs = generator.generate_valid_point_pairs(num_path_tests)

    if not point_pairs:
        print(f"\n  -> [警告] {algorithm_name} 在 {environment_type} "
              f"无法生成有效点对，跳过寻路测试")
        log_failure(log_filepath, 'NO_POINT_PAIRS',
                    algorithm_name, environment_type, num_samples, seed)

    # ── 寻路（search_time 来自 find_path 内部精确计时）────────────────
    successful_paths   = []
    total_path_tests   = 0
    first_success_path = None

    for start, goal in point_pairs:
        total_path_tests += 1
        try:
            path_nodes, path_edges, path_length, search_time = \
                generator.find_path(start, goal)

            if path_nodes and len(path_nodes) > 0:
                successful_paths.append({
                    'path_length': path_length,
                    'search_time': search_time   # ★ 来自算法内部，精确计时
                })
                if first_success_path is None:
                    first_success_path = path_nodes
            else:
                log_failure(log_filepath, 'PATH_NOT_FOUND',
                            algorithm_name, environment_type, num_samples, seed,
                            start=start, goal=goal)

        except Exception as e:
            tb_str = traceback.format_exc()
            print(f"\n  -> [寻路内部错误] {start} to {goal}: {e}")
            log_failure(log_filepath, 'EXCEPTION',
                        algorithm_name, environment_type, num_samples, seed,
                        start=start, goal=goal,
                        exception_msg=str(e), exception_tb=tb_str)

    # ── 快照留存 ──────────────────────────────────────────────────────
    if seed == 100 and num_samples == 1000:
        save_test_snapshot(grid_width, grid_height, nodes, edges, obstacles,
                           environment_type, algorithm_name,
                           num_samples, seed, first_success_path)

    # ── 基础指标 ──────────────────────────────────────────────────────
    success_rate = (len(successful_paths) / total_path_tests * 100) \
                   if total_path_tests > 0 else 0.0
    p_len  = (sum(p['path_length'] for p in successful_paths) / len(successful_paths)) \
             if successful_paths else 0.0
    s_time = (sum(p['search_time']  for p in successful_paths) / len(successful_paths)) \
             if successful_paths else 0.0
    edges_count = len(edges)

    # ── 节点利用率（★ 仅保留此一个高级指标）─────────────────────────
    # ── 单位节点空间覆盖率（替换原节点利用率）────────────────────────
    spatial_coverage = 0.0
    try:
        if hasattr(generator, 'calculate_spatial_coverage'):      # ★ 改
            ur = generator.calculate_spatial_coverage(num_test_paths=20)
            spatial_coverage = ur['avg_spatial_coverage'] if isinstance(ur, dict) else ur
    except Exception as e:
        print(f"\n  -> [空间覆盖率 计算报错]: {e}")
    # ★ 已删除: dispersion / clearance 计算块

    return {
        'timestamp':        datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'algorithm':        algorithm_name,
        'environment':      environment_type,
        'num_samples':      num_samples,
        'seed':             seed,
        'generation_time':  round(generation_time, 4),
        'search_time':      round(s_time, 4),
        'path_length':      round(p_len, 4),
        'edges_count':      edges_count,
        'path_success':     round(success_rate, 2),
        'spatial_coverage': round(spatial_coverage, 6), 
    }

def main():
    parser = argparse.ArgumentParser(description="PRM Algorithms Scalability Evaluator")
    parser.add_argument('--fast', action='store_true',
                        help="快速验证模式，仅用极少样本检验代码正确性")
    args = parser.parse_args()

    if args.fast:
        print("\n" + "=" * 50)
        print("⚠️  开启 FAST_MODE 快速验证模式")
        print("=" * 50 + "\n")
        algorithms     = ['delta', 'beam', 'spars', 'gsrm']
        environments   = ['random', 'maze', 'indoor']
        sample_counts  = [100, 200, 300, 500, 800, 1000, 1500, 2000]
        seeds          = [100]
        num_path_tests = 5
    else:
        print("\n" + "=" * 50)
        print("🚀  开启 FULL_MODE 完整可扩展性评测")
        print("=" * 50 + "\n")
        algorithms     = ['delta', 'beam', 'spars', 'gsrm']
        environments   = ['random', 'maze', 'indoor']
        sample_counts  = [100, 200, 300, 500, 800, 1000, 1500, 2000]
        seeds          = list(range(100, 105))
        num_path_tests = 50

    os.makedirs("./pursuer_strategies/PRM/results", exist_ok=True)
    mode_str   = "FAST" if args.fast else "FULL"
    time_str   = datetime.now().strftime('%Y%m%d%H%M')
    output_file = (f"./pursuer_strategies/PRM/results/"
                   f"scalability_evaluation_{mode_str}_{time_str}.csv")
    log_file    = (f"./pursuer_strategies/PRM/results/"
                   f"scalability_evaluation_{mode_str}_{time_str}_failures.log")

    for f in [output_file, log_file]:
        if os.path.exists(f):
            os.remove(f)

    with open(log_file, 'w', encoding='utf-8') as f:
        f.write("PRM Scalability Test — Failure Log\n")
        f.write(f"Run time   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Mode       : {mode_str}\n")
        f.write(f"Algorithms : {algorithms}\n")
        f.write(f"Envs       : {environments}\n")
        f.write(f"Samples    : {sample_counts}\n")
        f.write(f"Seeds      : {seeds}\n")
        f.write(f"Path tests : {num_path_tests} per config\n")
        f.write("\n" + "=" * 70 + "\n\n")

    total = (len(algorithms) * len(environments)
             * len(sample_counts) * len(seeds))
    done = ok = 0

    for env in environments:
        for algo in algorithms:
            for ns in sample_counts:
                for seed in seeds:
                    done += 1
                    print(f"\r进度: [{done}/{total}] "
                          f"{algo} on {env} (n={ns}, seed={seed})", end="")
                    try:
                        m = run_scalability_test(
                            algo, env, ns, seed, num_path_tests,
                            log_filepath=log_file
                        )
                        save_scalability_metrics_to_file(m, output_file)
                        ok += 1
                    except Exception as e:
                        print(f"\n\n🚨 [致命崩溃] {algo} 在 {env} "
                              f"(样本={ns}, 种子={seed}) 运行失败！")
                        traceback.print_exc()
                        log_failure(log_file, 'EXCEPTION',
                                    algo, env, ns, seed,
                                    exception_msg=str(e),
                                    exception_tb=traceback.format_exc())
                        print()

    print(f"\n\n✅ 评测完成！成功率 {ok}/{total}。")
    print(f"📄 数据已保存至 : {output_file}")
    print(f"📋 失败日志保存至: {log_file}")

if __name__ == "__main__":
    main()