"""
可扩展性测试与数据采集脚本
功能：执行跑测，保存 CSV 数据，并在特定参数下调用 PRMRenderer (无头模式) 留存测试地图快照
(移除了 odrm 算法)
"""

import csv
import os
from datetime import datetime
import time
import random
import numpy as np
import sys

sys.path.append('.')
from config import ENV_CONFIG
from generator import *
from map_generator import generate_maze_obstacles, generate_indoor_obstacles

# =====================================================================
# 辅助函数：快照留存
# =====================================================================
def save_test_snapshot(grid_width, grid_height, nodes, edges, obstacles, env_name, algo_name, num_samples, seed, path_nodes=None):
    """
    使用原生的 PRMRenderer 在远端服务器上安全地生成 PDF 快照。
    强制开启 headless=True 避免 Pygame 唤起窗口导致的死锁。
    """
    snapshot_dir = './eval_snapshots'
    os.makedirs(snapshot_dir, exist_ok=True)
    filename = os.path.join(snapshot_dir, f"snap_{env_name}_{algo_name}_n{num_samples}.pdf")
    
    try:
        # 强制开启无头模式
        renderer = PRMRenderer(grid_width, grid_height, cell_size=15, headless=True)
        # 将测试成功的路径包装为 medial_axis_paths 传入以渲染加粗路径线
        highlight_paths = [path_nodes] if (path_nodes and len(path_nodes) > 1) else []
        
        renderer.save_image(
            nodes=nodes, 
            edges=edges, 
            obstacles=obstacles, 
            filepath=filename,
            medial_axis_paths=highlight_paths, 
            env=env_name, 
            algorithm=algo_name
        )
    except Exception as e:
        print(f"\n[警告] 渲染快照 {filename} 失败，可能缺少相关依赖: {e}")


# =====================================================================
# 核心测试逻辑
# =====================================================================
def save_scalability_metrics_to_file(metrics, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    file_exists = os.path.exists(filepath)

    with open(filepath, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                'timestamp', 'algorithm', 'environment', 'num_samples', 'seed',
                'generation_time', 'actual_nodes_count', 'edges_count', 'path_length',
                'path_nodes_count', 'search_time', 'path_success', 'dispersion',
                'node_utilization', 'clearance'
            ])
        writer.writerow([
            metrics['timestamp'], metrics['algorithm'], metrics['environment'],
            metrics['num_samples'], metrics['seed'], metrics['generation_time'],
            metrics['actual_nodes_count'], metrics['edges_count'], metrics['path_length'],
            metrics['path_nodes_count'], metrics['search_time'], metrics['path_success'],
            metrics['dispersion'], metrics['node_utilization'], metrics['clearance']
        ])

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
        total_cells = grid_width * grid_height
        num_obstacles = int(total_cells * 0.25)
        obstacles = []
        for i in range(grid_width):
            obstacles.extend([(i, 0), (i, grid_height - 1), (0, i), (grid_width - 1, i)])
        while len(obstacles) < num_obstacles:
            x = np.random.randint(1, grid_width - 1)
            y = np.random.randint(1, grid_height - 1)
            if (x, y) not in obstacles:
                obstacles.append((x, y))
    else:
        raise ValueError(f"Unknown environment type: {environment_type}")

    np.random.set_state(np_state)
    random.setstate(py_state)

    ENV_CONFIG['gridnum_width'] = grid_width
    ENV_CONFIG['gridnum_height'] = grid_height

    return grid_width, grid_height, obstacles

def _create_generator(algorithm_name, environment_type, grid_width, grid_height, obstacles, num_samples):
    if algorithm_name == "delta":
        params = {"random": dict(connection_radius=1.6, max_failures=100, delta_radius=0.2), 
                  "maze": dict(connection_radius=1.6, max_failures=100, delta_radius=0.3), 
                  "indoor": dict(connection_radius=1.4, max_failures=100, delta_radius=0.3)}
        return DeltaPRM(grid_width, grid_height, obstacles, num_nodes=num_samples, **params[environment_type])
    elif algorithm_name == "beam":
        params = {"random": dict(connection_radius=1.2, beam_angle_step_deg=2.6, beam_ray_step=0.08, min_connection_radius=0.3), 
                  "maze": dict(connection_radius=1.2, beam_angle_step_deg=30, beam_ray_step=0.25, min_connection_radius=0.3), 
                  "indoor": dict(connection_radius=2.0, beam_angle_step_deg=25, beam_ray_step=0.2, min_connection_radius=0.4)}
        return BeamPRM(grid_width, grid_height, obstacles, num_nodes=num_samples, **params[environment_type])
    elif algorithm_name == "spars":
        params = {"random": dict(max_failures=100, delta=0.2), 
                  "maze": dict(max_failures=100, delta=0.15, visibility_radius=0.8, connection_radius=0.6), 
                  "indoor": dict(max_failures=200, delta=0.2, visibility_radius=1.4, connection_radius=1.0)}
        return SPARS2(grid_width, grid_height, obstacles, num_nodes=num_samples, **params[environment_type])
    elif algorithm_name == "gsrm":
        return GSRM(grid_width, grid_height, obstacles, iterations=num_samples)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm_name}")

def run_scalability_test(algorithm_name, environment_type, num_samples, seed, fixed_point_pairs=None):
    grid_width, grid_height, obstacles = generate_environment_obstacles(environment_type)
    np.random.seed(seed)
    random.seed(seed)

    start_time = time.time()
    generator = _create_generator(algorithm_name, environment_type, grid_width, grid_height, obstacles, num_samples)

    if algorithm_name == "beam":
        result = generator.generate_prm()
        nodes, edges = result[0], result[1] if len(result) >= 6 else result[:2]
    else:
        nodes, edges = generator.generate_prm()

    generation_time = time.time() - start_time

    num_path_tests = 10
    successful_paths = []
    total_path_tests = 0

    point_pairs = fixed_point_pairs if fixed_point_pairs else generator.generate_valid_point_pairs(num_path_tests)
    first_success_path = None

    for start, goal in point_pairs:
        total_path_tests += 1
        try:
            path_nodes, path_edges, path_length, search_time = generator.find_path(start, goal)
            if path_nodes and len(path_nodes) > 0:
                successful_paths.append({
                    'path_length': path_length, 'path_nodes_count': len(path_nodes), 'search_time': search_time
                })
                if first_success_path is None: 
                    first_success_path = path_nodes
        except Exception: pass

    # 【快照截取点】特定 seed (100) 且样本量为 1000 时，渲染并保存图片
    if seed == 100 and num_samples == 1000:
        save_test_snapshot(grid_width, grid_height, nodes, edges, obstacles, environment_type, algorithm_name, num_samples, seed, first_success_path)

    success_rate = (len(successful_paths) / num_path_tests * 100 if total_path_tests > 0 else 0.0)
    p_len = sum(p['path_length'] for p in successful_paths) / len(successful_paths) if successful_paths else 0.0
    pn_cnt = sum(p['path_nodes_count'] for p in successful_paths) / len(successful_paths) if successful_paths else 0
    s_time = sum(p['search_time'] for p in successful_paths) / len(successful_paths) if successful_paths else 0.0

    dispersion = generator.cal_dispersion(num_samples=500) if generator else 0.0
    ur = generator.calculate_node_utilization(num_test_paths=100) if hasattr(generator, 'calculate_node_utilization') else None
    node_utilization = ur['avg_utilization'] if ur else 0.0
    clearance = generator.cal_clearance() if hasattr(generator, 'cal_clearance') else 0.0

    return {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'algorithm': algorithm_name,
        'environment': environment_type, 'num_samples': num_samples, 'seed': seed,
        'generation_time': generation_time, 'actual_nodes_count': len(nodes), 'edges_count': len(edges),
        'path_length': p_len, 'path_nodes_count': pn_cnt, 'search_time': s_time,
        'path_success': success_rate, 'dispersion': round(dispersion, 4),
        'node_utilization': round(node_utilization, 4), 'clearance': round(clearance, 4)
    }

def generate_fixed_test_points(environment_type, num_pairs=10):
    grid_width, grid_height, obstacles = generate_environment_obstacles(environment_type)
    temp_generator = DeltaPRM(grid_width, grid_height, obstacles, num_nodes=1, connection_radius=1.6, max_failures=10, delta_radius=0.2)
    np.random.seed(42)
    random.seed(42)
    return temp_generator.generate_valid_point_pairs(num_pairs)

def main():
    algorithms   = ['delta', 'beam', 'spars', 'gsrm']   # 已移除 odrm
    environments = ['random', 'maze', 'indoor']
    sample_counts = [100, 200, 300, 500, 800, 1000, 1500, 2000]
    num_seeds = 5
    seeds = list(range(100, 100 + num_seeds))
    
    os.makedirs("./pursuer_strategies/PRM/results", exist_ok=True)
    output_file = f"./pursuer_strategies/PRM/results/scalability_evaluation_{datetime.now().strftime('%Y%m%d%H%M')}.csv"

    if os.path.exists(output_file): os.remove(output_file)

    total = len(algorithms) * len(environments) * len(sample_counts) * num_seeds
    print("=" * 60)
    print("🚀 开始可扩展性评测 (数据采集 + 快照留存)")
    print("=" * 60)

    fixed_points = {env: generate_fixed_test_points(env, 10) for env in environments}

    done, ok = 0, 0
    for env in environments:
        for algo in algorithms:
            for ns in sample_counts:
                for seed in seeds:
                    done += 1
                    try:
                        print(f"\r进度: [{done}/{total}] {algo} on {env} (n={ns}, seed={seed})", end="")
                        m = run_scalability_test(algo, env, ns, seed, fixed_point_pairs=fixed_points[env])
                        save_scalability_metrics_to_file(m, output_file)
                        ok += 1
                    except Exception as e:
                        pass
    print(f"\n\n✅ 评测完成！数据已保存至: {output_file}")

if __name__ == "__main__":
    main()