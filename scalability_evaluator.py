
"""
可扩展性评测脚本 - 测试不同采样次数下各算法的性能
支持多seed测试，生成包含标准差的数据

主要改动:
  - 横坐标从 num_nodes(目标节点数) 改为 num_samples(采样次数/采样预算)
  - 新增 gsrm, odrm 两种 baseline 算法
  - 保留全部原有指标（含 node_utilization）
"""

import json
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


def save_scalability_metrics_to_file(metrics, filepath):
    """保存可扩展性测试指标到CSV文件"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    file_exists = os.path.exists(filepath)

    with open(filepath, 'a', newline='') as f:
        writer = csv.writer(f)

        if not file_exists:
            # 写入表头 —— num_samples 替换原来的 num_nodes
            writer.writerow([
                'timestamp', 'algorithm', 'environment',
                'num_samples',          # ← 横坐标：采样次数（采样预算）
                'seed',
                'generation_time',
                'actual_nodes_count',
                'edges_count',
                'path_length',
                'path_nodes_count',
                'search_time',
                'path_success',
                'dispersion',
                'node_utilization',
                'clearance'
            ])

        writer.writerow([
            metrics['timestamp'],
            metrics['algorithm'],
            metrics['environment'],
            metrics['num_samples'],     # ← 对应表头
            metrics['seed'],
            metrics['generation_time'],
            metrics['actual_nodes_count'],
            metrics['edges_count'],
            metrics['path_length'],
            metrics['path_nodes_count'],
            metrics['search_time'],
            metrics['path_success'],
            metrics['dispersion'],
            metrics['node_utilization'],
            metrics['clearance']
        ])


def generate_environment_obstacles(environment_type):
    """为指定环境类型生成确定性的障碍物（固定 seed=42）

    保存/恢复外部随机状态，不干扰采样随机序列。

    返回: (grid_width, grid_height, obstacles)
    """
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
            obstacles.append((i, 0))
            obstacles.append((i, grid_height - 1))
            obstacles.append((0, i))
            obstacles.append((grid_width - 1, i))
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


def _create_generator(algorithm_name, environment_type,
                      grid_width, grid_height, obstacles, num_samples):
    """
    根据算法名和环境类型实例化 PRM Generator。

    所有算法统一接收 num_samples 作为采样预算：
      - 对于随机采样类算法（delta/beam/spars），num_samples 即最大采样尝试次数。
      - 对于 gsrm，num_samples 通过控制 Gray-Scott 模拟网格分辨率间接影响节点数；
        这里传入 target_nodes=num_samples 由算法内部自行映射。
      - 对于 odrm，num_samples 对应初始随机撒点数（SGD 优化阶段节点数不变）。
    """
    if algorithm_name == "delta":
        params = {
            "random": dict(connection_radius=1.6, max_failures=100, delta_radius=0.2),
            "maze":   dict(connection_radius=1.6, max_failures=100, delta_radius=0.3),
            "indoor": dict(connection_radius=1.4, max_failures=100, delta_radius=0.3),
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
                           beam_ray_step=0.2, min_connection_radius=0.4),
        }
        return BeamPRM(grid_width, grid_height, obstacles,
                       num_nodes=num_samples, **params[environment_type])

    elif algorithm_name == "spars":
        params = {
            "random": dict(max_failures=100, delta=0.2),
            "maze":   dict(max_failures=100, delta=0.15,
                           visibility_radius=0.8, connection_radius=0.6),
            "indoor": dict(max_failures=200, delta=0.2,
                           visibility_radius=1.4, connection_radius=1.0),
        }
        return SPARS(grid_width, grid_height, obstacles,
                     num_nodes=num_samples, **params[environment_type])

    elif algorithm_name == "gsrm":
        # GSRM：基于 Gray-Scott 反应扩散系统 + Delaunay 三角剖分
        # num_samples 作为目标节点数提示传入，由算法通过调整模拟分辨率来逼近
        # 若你的实现接口不同，请根据实际 GSRM 类的参数签名调整
        return GSRM(grid_width, grid_height, obstacles,
                    num_nodes=num_samples)

    elif algorithm_name == "odrm":
        # ODRM：随机初始化 + Delaunay 建初始图 + SGD 优化节点位置/边方向
        # num_samples 对应初始随机撒点数
        # 若你的实现接口不同，请根据实际 ODRM 类的参数签名调整
        return ODRM(grid_width, grid_height, obstacles,
                    num_nodes=num_samples)

    else:
        raise ValueError(f"Unknown algorithm: {algorithm_name}")


def run_scalability_test(algorithm_name, environment_type, num_samples, seed,
                         fixed_point_pairs=None):
    """运行单个可扩展性测试

    参数:
        algorithm_name:   算法名称
        environment_type: 环境类型
        num_samples:      采样次数/采样预算（横坐标），替换原来的 num_nodes
        seed:             随机种子（仅影响 PRM 节点采样，不影响环境障碍物）
        fixed_point_pairs: 固定起终点对；若为 None 则随机生成
    """
    print(f"\n运行测试: {algorithm_name} on {environment_type} "
          f"(samples={num_samples}, seed={seed})")

    # 1. 生成确定性障碍物（所有 seed 共享同一障碍物布局）
    grid_width, grid_height, obstacles = generate_environment_obstacles(environment_type)

    # 2. 设置采样随机种子（仅影响节点采样位置，不影响障碍物）
    np.random.seed(seed)
    random.seed(seed)

    # 3. 实例化算法
    start_time = time.time()
    generator = _create_generator(algorithm_name, environment_type,
                                  grid_width, grid_height, obstacles, num_samples)

    # 4. 生成 PRM
    if algorithm_name == "beam":
        result = generator.generate_prm()
        if len(result) >= 6:
            nodes, edges = result[0], result[1]
            # result[2..5] 为可视化辅助数据，不参与评测
        else:
            nodes, edges = result[:2]
    else:
        nodes, edges = generator.generate_prm()

    generation_time = time.time() - start_time

    # 5. 路径规划：每个 PRM 图测试多个起终点对，计算平均指标
    num_path_tests = 10
    successful_paths = []
    total_path_tests = 0

    if fixed_point_pairs is not None:
        point_pairs = fixed_point_pairs
        print(f"  使用预设的 {len(point_pairs)} 对固定起终点")
    else:
        point_pairs = generator.generate_valid_point_pairs(num_path_tests)
        print(f"  随机生成 {len(point_pairs)} 对起终点")

    for start, goal in point_pairs:
        total_path_tests += 1
        try:
            path_nodes, path_edges, path_length, search_time = \
                generator.find_path(start, goal)
            if path_nodes and len(path_nodes) > 0:
                successful_paths.append({
                    'path_length':     path_length,
                    'path_nodes_count': len(path_nodes),
                    'search_time':     search_time,
                })
        except Exception as e:
            print(f"  第 {total_path_tests} 对起终点规划失败: {e}")

    success_rate = (len(successful_paths) / num_path_tests * 100
                    if total_path_tests > 0 else 0.0)

    if successful_paths:
        path_length      = sum(p['path_length']      for p in successful_paths) / len(successful_paths)
        path_nodes_count = sum(p['path_nodes_count'] for p in successful_paths) / len(successful_paths)
        search_time      = sum(p['search_time']      for p in successful_paths) / len(successful_paths)
    else:
        path_length      = 0.0
        path_nodes_count = 0
        search_time      = 0.0

    # 6. 覆盖离散度
    dispersion = generator.cal_dispersion(num_samples=500) if generator else 0.0

    # 7. 节点利用率
    utilization_result = (generator.calculate_node_utilization(num_test_paths=100)
                          if generator else None)
    node_utilization = (utilization_result['avg_utilization']
                        if utilization_result else 0.0)
    clearance = generator.cal_clearance() if generator else 0.0
    # 8. 汇总指标
    metrics = {
        'timestamp':          datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'algorithm':          algorithm_name,
        'environment':        environment_type,
        'num_samples':        num_samples,      # ← 横坐标
        'seed':               seed,
        'generation_time':    generation_time,
        'actual_nodes_count': len(nodes),
        'edges_count':        len(edges),
        'path_length':        path_length,
        'path_nodes_count':   path_nodes_count,
        'search_time':        search_time,
        'path_success':       success_rate,
        'dispersion':         round(dispersion, 4),
        'node_utilization':   round(node_utilization, 4),
        'clearance':         round(clearance, 4)
    }

    return metrics


def generate_fixed_test_points(environment_type, num_pairs=10):
    """为指定环境生成固定起终点对（使用与算法相同的严格碰撞检测逻辑）

    固定使用 seed=42，保证每次生成结果完全一致。
    """
    grid_width, grid_height, obstacles = generate_environment_obstacles(environment_type)

    temp_generator = DeltaPRM(grid_width, grid_height, obstacles,
                               num_nodes=1, connection_radius=1.6,
                               max_failures=10, delta_radius=0.2)

    np.random.seed(42)
    random.seed(42)
    pairs = temp_generator.generate_valid_point_pairs(num_pairs)

    if len(pairs) < num_pairs:
        print(f"  警告: {environment_type} 环境只生成了 {len(pairs)} 对（目标 {num_pairs} 对）")
    else:
        print(f"  {environment_type}: 成功生成 {len(pairs)} 对固定起终点（严格碰撞检测）")

    return pairs


def main():
    """主函数 - 运行完整的可扩展性评测"""

    algorithms   = ['delta', 'beam', 'spars', 'gsrm']
    environments = ['random', 'maze', 'indoor']

    # 横坐标：采样次数（替换原来的 node_counts）
    sample_counts = [100, 200, 300, 500, 800, 1000, 1500, 2000]

    num_seeds = 5
    seeds = list(range(100, 100 + num_seeds))
    import datetime
    output_file = f"./pursuer_strategies/PRM/results/scalability_evaluation_{datetime.datetime.now().strftime('%Y%m%d%H%M')}.csv"

    if os.path.exists(output_file):
        os.remove(output_file)
        print(f"已删除旧文件: {output_file}\n")

    total = len(algorithms) * len(environments) * len(sample_counts) * num_seeds
    print("=" * 60)
    print("开始可扩展性评测（横坐标：采样次数）")
    print("=" * 60)
    print(f"算法:          {algorithms}")
    print(f"环境:          {environments}")
    print(f"采样次数范围:  {sample_counts}")
    print(f"每配置 seed 数: {num_seeds}")
    print(f"总测试数:      {total}")

    # 预生成固定起终点
    print("\n生成固定的测试起终点...")
    print("=" * 60)
    fixed_points = {}
    for env in environments:
        fixed_points[env] = generate_fixed_test_points(env, num_pairs=10)
        print(f"  {env.capitalize()}: {len(fixed_points[env])} 对")
    print("=" * 60)

    done, ok = 0, 0
    for env in environments:
        for algo in algorithms:
            for ns in sample_counts:
                for seed in seeds:
                    done += 1
                    try:
                        print(f"\n进度: {done}/{total}")
                        m = run_scalability_test(algo, env, ns, seed,
                                                 fixed_point_pairs=fixed_points[env])
                        save_scalability_metrics_to_file(m, output_file)
                        ok += 1
                        print(f"✓ 成功: {algo} on {env} (samples={ns}, seed={seed})")
                        print(f"  - 生成时间:    {m['generation_time']:.3f}s")
                        print(f"  - 实际节点数:  {m['actual_nodes_count']}")
                        print(f"  - 边数:        {m['edges_count']}")
                        print(f"  - 路径成功率:  {m['path_success']:.1f}%")
                        print(f"  - 离散度:      {m['dispersion']:.4f}")
                        print(f"  - 节点利用率:  {m['node_utilization']*100:.1f}%")
                        print(f"  - 平均净空:    {m['clearance']:.3f}")
                        if m['path_success'] > 0:
                            print(f"  - 路径长度:    {m['path_length']:.3f}")
                            print(f"  - 搜索时间:    {m['search_time']:.4f}s")
                    except Exception as e:
                        print(f"✗ 失败: {algo} on {env} (samples={ns}, seed={seed})")
                        print(f"  错误: {e}")
                        import traceback
                        traceback.print_exc()

    print("\n" + "=" * 60)
    print("评测完成！")
    print("=" * 60)
    print(f"结果文件: {output_file}")
    print(f"成功/总计: {ok}/{done}")
    print("=" * 60)


if __name__ == "__main__":
    main()
