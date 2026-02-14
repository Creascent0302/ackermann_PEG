"""
可扩展性评测脚本 - 测试不同顶点数下各算法的性能
支持多seed测试，生成包含标准差的数据
"""

import json
import csv
import os
from datetime import datetime
import time
import random
import numpy as np
import pygame
import sys
sys.path.append('.')
from config import ENV_CONFIG
from generator import *
from map_generator import generate_maze_obstacles, generate_indoor_obstacles

def save_scalability_metrics_to_file(metrics, filepath):
    """保存可扩展性测试指标到CSV文件"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # 检查文件是否存在，如果不存在则写入表头
    file_exists = os.path.exists(filepath)
    
    with open(filepath, 'a', newline='') as f:
        writer = csv.writer(f)
        
        if not file_exists:
            # 写入表头
            writer.writerow([
                'timestamp', 'algorithm', 'environment', 'num_nodes', 'seed',
                'generation_time', 'actual_nodes_count', 'edges_count',
                'path_length', 'path_nodes_count', 'search_time', 'path_success'
            ])
        
        # 写入数据
        writer.writerow([
            metrics['timestamp'],
            metrics['algorithm'],
            metrics['environment'],
            metrics['num_nodes'],
            metrics['seed'],
            metrics['generation_time'],
            metrics['actual_nodes_count'],
            metrics['edges_count'],
            metrics['path_length'],
            metrics['path_nodes_count'],
            metrics['search_time'],
            metrics['path_success']
        ])

def generate_environment_obstacles(environment_type):
    """为指定环境类型生成确定性的障碍物
    
    所有环境类型都使用固定seed=42生成障碍物，确保：
    - maze: random.shuffle产生相同的迷宫布局
    - random: 随机障碍物位置固定
    - indoor: 本身是确定性的，但为统一性也使用固定seed
    
    返回:
        (grid_width, grid_height, obstacles)
    """
    ENV_OBSTACLE_SEED = 42
    
    # 保存当前随机状态
    np_state = np.random.get_state()
    py_state = random.getstate()
    # 使用固定seed，确保障碍物生成完全确定
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
        # 添加边界
        for i in range(grid_width):
            obstacles.append((i, 0))
            obstacles.append((i, grid_height - 1))
            obstacles.append((0, i))
            obstacles.append((grid_width - 1, i))
        # 添加随机障碍物
        while len(obstacles) < num_obstacles:
            x = np.random.randint(1, grid_width - 1)
            y = np.random.randint(1, grid_height - 1)
            if (x, y) not in obstacles:
                obstacles.append((x, y))
    else:
        raise ValueError(f"Unknown environment type: {environment_type}")
    
    # 恢复随机状态（不干扰外部的随机序列）
    np.random.set_state(np_state)
    random.setstate(py_state)
    
    # 更新全局配置
    ENV_CONFIG['gridnum_width'] = grid_width
    ENV_CONFIG['gridnum_height'] = grid_height
    
    return grid_width, grid_height, obstacles


def run_scalability_test(algorithm_name, environment_type, num_nodes, seed, fixed_point_pairs=None):
    """运行单个可扩展性测试
    
    参数:
        algorithm_name: 算法名称
        environment_type: 环境类型
        num_nodes: 节点数
        seed: 随机种子（仅影响PRM节点采样，不影响环境障碍物）
        fixed_point_pairs: 固定的起终点对列表，如果提供则使用这些点而不是随机生成
    """
    print(f"\n运行测试: {algorithm_name} on {environment_type} (nodes={num_nodes}, seed={seed})")
    
    # 生成确定性的环境障碍物（所有seed使用相同障碍物布局）
    grid_width, grid_height, obstacles = generate_environment_obstacles(environment_type)
    
    # 设置PRM生成的随机种子（仅影响节点采样，不影响障碍物）
    np.random.seed(seed)
    random.seed(seed)
    
    # 记录开始时间
    start_time = time.time()
    generator = None
    
    # 创建算法实例
    if algorithm_name == "delta":
        if environment_type == "random":
            generator = DeltaPRM(grid_width, grid_height, obstacles, num_nodes=num_nodes, 
                               connection_radius=1.6, max_failures=100, delta_radius=0.2)
        elif environment_type == "maze":
            generator = DeltaPRM(grid_width, grid_height, obstacles, num_nodes=num_nodes, 
                               connection_radius=1.6, max_failures=100, delta_radius=0.3)
        elif environment_type == "indoor":
            generator = DeltaPRM(grid_width, grid_height, obstacles, num_nodes=num_nodes, 
                               connection_radius=1.4, max_failures=100, delta_radius=0.3)
    
    elif algorithm_name == "beam":
        if environment_type == "random":
            generator = BeamPRM(grid_width, grid_height, obstacles,
                               num_nodes=num_nodes, connection_radius=1.2,
                               beam_angle_step_deg=2.6, beam_ray_step=0.08,
                               min_connection_radius=0.3)
        elif environment_type == "maze":
            generator = BeamPRM(grid_width, grid_height, obstacles,
                               num_nodes=num_nodes, connection_radius=1.2,
                               beam_angle_step_deg=30, beam_ray_step=0.25,
                               min_connection_radius=0.3)
        elif environment_type == "indoor":
            generator = BeamPRM(grid_width, grid_height, obstacles,
                               num_nodes=num_nodes, connection_radius=2.0,
                               beam_angle_step_deg=25, beam_ray_step=0.2,
                               min_connection_radius=0.4)
    
    elif algorithm_name == "spars":
        if environment_type == "random":
            generator = SPARS(grid_width, grid_height, obstacles, num_nodes=num_nodes, 
                            max_failures=100, delta=0.2)
        elif environment_type == "maze":
            generator = SPARS(grid_width, grid_height, obstacles, num_nodes=num_nodes, 
                            max_failures=100, delta=0.15, visibility_radius=0.8, connection_radius=0.6)
        elif environment_type == "indoor":
            generator = SPARS(grid_width, grid_height, obstacles, num_nodes=num_nodes, 
                            max_failures=200, delta=0.2, visibility_radius=1.4, connection_radius=1.0)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm_name}")
    
    # 生成PRM
    if algorithm_name == "beam":
        result = generator.generate_prm()
        if len(result) >= 6:
            nodes, edges, medial_axis_nodes, medial_axis_all_nodes, medial_axis_edges, medial_axis_paths = result
        else:
            nodes, edges = result[:2]
    else:
        nodes, edges = generator.generate_prm()
    
    # 记录生成时间
    end_time = time.time()
    generation_time = end_time - start_time
    
    # 测试路径规划 - 每个PRM图测试多个起终点对
    num_path_tests = 10  # 每个PRM图测试10个不同的路径
    successful_paths = []
    total_path_tests = 0
    
    # 如果提供了固定的起终点对，使用这些点；否则随机生成
    if fixed_point_pairs is not None:
        point_pairs = fixed_point_pairs
        print(f"  使用预设的{len(point_pairs)}对固定起终点")
    else:
        # 生成多个有效的起终点对（旧方法，保持向后兼容）
        point_pairs = generator.generate_valid_point_pairs(num_path_tests)
        print(f"  随机生成{len(point_pairs)}对起终点")
    
    for start, goal in point_pairs:
        total_path_tests += 1
        try:
            # 路径规划 - find_path返回 (path_nodes, path_edges, path_length, search_time)
            path_nodes, path_edges, path_length, search_time = generator.find_path(start, goal)
            
            if path_nodes and len(path_nodes) > 0:
                successful_paths.append({
                    'path_length': path_length,
                    'path_nodes_count': len(path_nodes),
                    'search_time': search_time
                })
        except Exception as e:
            print(f"  第{total_path_tests}对起终点规划失败: {e}")
    
    # 计算平均值
    success_rate = len(successful_paths) / num_path_tests * 100 if total_path_tests > 0 else 0.0
    
    if successful_paths:
        path_length = sum(p['path_length'] for p in successful_paths) / len(successful_paths)
        path_nodes_count = sum(p['path_nodes_count'] for p in successful_paths) / len(successful_paths)
        search_time = sum(p['search_time'] for p in successful_paths) / len(successful_paths)
    else:
        path_length = 0.0
        path_nodes_count = 0
        search_time = 0.0
    
    # 收集指标
    metrics = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'algorithm': algorithm_name,
        'environment': environment_type,
        'num_nodes': num_nodes,
        'seed': seed,
        'generation_time': generation_time,
        'actual_nodes_count': len(nodes),
        'edges_count': len(edges),
        'path_length': path_length,
        'path_nodes_count': path_nodes_count,
        'search_time': search_time,
        'path_success': success_rate  # 现在是0-100的百分比
    }
    
    return metrics

def generate_fixed_test_points(environment_type, num_pairs=10):
    """为每个环境生成固定的起终点对
    
    使用与算法完全相同的碰撞检测逻辑（_is_valid_position多点采样），
    确保生成的点在测试时不会因为碰撞检测不一致而失败。
    使用与run_scalability_test相同的generate_environment_obstacles，
    确保障碍物完全一致。
    
    参数:
        environment_type: 环境类型 ('maze', 'indoor', 'random')
        num_pairs: 生成的起终点对数量
    
    返回:
        List[Tuple]: 起终点对列表 [(start1, goal1), (start2, goal2), ...]
    """
    # 生成确定性障碍物（与测试时完全一致）
    grid_width, grid_height, obstacles = generate_environment_obstacles(environment_type)
    
    # 创建临时算法实例，用于严格的碰撞检测（_is_valid_position多点采样）
    temp_generator = DeltaPRM(grid_width, grid_height, obstacles,
                               num_nodes=1, connection_radius=1.6,
                               max_failures=10, delta_radius=0.2)
    
    # 使用固定种子确保每次生成相同的起终点
    np.random.seed(42)
    random.seed(42)
    
    # 使用算法内置的generate_valid_point_pairs（使用_is_valid_position严格检测）
    pairs = temp_generator.generate_valid_point_pairs(num_pairs)
    
    if len(pairs) < num_pairs:
        print(f"  警告: {environment_type}环境只生成了{len(pairs)}对点（目标{num_pairs}对）")
    else:
        print(f"  {environment_type}: 成功生成{len(pairs)}对固定起终点（使用严格碰撞检测）")
    
    return pairs

def main():
    """主函数 - 运行完整的可扩展性评测"""
    
    # 配置参数
    algorithms = ['delta', 'beam', 'spars']
    environments = ['random', 'maze', 'indoor']
    
    # 不同顶点数的配置
    node_counts = [100, 200, 300, 500, 800, 1000, 1500, 2000]
    
    # 每个配置测试的seed数量
    num_seeds = 5
    seeds = list(range(100, 100 + num_seeds))
    
    # 输出路径
    output_file = "./pursuer_strategies/PRM/results/scalability_evaluation.csv"
    
    # 删除旧的CSV文件（如果存在）
    if os.path.exists(output_file):
        os.remove(output_file)
        print(f"已删除旧文件: {output_file}\n")
    
    print("="*60)
    print("开始可扩展性评测")
    print("="*60)
    print(f"算法: {algorithms}")
    print(f"环境: {environments}")
    print(f"顶点数范围: {node_counts}")
    print(f"每个配置测试 {num_seeds} 个不同的seed")
    print(f"总测试数: {len(algorithms) * len(environments) * len(node_counts) * num_seeds}")
    
    # 为每个环境生成固定的起终点对
    print("\n生成固定的测试起终点...")
    print("="*60)
    fixed_points = {}
    for env in environments:
        points = generate_fixed_test_points(env, num_pairs=10)
        fixed_points[env] = points
        print(f"{env.capitalize()}: 生成了{len(points)}对固定起终点")
    print("="*60)
    
    total_tests = 0
    successful_tests = 0
    
    # 遍历所有组合
    for env in environments:
        for algo in algorithms:
            for num_nodes in node_counts:
                for seed in seeds:
                    total_tests += 1
                    try:
                        print(f"\n进度: {total_tests}/{len(algorithms) * len(environments) * len(node_counts) * num_seeds}")
                        # 使用该环境的固定起终点
                        metrics = run_scalability_test(algo, env, num_nodes, seed, 
                                                      fixed_point_pairs=fixed_points[env])
                        save_scalability_metrics_to_file(metrics, output_file)
                        successful_tests += 1
                        print(f"✓ 成功: {algo} on {env} (nodes={num_nodes}, seed={seed})")
                        print(f"  - 生成时间: {metrics['generation_time']:.3f}s")
                        print(f"  - 实际节点数: {metrics['actual_nodes_count']}")
                        print(f"  - 边数: {metrics['edges_count']}")
                        print(f"  - 路径成功: {metrics['path_success']}")
                        if metrics['path_success']:
                            print(f"  - 路径长度: {metrics['path_length']:.3f}")
                            print(f"  - 搜索时间: {metrics['search_time']:.4f}s")
                    
                    except Exception as e:
                        print(f"✗ 失败: {algo} on {env} (nodes={num_nodes}, seed={seed})")
                        print(f"  错误: {e}")
                        import traceback
                        traceback.print_exc()
    
    print("\n" + "="*60)
    print("评测完成！")
    print("="*60)
    print(f"结果保存在: {output_file}")
    print(f"总计测试: {total_tests} 个")
    print(f"成功测试: {successful_tests} 个")
    print("="*60)

if __name__ == "__main__":
    main()
