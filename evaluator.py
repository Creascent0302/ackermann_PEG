import json
import csv
import os
from datetime import datetime
import time
import numpy as np
import pygame
import sys
sys.path.append('.')
from config import ENV_CONFIG
from generator import *
from map_generator import generate_maze_obstacles, generate_indoor_obstacles

def save_metrics_to_file(metrics, filepath):
    """保存指标到文件"""
    # 确保目录存在
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'a', newline='') as f:
        writer = csv.writer(f)

        # 写入数据
        writer.writerow([
            metrics['timestamp'],
            metrics['algorithm'],
            metrics['environment'],
            metrics['seed'],
            metrics['generation_time'],
            metrics['nodes_count'],
            metrics['edges_count'],
            metrics['dispersion'],
            metrics['discrepancy'],
        ])

def run_algorithm_test(algorithm_name, environment_type, seed, save_images=True):
    """运行单个算法测试"""
    np.random.seed(seed)
    import random
    random.seed(seed)
    
    # 设置基础路径 - 修复路径
    base_results_path = "./pursuer_strategies/PRM/results"
    # 每种环境和算法的采样节点数
    sample_nodes_map = {
        'maze': {'delta': 2000, 'beam': 500, 'spars': 3000},
        'indoor': {'delta': 2000, 'beam': 500, 'spars': 3000},
        'random': {'delta': 2000, 'beam': 500, 'spars': 2000}
    }
    # 设置环境参数
    if environment_type == "maze":
        ENV_CONFIG['gridnum_width'] = 49
        ENV_CONFIG['gridnum_height'] = 49
        grid_width = ENV_CONFIG['gridnum_width']
        grid_height = ENV_CONFIG['gridnum_height']
        obstacles = generate_maze_obstacles(grid_width, grid_height)
    elif environment_type == "indoor":
        ENV_CONFIG['gridnum_width'] = 51
        ENV_CONFIG['gridnum_height'] = 51
        grid_width = ENV_CONFIG['gridnum_width']
        grid_height = ENV_CONFIG['gridnum_height']
        obstacles = generate_indoor_obstacles(grid_width, grid_height)
    elif environment_type == "random":
        ENV_CONFIG['gridnum_width'] = 40
        ENV_CONFIG['gridnum_height'] = 40
        grid_width = ENV_CONFIG['gridnum_width']
        grid_height = ENV_CONFIG['gridnum_height']
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
    
    print(f"\n运行测试: {algorithm_name} on {environment_type} (seed={seed})")
    num_nodes = sample_nodes_map.get(environment_type, {}).get(algorithm_name, 300)
    # 记录开始时间
    start_time = time.time()
    generator = None
    if algorithm_name == "delta":
        if environment_type == "random":
            generator = DeltaPRM(grid_width, grid_height, obstacles, num_nodes=num_nodes, connection_radius=1.6, max_failures=100, delta_radius=0.2)
        elif environment_type == "maze":
            generator = DeltaPRM(grid_width, grid_height, obstacles, num_nodes=num_nodes, connection_radius=1.6, max_failures=100, delta_radius=0.3)
        elif environment_type == "indoor":
            generator = DeltaPRM(grid_width, grid_height, obstacles, num_nodes=num_nodes, connection_radius=1.4, max_failures=100, delta_radius=0.3)
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
            generator = SPARS(grid_width, grid_height, obstacles, num_nodes=num_nodes, max_failures=100, delta=0.2)
        elif environment_type == "maze":
            generator = SPARS(grid_width, grid_height, obstacles, num_nodes=num_nodes, max_failures=100, delta=0.15, visibility_radius=0.8, connection_radius=0.6)
        elif environment_type == "indoor":
            generator = SPARS(grid_width, grid_height, obstacles, num_nodes=num_nodes, max_failures=200, delta=0.2, visibility_radius=1.4, connection_radius=1.0)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm_name}")
    
    if algorithm_name == "beam":
        result = generator.generate_prm()
        if len(result) >= 6:
            nodes, edges, medial_axis_nodes, medial_axis_all_nodes, medial_axis_edges, medial_axis_paths = result
        else:
            nodes, edges = result[:2]
            medial_axis_nodes, medial_axis_edges, medial_axis_paths = set(), set(), []
            medial_axis_all_nodes = set()
    else:
        nodes, edges = generator.generate_prm()
        medial_axis_nodes, medial_axis_edges, medial_axis_paths = set(), set(), []
        medial_axis_all_nodes = set()

    # 记录结束时间
    end_time = time.time()
    generation_time = end_time - start_time
    dispersion = generator.cal_dispersion() if generator else 0.0
    discrepancy = generator.cal_discrepancy() if generator else 0.0
    
    # 计算节点利用率
    print("  计算节点利用率...")
    utilization_result = generator.calculate_node_utilization(num_test_paths=20) if generator else None
    if utilization_result:
        node_utilization = utilization_result['avg_utilization']
        avg_nodes_per_path = utilization_result['avg_nodes_per_path']
    else:
        node_utilization = 0.0
        avg_nodes_per_path = 0.0

    # 计算beam算法中轴图的指标
    if algorithm_name == "beam" and medial_axis_all_nodes and medial_axis_edges:
        # 需实现或已有相关方法
        medial_dispersion = generator.cal_dispersion(media=True)
        medial_discrepancy = generator.cal_discrepancy(media=True)
        medial_axis_nodes_count = len(medial_axis_all_nodes)
        medial_axis_edges_count = len(medial_axis_edges)
    else:
        medial_dispersion = 0.0
        medial_discrepancy = 0.0
        medial_axis_nodes_count = 0
        medial_axis_edges_count = 0

    # 准备指标
    metrics = {
        'timestamp': datetime.now().isoformat(),
        'algorithm': algorithm_name,
        'environment': environment_type,
        'seed': seed if environment_type != 'indoor' else 'N/A',  # indoor不使用随机种子
        'generation_time': round(generation_time, 4),
        'nodes_count': len(nodes),
        'edges_count': len(edges),
        'dispersion': round(dispersion, 4),
        'discrepancy': round(discrepancy, 4),
        'node_utilization': round(node_utilization, 4),
        'avg_nodes_per_path': round(avg_nodes_per_path, 2),
        'medial_axis_nodes_count': medial_axis_nodes_count,
        'medial_axis_edges_count': medial_axis_edges_count,
        'medial_axis_dispersion': round(medial_dispersion, 4),
        'medial_axis_discrepancy': round(medial_discrepancy, 4),
    }
    
    print(f"完成: {len(nodes)} 节点, {len(edges)} 边, "
          f" 时间: {generation_time:.2f}s")
    
    # 保存图像 - 使用generator.py中的PRMRenderer
    if save_images:
        # 创建保存路径
        env_dir = os.path.join(base_results_path, environment_type)
        os.makedirs(env_dir, exist_ok=True)
        
        # 生成文件名
        # 在run_algorithm_test函数中修改文件名生成逻辑
        if environment_type == 'indoor':
            filename = f"{algorithm_name}_{environment_type}.pdf"
        else:
            filename = f"{algorithm_name}_{environment_type}_{seed}.pdf"        
        filepath = os.path.join(env_dir, filename)
        
        # 使用generator.py中的PRMRenderer保存图像
        renderer = PRMRenderer(grid_width, grid_height, headless=True)
        if algorithm_name == "beam":
            renderer.save_image(nodes, edges, obstacles, filepath, 
                              medial_axis_all_nodes, medial_axis_edges, medial_axis_paths, env=environment_type, algorithm=algorithm_name)
        else:
            renderer.save_image(nodes, edges, obstacles, filepath, env=environment_type, algorithm=algorithm_name)
        pygame.quit()
    
    return metrics

def run_full_evaluation():
    """运行完整的评测流程"""
    algorithms = ['delta', 'beam', 'spars', 'gsrm', 'odrm']
    seeds = [43, 114, 520]  # 三个固定的随机种子
    # 定义测试配置: (环境, 是否使用随机种子)
    test_configs = [
        ('maze', True),      # 迷宫使用随机种子
        ('indoor', False),   # 室内不使用随机种子
        ('random', True)     # 随机使用随机种子
    ]
    
    all_metrics = []
    
    # 计算总测试数
    total_tests = 0
    for env, use_seeds in test_configs:
        if use_seeds:
            total_tests += len(algorithms) * len(seeds)
        else:
            total_tests += len(algorithms)
    
    current_test = 0
    base_results_path = "./pursuer_strategies/PRM/results"

    print(f"开始运行 {total_tests} 个测试...")
    print(f"结果将保存到: {base_results_path}")

    # 创建汇总CSV文件
    summary_file = os.path.join(base_results_path, "evaluation_summary.csv")
    os.makedirs(os.path.dirname(summary_file), exist_ok=True)
    with open(summary_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'algorithm', 'environment', 'seed',
            'generation_time', 'nodes_count', 'edges_count',
            'dispersion', 'discrepancy', 'node_utilization', 'avg_nodes_per_path',
            'medial_axis_nodes_count', 'medial_axis_edges_count',
            'medial_axis_dispersion', 'medial_axis_discrepancy'
        ])
    
    for env, use_seeds in test_configs:
        print(f"\n{'='*50}")
        print(f"测试环境: {env} (使用随机种子: {use_seeds})")
        print(f"{'='*50}")
        
        for alg in algorithms:
            if use_seeds:
                # 使用多个种子测试
                for seed in seeds:
                    current_test += 1
                    print(f"\n进度: {current_test}/{total_tests}")
                    
                    try:
                        metrics = run_algorithm_test(alg, env, seed, save_images=True)
                        if metrics:
                            all_metrics.append(metrics)
                            
                            # 保存到汇总CSV
                            with open(summary_file, 'a', newline='') as f:
                                writer = csv.writer(f)
                                writer.writerow([
                                    metrics['algorithm'],
                                    metrics['environment'],
                                    metrics['seed'],
                                    metrics['generation_time'],
                                    metrics['nodes_count'],
                                    metrics['edges_count'],
                                    metrics['dispersion'],
                                    metrics['discrepancy'],
                                    metrics['node_utilization'],
                                    metrics['avg_nodes_per_path'],
                                    metrics['medial_axis_nodes_count'],
                                    metrics['medial_axis_edges_count'],
                                    metrics['medial_axis_dispersion'],
                                    metrics['medial_axis_discrepancy']
                                ])
                        
                    except Exception as e:
                        print(f"错误: {alg} on {env} (seed={seed}): {str(e)}")
                        import traceback
                        traceback.print_exc()
            else:
                # 不使用随机种子，只测试一次
                current_test += 1
                print(f"\n进度: {current_test}/{total_tests}")
                
                try:
                    # 对于indoor环境，使用固定的种子42但不在文件名中体现
                    metrics = run_algorithm_test(alg, env, 42, save_images=True)
                    if metrics:
                        all_metrics.append(metrics)
                        
                        # 保存到汇总CSV
                        with open(summary_file, 'a', newline='') as f:
                            writer = csv.writer(f)
                            writer.writerow([
                                metrics['algorithm'],
                                metrics['environment'],
                                metrics['seed'],
                                metrics['generation_time'],
                                metrics['nodes_count'],
                                metrics['edges_count'],
                                metrics['dispersion'],
                                metrics['discrepancy'],
                                metrics['medial_axis_nodes_count'],
                                metrics['medial_axis_edges_count'],
                                metrics['medial_axis_dispersion'],
                                metrics['medial_axis_discrepancy']
                            ])
                    
                except Exception as e:
                    print(f"错误: {alg} on {env}: {str(e)}")
                    import traceback
                    traceback.print_exc()
    
    # 保存汇总结果到JSON
    json_summary_file = os.path.join(base_results_path, "evaluation_summary.json")
    with open(json_summary_file, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    
    print(f"\n{'='*60}")
    print("评测完成！")
    print(f"{'='*60}")
    print(f"结果保存在: {base_results_path}")
    print(f"图像文件: {base_results_path}/{{environment}}/{{algorithm}}_{{environment}}_{{seed}}.pdf")
    print(f"汇总文件: {summary_file} 和 {json_summary_file}")
    
    # 简单统计
    print(f"\n总计测试: {len(all_metrics)} 个")
    successful_tests = [m for m in all_metrics if m.get('nodes_count', 0) > 0]
    print(f"成功测试: {len(successful_tests)} 个")
        
    return all_metrics

if __name__ == "__main__":
    run_full_evaluation()
