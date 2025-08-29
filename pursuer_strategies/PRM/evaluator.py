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
            metrics['connection_radius']
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
        'maze': {'classical': 600, 'star': 600, 'beam': 300, 'spars': 1500},
        'indoor': {'classical': 500, 'star': 300, 'beam': 300, 'spars': 1800},
        'random': {'classical': 400, 'star': 400, 'beam': 300, 'spars': 1000}
    }
    # 设置环境参数
    if environment_type == "maze":
        ENV_CONFIG['gridnum_width'] = 49
        ENV_CONFIG['gridnum_height'] = 49
        grid_width = ENV_CONFIG['gridnum_width']
        grid_height = ENV_CONFIG['gridnum_height']
        obstacles = generate_maze_obstacles(grid_width, grid_height)
        connection_radius = 2.0
    elif environment_type == "indoor":
        ENV_CONFIG['gridnum_width'] = 50
        ENV_CONFIG['gridnum_height'] = 50
        grid_width = ENV_CONFIG['gridnum_width']
        grid_height = ENV_CONFIG['gridnum_height']
        obstacles = generate_indoor_obstacles(grid_width, grid_height)
        connection_radius = 1.5
    elif environment_type == "random":
        ENV_CONFIG['gridnum_width'] = 30
        ENV_CONFIG['gridnum_height'] = 30
        grid_width = ENV_CONFIG['gridnum_width']
        grid_height = ENV_CONFIG['gridnum_height']
        total_cells = grid_width * grid_height
        num_obstacles = int(total_cells * 0.25)
        obstacles = []
        while len(obstacles) < num_obstacles:
            x = np.random.randint(0, grid_width)
            y = np.random.randint(0, grid_height)
            if (x, y) not in obstacles:
                obstacles.append((x, y))
        connection_radius = 1.0
    else:
        raise ValueError(f"Unknown environment type: {environment_type}")
    
    print(f"\n运行测试: {algorithm_name} on {environment_type} (seed={seed})")
    num_nodes = sample_nodes_map.get(environment_type, {}).get(algorithm_name, 300)
    # 记录开始时间
    start_time = time.time()
    generator = None
    # 运行算法
    try:
        if algorithm_name == "classical":
            generator = ClassicalPRM(grid_width, grid_height, obstacles, num_nodes=num_nodes, connection_radius=connection_radius)
            nodes, edges = generator.generate_prm()
            medial_axis_nodes, medial_axis_edges, medial_axis_paths = set(), set(), []
        
        elif algorithm_name == "star":
            generator = PRMStar(grid_width, grid_height, obstacles, 
                               num_nodes=num_nodes, gamma_prm_star=15.0)
            nodes, edges = generator.generate_prm()
            medial_axis_nodes, medial_axis_edges, medial_axis_paths = set(), set(), []
        
        elif algorithm_name == "beam":
            if environment_type == "random":
                generator = BeamPRM(grid_width, grid_height, obstacles,
                                       num_nodes=num_nodes, connection_radius=1.2,
                                       beam_angle_step_deg=3, beam_ray_step=0.08,
                                       min_connection_radius=0.3)
            elif environment_type == "maze":
                generator = BeamPRM(grid_width, grid_height, obstacles,
                                       num_nodes=num_nodes, connection_radius=1.5,
                                       beam_angle_step_deg=25, beam_ray_step=0.2,
                                       min_connection_radius=0.4)
            elif environment_type == "indoor":
                generator = BeamPRM(grid_width, grid_height, obstacles,
                                       num_nodes=num_nodes, connection_radius=2,
                                       beam_angle_step_deg=25, beam_ray_step=0.2,
                                       min_connection_radius=0.4)
            
            result = generator.generate_prm()
            if len(result) >= 6:
                nodes, edges, medial_axis_nodes, medial_axis_all_nodes, medial_axis_edges, medial_axis_paths = result
            else:
                nodes, edges = result[:2]
                medial_axis_nodes, medial_axis_edges, medial_axis_paths = set(), set(), []
        
        elif algorithm_name == "spars":
            generator = SPARS(grid_width, grid_height, obstacles,
                             num_nodes=num_nodes)
            result = generator.generate_prm()
            if len(result) >= 2:
                nodes, edges = result[:2]
            else:
                nodes, edges = [], []
            medial_axis_nodes, medial_axis_edges, medial_axis_paths = set(), set(), []
        
        else:
            raise ValueError(f"Unknown algorithm: {algorithm_name}")
    
    except Exception as e:
        print(f"算法执行失败: {e}")
        return None
    
    # 记录结束时间
    end_time = time.time()
    generation_time = end_time - start_time
    dispersion = generator.cal_dispersion() if generator else 0.0
    discrepancy = generator.cal_discrepancy() if generator else 0.0
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
        'connection_radius': connection_radius
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
                              medial_axis_nodes, medial_axis_edges, medial_axis_paths)
        else:
            renderer.save_image(nodes, edges, obstacles, filepath)
        pygame.quit()
    
    return metrics

def run_full_evaluation():
    """运行完整的评测流程"""
    algorithms = ['classical', 'star', 'beam', 'spars']
    seeds = [42, 123, 456]  # 三个固定的随机种子
    # num_nodes = [100, 200, 300, 500]
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

    # 清空所有CSV文件
    for env, _ in test_configs:
        metrics_file = os.path.join(base_results_path, f"metrics_{env}.csv")
        # 创建文件夹如果不存在
        os.makedirs(os.path.dirname(metrics_file), exist_ok=True)
        # 写入表头
        with open(metrics_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'algorithm', 'environment', 'seed',
                'generation_time', 'nodes_count', 'edges_count',
                'dispersion', 'discrepancy', 'connection_radius'
            ])
    
    # for num_node in num_nodes:
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
                            
                            # 保存单个结果
                            metrics_file = os.path.join(base_results_path, f"metrics_{env}.csv")
                            save_metrics_to_file(metrics, metrics_file)
                        
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
                        
                        # 保存单个结果
                        metrics_file = os.path.join(base_results_path, f"metrics_{env}.csv")
                        save_metrics_to_file(metrics, metrics_file)
                    
                except Exception as e:
                    print(f"错误: {alg} on {env}: {str(e)}")
                    import traceback
                    traceback.print_exc()
    
    # 保存汇总结果
    summary_file = os.path.join(base_results_path, "evaluation_summary.json")
    os.makedirs(base_results_path, exist_ok=True)
    with open(summary_file, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    
    print(f"\n{'='*60}")
    print("评测完成！")
    print(f"{'='*60}")
    print(f"结果保存在: {base_results_path}")
    print(f"图像文件: {base_results_path}/{{environment}}/{{algorithm}}_{{environment}}_{{seed}}.pdf")
    print(f"指标文件: {base_results_path}/metrics_{{environment}}.csv")
    print(f"汇总文件: {summary_file}")
    
    # 简单统计
    print(f"\n总计测试: {len(all_metrics)} 个")
    successful_tests = [m for m in all_metrics if m.get('nodes_count', 0) > 0]
    print(f"成功测试: {len(successful_tests)} 个")
        
    return all_metrics

if __name__ == "__main__":
    run_full_evaluation()
