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

def calculate_beam_connectivity_score(nodes, edges, obstacles, grid_width, grid_height):
    """
    专门为Beam算法设计的连通性评分
    重点评估: 拓扑完整性、节点精简性、覆盖度
    """
    if not nodes or not edges:
        return 0.0
    
    # 1. 连通分量分析 (40%权重)
    adj = {i: [] for i in range(len(nodes))}
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    
    for edge in edges:
        if edge[0] in node_to_idx and edge[1] in node_to_idx:
            i, j = node_to_idx[edge[0]], node_to_idx[edge[1]]
            adj[i].append(j)
            adj[j].append(i)
    
    # 找连通分量
    visited = [False] * len(nodes)
    components = []
    
    def bfs(start):
        queue = [start]
        visited[start] = True
        component = [start]
        while queue:
            node = queue.pop(0)
            for neighbor in adj[node]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append(neighbor)
                    component.append(neighbor)
        return component
    
    for i in range(len(nodes)):
        if not visited[i]:
            components.append(bfs(i))
    
    # 连通性评分: 严重惩罚多个连通分量
    if len(components) == 0:
        connectivity_score = 0.0
    elif len(components) == 1:
        connectivity_score = 1.0  # 完全连通
    else:
        # 有离群节点，按最大连通分量比例评分，但有惩罚
        largest_comp_size = max(len(comp) for comp in components)
        connectivity_score = (largest_comp_size / len(nodes)) * 0.7  # 最高只能得70分
    
    # 2. 节点效率评分 (25%权重)
    total_cells = grid_width * grid_height
    obstacle_cells = len(obstacles)
    reachable_area = total_cells - obstacle_cells
    
    node_density = len(nodes) / max(reachable_area, 1)
    if node_density <= 0.05:  # 每20个cell一个节点，很好
        density_score = 1.0
    elif node_density <= 0.1:  # 每10个cell一个节点，还行
        density_score = 0.8
    elif node_density <= 0.2:  # 每5个cell一个节点，偏密
        density_score = 0.6
    else:  # 太密集了
        density_score = max(0.2, 1.0 - (node_density - 0.2) * 2)
    
    # 3. 拓扑复杂度评分 (20%权重)
    total_degree = sum(len(adj[i]) for i in range(len(nodes)))
    avg_degree = total_degree / len(nodes) if len(nodes) > 0 else 0
    
    if 2.5 <= avg_degree <= 4.0:
        topology_score = 1.0
    elif 2.0 <= avg_degree < 2.5:
        topology_score = 0.8
    elif 4.0 < avg_degree <= 6.0:
        topology_score = 0.7
    else:
        topology_score = max(0.3, 1.0 - abs(avg_degree - 3.25) * 0.2)
    
    # 4. 覆盖均匀性评分 (15%权重)
    if len(nodes) < 3:
        coverage_score = 0.5
    else:
        distances = []
        for i, node1 in enumerate(nodes):
            min_dist = float('inf')
            for j, node2 in enumerate(nodes):
                if i != j:
                    dist = ((node1[0] - node2[0])**2 + (node1[1] - node2[1])**2)**0.5
                    min_dist = min(min_dist, dist)
            distances.append(min_dist)
        
        if distances:
            mean_dist = sum(distances) / len(distances)
            if mean_dist > 0:
                variance = sum((d - mean_dist)**2 for d in distances) / len(distances)
                cv = (variance**0.5) / mean_dist
                coverage_score = max(0.2, 1.0 - cv)
            else:
                coverage_score = 0.2
        else:
            coverage_score = 0.5
    
    # 综合评分
    final_score = (
        0.40 * connectivity_score +
        0.25 * density_score +
        0.20 * topology_score +
        0.15 * coverage_score
    )
    
    return min(final_score, 1.0)

def calculate_additional_metrics(nodes, edges, obstacles, grid_width, grid_height):
    """计算额外的评估指标"""
    if not nodes:
        return {
            'num_components': 0,
            'largest_component_ratio': 0,
            'average_degree': 0,
            'node_density': 0,
            'edge_density': 0
        }
    
    # 连通分量分析
    adj = {i: [] for i in range(len(nodes))}
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    
    for edge in edges:
        if edge[0] in node_to_idx and edge[1] in node_to_idx:
            i, j = node_to_idx[edge[0]], node_to_idx[edge[1]]
            adj[i].append(j)
            adj[j].append(i)
    
    # 找连通分量
    visited = [False] * len(nodes)
    components = []
    
    def bfs(start):
        queue = [start]
        visited[start] = True
        component = [start]
        while queue:
            node = queue.pop(0)
            for neighbor in adj[node]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append(neighbor)
                    component.append(neighbor)
        return component
    
    for i in range(len(nodes)):
        if not visited[i]:
            components.append(bfs(i))
    
    metrics = {}
    metrics['num_components'] = len(components)
    if components:
        largest_comp_size = max(len(comp) for comp in components)
        metrics['largest_component_ratio'] = largest_comp_size / len(nodes)
    else:
        metrics['largest_component_ratio'] = 0
    
    # 度数统计
    total_degree = sum(len(adj[i]) for i in range(len(nodes)))
    metrics['average_degree'] = total_degree / len(nodes) if len(nodes) > 0 else 0
    
    # 密度统计
    total_cells = grid_width * grid_height
    obstacle_cells = len(obstacles)
    reachable_area = total_cells - obstacle_cells
    metrics['node_density'] = len(nodes) / max(reachable_area, 1)
    
    max_possible_edges = len(nodes) * (len(nodes) - 1) / 2
    metrics['edge_density'] = len(edges) / max(max_possible_edges, 1)
    
    return metrics

def save_metrics_to_file(metrics, filepath):
    """保存指标到文件"""
    # 确保目录存在
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # 保存为CSV
    csv_filepath = filepath
    file_exists = os.path.isfile(csv_filepath)
    
    with open(csv_filepath, 'a', newline='') as f:
        writer = csv.writer(f)
        
        # 如果文件不存在，写入表头
        if not file_exists:
            writer.writerow([
                'timestamp', 'algorithm', 'environment', 'seed',
                'generation_time', 'nodes_count', 'edges_count',
                'beam_connectivity_score', 'num_components', 'largest_component_ratio',
                'average_degree', 'node_density', 'edge_density',
                'target_nodes', 'connection_radius'
            ])
        
        # 写入数据
        writer.writerow([
            metrics['timestamp'],
            metrics['algorithm'],
            metrics['environment'],
            metrics['seed'],
            metrics['generation_time'],
            metrics['nodes_count'],
            metrics['edges_count'],
            metrics['beam_connectivity_score'],
            metrics['num_components'],
            metrics['largest_component_ratio'],
            metrics['average_degree'],
            metrics['node_density'],
            metrics['edge_density'],
            metrics['target_nodes'],
            metrics['connection_radius']
        ])

def run_algorithm_test(algorithm_name, environment_type, seed, num_nodes, save_images=True):
    """运行单个算法测试"""
    np.random.seed(seed)
    import random
    random.seed(seed)
    
    # 设置基础路径 - 修复路径
    base_results_path = "/Users/bytedance/Desktop/2025summer/code/ackermann_PEG/pursuer_strategies/PRM/results"
    
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
        ENV_CONFIG['gridnum_width'] = 40
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
    
    # 记录开始时间
    start_time = time.time()
    
    # 运行算法
    try:
        if algorithm_name == "classical":
            generator = ClassicalPRM(grid_width, grid_height, obstacles, 
                                    num_nodes=num_nodes, connection_radius=connection_radius)
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
                             max_samples=6000, target_guards=num_nodes,
                             delta=connection_radius, stretch_factor=1.3)
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
    
    # 计算连通性评分和额外指标
    beam_connectivity_score = calculate_beam_connectivity_score(nodes, edges, obstacles, grid_width, grid_height)
    additional_metrics = calculate_additional_metrics(nodes, edges, obstacles, grid_width, grid_height)
    
    # 准备指标
    metrics = {
        'timestamp': datetime.now().isoformat(),
        'algorithm': algorithm_name,
        'environment': environment_type,
        'seed': seed if environment_type != 'indoor' else 'N/A',  # indoor不使用随机种子
        'generation_time': round(generation_time, 4),
        'nodes_count': len(nodes),
        'edges_count': len(edges),
        'beam_connectivity_score': round(beam_connectivity_score, 4),
        'num_components': additional_metrics['num_components'],
        'largest_component_ratio': round(additional_metrics['largest_component_ratio'], 4),
        'average_degree': round(additional_metrics['average_degree'], 4),
        'node_density': round(additional_metrics['node_density'], 6),
        'edge_density': round(additional_metrics['edge_density'], 6),
        'target_nodes': num_nodes,
        'connection_radius': connection_radius
    }
    
    print(f"完成: {len(nodes)} 节点, {len(edges)} 边, "
          f"连通分量: {additional_metrics['num_components']}, "
          f"Beam评分: {beam_connectivity_score:.3f}, 时间: {generation_time:.2f}s")
    
    # 保存图像 - 使用generator.py中的PRMRenderer
    if save_images:
        # 创建保存路径
        env_dir = os.path.join(base_results_path, environment_type)
        os.makedirs(env_dir, exist_ok=True)
        
        # 生成文件名
        # 在run_algorithm_test函数中修改文件名生成逻辑
        if environment_type == 'indoor':
            filename = f"{algorithm_name}_{environment_type}_{num_nodes}.pdf"
        else:
            filename = f"{algorithm_name}_{environment_type}_{seed}_{num_nodes}.pdf"        
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
    num_nodes = [100, 200, 500, 800]
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
    base_results_path = "/Users/bytedance/Desktop/2025summer/code/ackermann_PEG/pursuer_strategies/PRM/results"
    
    print(f"开始运行 {total_tests} 个测试...")
    print(f"结果将保存到: {base_results_path}")
    
    for num_node in num_nodes:
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
                            metrics = run_algorithm_test(alg, env, seed, num_node, save_images=True)
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
                        metrics = run_algorithm_test(alg, env, 42, num_node, save_images=True)
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
    
    if successful_tests:
        # 按算法统计Beam评分
        print(f"\nBeam连通性评分统计:")
        from collections import defaultdict
        algo_scores = defaultdict(list)
        for m in successful_tests:
            algo_scores[m['algorithm']].append(m['beam_connectivity_score'])
        
        for algo, scores in algo_scores.items():
            avg_score = sum(scores) / len(scores)
            print(f"  {algo}: {avg_score:.4f} (测试数: {len(scores)})")
    
    return all_metrics

if __name__ == "__main__":
    run_full_evaluation()
