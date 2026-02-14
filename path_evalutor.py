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
from base_generator import BasePathPlanner

def save_path_metrics_to_file(metrics, filepath):
    """保存路径规划指标到文件"""
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
            metrics['path_length'],
            metrics['search_time'],
            metrics['start'],
            metrics['goal'],
            metrics['success']
        ])

class PathPRMRenderer(PRMRenderer):
    """扩展PRMRenderer以支持路径可视化"""
    
    def render_path(self, nodes, edges, obstacles, path_nodes, path_edges, 
                   medial_axis_nodes=None, medial_axis_edges=None, medial_axis_paths=None, environment_type="random", algorithm="delta"):
        """渲染PRM和路径"""
        # 首先渲染基本的PRM
        screen = self.render(nodes, edges, obstacles, medial_axis_nodes, medial_axis_edges, medial_axis_paths, environment_type, algorithm)
        
        # 再渲染路径（如果有）
        if path_nodes and path_edges:
            # 绘制路径边
            for edge in path_edges:
                a, b = edge
                x1, y1 = a
                x2, y2 = b
                # 绘制粗红色线表示路径
                pygame.draw.line(screen, (255, 0, 0),
                                (int(x1 * self.cell_size / ENV_CONFIG['cell_size']),
                                 int(y1 * self.cell_size / ENV_CONFIG['cell_size'])),
                                (int(x2 * self.cell_size / ENV_CONFIG['cell_size']),
                                 int(y2 * self.cell_size / ENV_CONFIG['cell_size'])), 4)
            
            # 突出显示起点和终点
            if len(path_nodes) >= 2:
                # 起点（绿色）
                start = path_nodes[0]
                pygame.draw.circle(screen, (0, 255, 0),
                                  (int(start[0] * self.cell_size / ENV_CONFIG['cell_size']),
                                   int(start[1] * self.cell_size / ENV_CONFIG['cell_size'])),
                                  self.cell_size // 2, 0)
                
                # 终点（蓝色）
                goal = path_nodes[-1]
                pygame.draw.circle(screen, (0, 0, 255),
                                  (int(goal[0] * self.cell_size / ENV_CONFIG['cell_size']),
                                   int(goal[1] * self.cell_size / ENV_CONFIG['cell_size'])),
                                  self.cell_size // 2, 0)
        
        if not self.headless:
            pygame.display.flip()
            
        return screen
    
    def save_path_image(self, nodes, edges, obstacles, path_nodes, path_edges, filepath, medial_axis_nodes=None, medial_axis_edges=None, medial_axis_paths=None, environment_type="random", algorithm="delta"):
        """渲染并保存带路径的图像"""
        screen = self.render_path(nodes, edges, obstacles, path_nodes, path_edges, medial_axis_nodes, medial_axis_edges, medial_axis_paths, environment_type, algorithm)
        save_pdf_image(screen, filepath)
        # print(f"路径图像已保存到: {filepath}")
    
    def run_with_path(self, nodes, edges, obstacles, path_nodes, path_edges, medial_axis_nodes=None, medial_axis_edges=None, medial_axis_paths=None):
        """运行带路径的渲染器（交互模式）"""
        running = True
        
        # 先渲染一次，保存背景图像
        screen = self.render(nodes, edges, obstacles, medial_axis_nodes, medial_axis_edges, medial_axis_paths)
        background = screen.copy()
        
        # 在背景上绘制路径
        if path_nodes and path_edges:
            for edge in path_edges:
                a, b = edge
                x1, y1 = a
                x2, y2 = b
                pygame.draw.line(screen, (255, 0, 0),
                                (int(x1 * self.cell_size / ENV_CONFIG['cell_size']),
                                 int(y1 * self.cell_size / ENV_CONFIG['cell_size'])),
                                (int(x2 * self.cell_size / ENV_CONFIG['cell_size']),
                                 int(y2 * self.cell_size / ENV_CONFIG['cell_size'])), 4)
        
        if len(path_nodes) >= 2:
            # 绘制起点（绿色）和终点（蓝色）
            start = path_nodes[0]
            pygame.draw.circle(screen, (0, 255, 0),
                              (int(start[0] * self.cell_size / ENV_CONFIG['cell_size']),
                               int(start[1] * self.cell_size / ENV_CONFIG['cell_size'])),
                              self.cell_size // 2, 0)
            
            goal = path_nodes[-1]
            pygame.draw.circle(screen, (0, 0, 255),
                              (int(goal[0] * self.cell_size / ENV_CONFIG['cell_size']),
                               int(goal[1] * self.cell_size / ENV_CONFIG['cell_size'])),
                              self.cell_size // 2, 0)
    
        # 更新显示一次
        pygame.display.flip()
        
        # 主循环中只处理事件，不重复渲染
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            self.clock.tick(30)
        pygame.quit()

def test_path_planning(algorithm_name, environment_type, seed, num_path_tests=10, save_images=True):
    """测试单个算法的路径规划能力"""
    # 设置随机种子
    np.random.seed(seed)
    import random
    random.seed(seed)
    
    # 设置基础路径
    base_results_path = "./pursuer_strategies/PRM/results/paths"
    
    # 设置环境参数
    if environment_type == "maze":
        ENV_CONFIG['gridnum_width'] = 49
        ENV_CONFIG['gridnum_height'] = 49
        grid_width = ENV_CONFIG['gridnum_width']
        grid_height = ENV_CONFIG['gridnum_height']
        obstacles = generate_maze_obstacles(grid_width, grid_height)
    elif environment_type == "indoor":
        ENV_CONFIG['gridnum_width'] = 50
        ENV_CONFIG['gridnum_height'] = 50
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
    
    # 设置算法参数
    sample_nodes_map = {
        'maze': {'delta': 2000, 'star': 1000, 'beam': 500, 'spars': 3000},
        'indoor': {'delta': 2000, 'star': 1000, 'beam': 500, 'spars': 3000},
        'random': {'delta': 2000, 'star': 800, 'beam': 500, 'spars': 2000}
    }
    
    print(f"\n运行路径规划测试: {algorithm_name} on {environment_type} (seed={seed})")
    num_nodes = sample_nodes_map.get(environment_type, {}).get(algorithm_name, 300)
    
    # 初始化PRM生成器
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
    
    # 生成PRM
    print(f"正在生成 {algorithm_name} PRM...")
    start_time = time.time()
    if algorithm_name == "beam":
        result = generator.generate_prm()
        if len(result) >= 6:
            nodes, edges, medial_axis_nodes, medial_axis_all_nodes, medial_axis_edges, medial_axis_paths = result
        else:
            nodes, edges = result[:2]
            medial_axis_nodes, medial_axis_edges, medial_axis_paths = set(), set(), []
    else:
        nodes, edges = generator.generate_prm()
        medial_axis_nodes, medial_axis_edges, medial_axis_paths = set(), set(), []
    
    prm_gen_time = time.time() - start_time
    print(f"PRM生成完成，用时: {prm_gen_time:.2f}秒")
    
    np.random.seed(seed)
    random.seed(seed) # 确保路径测试的一致性
    # 生成测试点对
    print(f"生成 {num_path_tests} 个起终点对...")
    point_pairs = generator.generate_valid_point_pairs(num_path_tests)
    
    # 测试每个点对的路径规划
    path_metrics = []
    
    for i, (start, goal) in enumerate(point_pairs):
        # print(f"测试路径 {i+1}/{num_path_tests}: {start} -> {goal}")
        
        # 测量路径搜索时间和结果
        path_nodes, path_edges, path_length, search_time = generator.find_path(start, goal)
        
        success = path_nodes is not None
        status = "成功" if success else "失败"
        # print(f"  路径搜索{status}，长度: {path_length if success else 'N/A'}，用时: {search_time:.4f}秒")
        
        # 记录指标
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'algorithm': algorithm_name,
            'environment': environment_type,
            'seed': seed if environment_type != 'indoor' else 'N/A',
            'path_length': path_length if success else -1,
            'search_time': search_time,
            'start': str(start),
            'goal': str(goal),
            'success': success
        }
        path_metrics.append(metrics)
        
        # 保存路径规划结果图像
        if save_images and success:
            # 创建保存路径
            env_dir = os.path.join(base_results_path, environment_type, algorithm_name)
            os.makedirs(env_dir, exist_ok=True)
            
            # 生成文件名
            filename = f"path_{i}_{seed}.pdf" if environment_type != 'indoor' else f"path_{i}.pdf"
            filepath = os.path.join(env_dir, filename)
            
            # 渲染并保存路径图像
            renderer = PathPRMRenderer(grid_width, grid_height, headless=True)
            renderer.save_path_image(nodes, edges, obstacles, path_nodes, path_edges, filepath, medial_axis_nodes, medial_axis_edges, medial_axis_paths, environment_type, algorithm_name)

    # 保存所有指标到CSV
    metrics_file = os.path.join(base_results_path, f"path_metrics_{environment_type}.csv")
    os.makedirs(os.path.dirname(metrics_file), exist_ok=True)
    
    # 检查文件是否存在，若不存在则写入表头
    if not os.path.exists(metrics_file):
        with open(metrics_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'algorithm', 'environment', 'seed',
                'path_length', 'search_time', 'start', 'goal', 'success'
            ])
    
    # 写入数据
    for metrics in path_metrics:
        save_path_metrics_to_file(metrics, metrics_file)
    
    # 计算成功率和平均值
    success_count = sum(1 for m in path_metrics if m['success'])
    success_rate = success_count / len(path_metrics) if path_metrics else 0
    
    # 只计算成功路径的平均值
    successful_metrics = [m for m in path_metrics if m['success']]
    avg_path_length = sum(m['path_length'] for m in successful_metrics) / len(successful_metrics) if successful_metrics else 0
    avg_search_time = sum(m['search_time'] for m in successful_metrics) / len(successful_metrics) if successful_metrics else 0
    
    print(f"\n测试完成！")
    print(f"成功率: {success_rate*100:.1f}% ({success_count}/{len(path_metrics)})")
    print(f"平均路径长度: {avg_path_length:.4f}")
    print(f"平均搜索时间: {avg_search_time:.4f}秒")
    
    # 返回总结统计
    summary = {
        'algorithm': algorithm_name,
        'environment': environment_type,
        'seed': seed,
        'prm_gen_time': prm_gen_time,
        'success_rate': success_rate,
        'avg_path_length': avg_path_length,
        'avg_search_time': avg_search_time,
        'test_count': len(path_metrics)
    }
    
    return summary, path_metrics

def run_full_path_evaluation(num_path_tests=10):
    """运行完整的路径规划评测流程"""
    # algorithms = ['delta', 'star', 'beam', 'spars']
    algorithms = ['delta', 'beam', 'spars']
    seeds = [43, 114, 520]  # 三个固定的随机种子
    
    # 定义测试配置: (环境, 是否使用随机种子)
    test_configs = [
        ('maze', True),      # 迷宫使用随机种子
        ('indoor', False),   # 室内不使用随机种子
        ('random', True)     # 随机使用随机种子
    ]
    
    all_summaries = []
    
    # 计算总测试数
    total_tests = 0
    for env, use_seeds in test_configs:
        if use_seeds:
            total_tests += len(algorithms) * len(seeds)
        else:
            total_tests += len(algorithms)
    
    current_test = 0
    base_results_path = "./pursuer_strategies/PRM/results/paths"
    
    print(f"开始运行 {total_tests} 个路径规划评测...")
    print(f"每个评测将测试 {num_path_tests} 条路径")
    print(f"结果将保存到: {base_results_path}")
    
    # 清空汇总CSV文件
    summary_file = os.path.join(base_results_path, "path_evaluation_summary.csv")
    os.makedirs(os.path.dirname(summary_file), exist_ok=True)
    with open(summary_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'algorithm', 'environment', 'seed',
            'prm_gen_time', 'success_rate', 'avg_path_length', 
            'avg_search_time', 'test_count'
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
                        summary, _ = test_path_planning(alg, env, seed, num_path_tests)
                        all_summaries.append(summary)
                        
                        # 保存单个汇总结果
                        with open(summary_file, 'a', newline='') as f:
                            writer = csv.writer(f)
                            writer.writerow([
                                summary['algorithm'],
                                summary['environment'],
                                summary['seed'],
                                summary['prm_gen_time'],
                                summary['success_rate'],
                                summary['avg_path_length'],
                                summary['avg_search_time'],
                                summary['test_count']
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
                    summary, _ = test_path_planning(alg, env, 42, num_path_tests)
                    all_summaries.append(summary)
                    
                    # 保存单个汇总结果
                    with open(summary_file, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            summary['algorithm'],
                            summary['environment'],
                            summary['seed'],
                            summary['prm_gen_time'],
                            summary['success_rate'],
                            summary['avg_path_length'],
                            summary['avg_search_time'],
                            summary['test_count']
                        ])
                    
                except Exception as e:
                    print(f"错误: {alg} on {env}: {str(e)}")
                    import traceback
                    traceback.print_exc()
    
    # 保存汇总结果到JSON
    json_summary_file = os.path.join(base_results_path, "path_evaluation_summary.json")
    with open(json_summary_file, 'w') as f:
        json.dump(all_summaries, f, indent=2)
    
    print(f"\n{'='*60}")
    print("路径规划评测完成！")
    print(f"{'='*60}")
    print(f"结果保存在: {base_results_path}")
    print(f"路径图像: {base_results_path}/{{environment}}/{{algorithm}}/path_{{i}}_{{seed}}.pdf")
    print(f"路径指标: {base_results_path}/path_metrics_{{environment}}.csv")
    print(f"汇总文件: {summary_file} 和 {json_summary_file}")
    
    return all_summaries

def demo_path_planning(algorithm_name="beam", environment_type="maze", seed=42, interactive=True, path_seed=114):
    """演示单条路径规划，支持交互式查看"""
    np.random.seed(seed)
    import random
    random.seed(seed)
    
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
        connection_radius = 1.0
    
    # 选择算法并设置参数
    sample_nodes_map = {
        'maze': {'delta': 1500, 'star': 1500, 'beam': 300, 'spars': 1500},
        'indoor': {'delta': 1500, 'star': 1500, 'beam': 300, 'spars': 1800},
        'random': {'delta': 1200, 'star': 1200, 'beam': 300, 'spars': 1000}
    }
    num_nodes = sample_nodes_map.get(environment_type, {}).get(algorithm_name, 300)
    
    # 初始化PRM生成器
    generator = None
    if algorithm_name == "delta":
        generator = DeltaPRM(grid_width, grid_height, obstacles, num_nodes=num_nodes, connection_radius=connection_radius)
    elif algorithm_name == "star":
        generator = PRMStar(grid_width, grid_height, obstacles, num_nodes=num_nodes, gamma_prm_star=15.0)
    elif algorithm_name == "beam":
        if environment_type == "random":
            generator = BeamPRM(grid_width, grid_height, obstacles,
                               num_nodes=num_nodes, connection_radius=1.2,
                               beam_angle_step_deg=2, beam_ray_step=0.08,
                               min_connection_radius=0.25)
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
    elif algorithm_name == "spars":
        generator = SPARS(grid_width, grid_height, obstacles, num_nodes=num_nodes)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm_name}")
    
    # 生成PRM
    print(f"正在生成 {algorithm_name} PRM...")
    if algorithm_name == "beam":
        result = generator.generate_prm()
        if len(result) >= 6:
            nodes, edges, medial_axis_nodes, medial_axis_all_nodes, medial_axis_edges, medial_axis_paths = result
        else:
            nodes, edges = result[:2]
            medial_axis_nodes, medial_axis_edges, medial_axis_paths = set(), set(), []
    else:
        nodes, edges = generator.generate_prm()
        medial_axis_nodes, medial_axis_edges, medial_axis_paths = set(), set(), []
    
    np.random.seed(path_seed)  # 重置随机种子
    random.seed(path_seed)
    # 生成测试点对
    point_pair = generator.generate_valid_point_pairs(1)[0]
    start, goal = point_pair
    
    # 执行路径规划
    print(f"路径规划: {start} -> {goal}")
    path_nodes, path_edges, path_length, search_time = generator.find_path(start, goal)
    
    if path_nodes:
        print(f"成功找到路径，长度: {path_length:.4f}，用时: {search_time:.4f}秒")
        
        # 渲染路径
        renderer = PathPRMRenderer(grid_width, grid_height, headless=not interactive)
        
        if interactive:
            # 交互式显示
            renderer.run_with_path(nodes, edges, obstacles, path_nodes, path_edges,
                                 medial_axis_nodes, medial_axis_edges, medial_axis_paths)
        else:
            # 保存图像
            filepath = f"./demo_{algorithm_name}_{environment_type}_{seed}.pdf"
            renderer.save_path_image(nodes, edges, obstacles, path_nodes, path_edges,
                                    filepath, medial_axis_nodes, medial_axis_edges, medial_axis_paths)
            print(f"路径图像已保存到: {filepath}")
    else:
        print(f"未能找到路径，搜索用时: {search_time:.4f}秒")
    
    return path_nodes, path_edges, path_length, search_time

if __name__ == "__main__":
    # 您可以选择运行完整评测或者演示单个路径规划
    
    # 选项1: 运行完整评测 (测试所有算法在所有环境下的性能)
    run_full_path_evaluation(num_path_tests=100)
    
    # 选项2: 演示单个路径规划 (交互式查看结果)
    # demo_path_planning(algorithm_name="beam", environment_type="maze", seed=42, interactive=True)

    # 选项3: 测试单个算法在单个环境下的路径规划能力
    # test_path_planning("beam", "maze", 42, num_path_tests=5)