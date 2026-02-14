"""
快速可扩展性评测脚本 - 用于快速测试
"""

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

# 导入完整版的函数
from scalability_evaluator import run_scalability_test, save_scalability_metrics_to_file

def main():
    """主函数 - 运行快速可扩展性评测"""
    
    # 配置参数 - 减少测试量
    algorithms = ['delta', 'beam', 'spars']
    environments = ['random', 'maze', 'indoor']  # 测试所有环境
    
    # 更少的顶点数配置
    node_counts = [200, 500, 1000]
    
    # 更少的seed数量
    num_seeds = 3
    seeds = list(range(100, 100 + num_seeds))
    
    # 输出路径
    output_file = "./pursuer_strategies/PRM/results/scalability_evaluation.csv"
    
    # 删除旧的CSV文件（如果存在）
    if os.path.exists(output_file):
        os.remove(output_file)
        print(f"已删除旧文件: {output_file}\n")
    
    print("="*60)
    print("开始快速可扩展性评测")
    print("="*60)
    print(f"算法: {algorithms}")
    print(f"环境: {environments}")
    print(f"顶点数范围: {node_counts}")
    print(f"每个配置测试 {num_seeds} 个不同的seed")
    print(f"总测试数: {len(algorithms) * len(environments) * len(node_counts) * num_seeds}")
    
    # 为每个环境生成固定的起终点对
    print("\n生成固定的测试起终点...")
    print("="*60)
    from scalability_evaluator import generate_fixed_test_points
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
