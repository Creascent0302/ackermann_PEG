
"""
快速可扩展性评测脚本 - 用于快速验证代码正确性

与完整版的唯一区别是减少了 sample_counts、num_seeds 和算法数量。
所有核心逻辑直接复用 scalability_evaluator.py 中的函数。
"""

import os
import sys
import traceback
sys.path.append('.')

from scalability_evaluator import (
    run_scalability_test,
    save_scalability_metrics_to_file,
    generate_fixed_test_points,
)


def main():
    """快速可扩展性评测入口"""

    algorithms   = ['delta', 'beam', 'spars', 'gsrm', 'odrm']
    environments = ['random', 'maze', 'indoor']

    # 横坐标：采样次数（比完整版更稀疏）
    sample_counts = [200, 500, 1000]

    num_seeds = 3
    seeds = list(range(100, 100 + num_seeds))

    output_file = "./pursuer_strategies/PRM/results/scalability_evaluation.csv"

    if os.path.exists(output_file):
        os.remove(output_file)
        print(f"已删除旧文件: {output_file}\n")

    total = len(algorithms) * len(environments) * len(sample_counts) * num_seeds
    print("=" * 60)
    print("开始快速可扩展性评测（横坐标：采样次数）")
    print("=" * 60)
    print(f"算法:       {algorithms}")
    print(f"环境:       {environments}")
    print(f"采样次数:   {sample_counts}")
    print(f"Seeds/配置: {num_seeds}")
    print(f"总测试数:   {total}")

    # 预生成固定起终点（复用主模块函数，确保与完整版完全一致）
    print("\n生成固定测试起终点...")
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
                        m = run_scalability_test(
                            algo, env, ns, seed,
                            fixed_point_pairs=fixed_points[env]
                        )
                        save_scalability_metrics_to_file(m, output_file)
                        ok += 1
                        print(f"✓  {algo} | {env} | samples={ns} | seed={seed}")
                        print(f"   生成时间={m['generation_time']:.3f}s  "
                              f"实际节点={m['actual_nodes_count']}  "
                              f"采样效率={m['sampling_efficiency']*100:.1f}%")
                        print(f"   平均度={m['avg_degree']:.2f}  "
                              f"连通比={m['largest_component_ratio']*100:.1f}%  "
                              f"离散度={m['dispersion']:.4f}")
                        if m['path_success'] > 0:
                            print(f"   成功率={m['path_success']:.1f}%  "
                                  f"路径长={m['path_length']:.3f}  "
                                  f"优化比={m['path_optimality_ratio']:.3f}  "
                                  f"搜索时间={m['search_time']:.4f}s")
                    except Exception as e:
                        print(f"✗  {algo} | {env} | samples={ns} | seed={seed}")
                        print(f"   错误: {e}")
                        traceback.print_exc()

    print("\n" + "=" * 60)
    print("快速评测完成！")
    print("=" * 60)
    print(f"结果文件: {output_file}")
    print(f"成功/总计: {ok}/{done}")
    print("=" * 60)


if __name__ == "__main__":
    main()
