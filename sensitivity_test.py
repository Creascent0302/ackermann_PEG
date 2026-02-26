import numpy as np
import time
import pandas as pd
import sys
import os
import random
import pygame
import argparse
import math
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import griddata
import warnings

# 忽略因为 NaN 导致的绘图警告
warnings.filterwarnings('ignore')

sys.path.append('.')
from config import ENV_CONFIG
from generator import BeamPRM, PRMRenderer
from map_generator import generate_maze_obstacles, generate_indoor_obstacles

# =====================================================================
# 1. 图像渲染与保存模块 (纯白底色，学术规范)
# =====================================================================
class PathPRMRenderer(PRMRenderer):
    def render_path(self, nodes, edges, obstacles, path_nodes, path_edges, 
                   medial_axis_nodes=None, medial_axis_edges=None, medial_axis_paths=None, env="random", algorithm="beam"):
        screen = self.render(nodes, edges, obstacles, medial_axis_nodes, medial_axis_edges, medial_axis_paths, env, algorithm)
        
        if path_nodes and path_edges:
            for edge in path_edges:
                a, b = edge
                x1, y1 = a
                x2, y2 = b
                pygame.draw.line(screen, (220, 20, 60),  
                                (int(x1 * self.cell_size / ENV_CONFIG['cell_size']),
                                 int(y1 * self.cell_size / ENV_CONFIG['cell_size'])),
                                (int(x2 * self.cell_size / ENV_CONFIG['cell_size']),
                                 int(y2 * self.cell_size / ENV_CONFIG['cell_size'])), 4)
            
            if len(path_nodes) >= 2:
                start = path_nodes[0]
                pygame.draw.circle(screen, (34, 139, 34), 
                                  (int(start[0] * self.cell_size / ENV_CONFIG['cell_size']),
                                   int(start[1] * self.cell_size / ENV_CONFIG['cell_size'])),
                                  self.cell_size // 2, 0)
                goal = path_nodes[-1]
                pygame.draw.circle(screen, (30, 144, 255), 
                                  (int(goal[0] * self.cell_size / ENV_CONFIG['cell_size']),
                                   int(goal[1] * self.cell_size / ENV_CONFIG['cell_size'])),
                                  self.cell_size // 2, 0)
        
        if not self.headless:
            pygame.display.flip()
        return screen
    
    def save_path_image(self, nodes, edges, obstacles, path_nodes, path_edges, filepath, 
                       medial_axis_nodes=None, medial_axis_edges=None, medial_axis_paths=None, env="random", algorithm="beam"):
        screen = self.render_path(nodes, edges, obstacles, path_nodes, path_edges, 
                                 medial_axis_nodes, medial_axis_edges, medial_axis_paths, env, algorithm)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        pygame.image.save(screen, filepath)

# =====================================================================
# 2. 测试与数据生成模块
# =====================================================================
class BeamPRMParameterTester:
    def __init__(self, environment_type="maze", num_runs=3, num_paths=50):
        self.environment_type = environment_type
        self.num_runs = num_runs
        self.num_paths = num_paths 
        self.results = []
        self._setup_environment()
        self._generate_fixed_test_set()

    def _setup_environment(self):
        # 补全了 random 环境的定义，生成 40x40 大小、20% 障碍物密度的随机地图
        if self.environment_type == "random":
            ENV_CONFIG['gridnum_width'] = 40
            ENV_CONFIG['gridnum_height'] = 40
            self.grid_width = 40
            self.grid_height = 40
            self.obstacles = []
            
            # 添加边框
            for i in range(self.grid_width):
                self.obstacles.extend([(i, 0), (i, self.grid_height - 1)])
            for i in range(self.grid_height):
                self.obstacles.extend([(0, i), (self.grid_width - 1, i)])
            self.obstacles = list(set(self.obstacles))
            
            # 随机添加内部障碍物块
            np.random.seed(42)
            total_cells = 40 * 40
            num_obs = int(total_cells * 0.20)
            while len(self.obstacles) < num_obs:
                x = np.random.randint(1, self.grid_width - 1)
                y = np.random.randint(1, self.grid_height - 1)
                if (x, y) not in self.obstacles:
                    self.obstacles.append((x, y))
            self.base_connection_radius = 1.2
            
        elif self.environment_type == "maze":
            ENV_CONFIG['gridnum_width'] = 49
            ENV_CONFIG['gridnum_height'] = 49
            self.grid_width = ENV_CONFIG['gridnum_width']
            self.grid_height = ENV_CONFIG['gridnum_height']
            self.obstacles = generate_maze_obstacles(self.grid_width, self.grid_height)
            self.base_connection_radius = 1.5
            
        elif self.environment_type == "indoor":
            ENV_CONFIG['gridnum_width'] = 50
            ENV_CONFIG['gridnum_height'] = 50
            self.grid_width = ENV_CONFIG['gridnum_width']
            self.grid_height = ENV_CONFIG['gridnum_height']
            self.obstacles = generate_indoor_obstacles(self.grid_width, self.grid_height)
            self.base_connection_radius = 2.0
        else:
            raise ValueError("不支持的环境类型。")

    def _generate_fixed_test_set(self):
        print(f"[{self.environment_type.upper()}] 正在生成统一且具有挑战性的 {self.num_paths} 个测试起点与终点对...")
        np.random.seed(999); random.seed(999)
        dummy_planner = BeamPRM(self.grid_width, self.grid_height, self.obstacles)
        
        raw_pairs = dummy_planner.generate_valid_point_pairs(self.num_paths * 10)
        self.fixed_point_pairs = []
        map_diagonal = math.hypot(self.grid_width, self.grid_height)
        min_required_dist = map_diagonal / 3.0
        
        for start, goal in raw_pairs:
            dist = math.hypot(start[0] - goal[0], start[1] - goal[1])
            if dist >= min_required_dist:
                self.fixed_point_pairs.append((start, goal))
            if len(self.fixed_point_pairs) == self.num_paths:
                break
                
        if len(self.fixed_point_pairs) < self.num_paths:
            remaining = self.num_paths - len(self.fixed_point_pairs)
            self.fixed_point_pairs.extend(raw_pairs[:remaining])

    def _calculate_true_euclidean_length(self, path_nodes):
        if not path_nodes or len(path_nodes) < 2:
            return 0.0
        total_distance = 0.0
        for i in range(len(path_nodes) - 1):
            p1 = path_nodes[i]
            p2 = path_nodes[i+1]
            total_distance += math.hypot(p1[0] - p2[0], p1[1] - p2[1])
        return total_distance

    def run_tests(self, angle_steps, min_radii, node_budget=1000, save_samples=True, csv_filename="data.csv"):
        os.makedirs("prm_samples", exist_ok=True)
        self.results = [] 
        total_configs = len(angle_steps) * len(min_radii)
        current = 0
        
        for angle in angle_steps:
            for radius in min_radii:
                current += 1
                print(f"[{self.environment_type}] [{current}/{total_configs}] 参数: Angle={angle:2d}°, Radius={radius:.2f} ", end="", flush=True)
                
                config_results = {'times': [], 'path_lengths': [], 'search_times': [], 'success_rates': [], 'node_counts': []}
                sample_saved = False 
                
                for run_id in range(self.num_runs):
                    seed = 1000 + current * 10 + run_id 
                    np.random.seed(seed); random.seed(seed)

                    planner = BeamPRM(
                        grid_width=self.grid_width, grid_height=self.grid_height, 
                        obstacles=self.obstacles, num_nodes=node_budget,
                        connection_radius=self.base_connection_radius,
                        beam_angle_step_deg=angle, min_connection_radius=radius, beam_ray_step=0.2
                    )
                    
                    t_start = time.time()
                    try:
                        result = planner.generate_prm()
                        nodes, edges = planner.nodes, planner.edges
                        ma_nodes, ma_edges, ma_paths = (result[2], result[4], result[5]) if len(result) >= 6 else (set(), set(), [])
                    except Exception:
                        continue
                        
                    config_results['times'].append(time.time() - t_start)
                    config_results['node_counts'].append(len(nodes))
                        
                    run_success_count = 0
                    first_path_nodes, first_path_edges = None, None
                    
                    for (start, goal) in self.fixed_point_pairs:
                        try:
                            path_nodes, path_edges, _, s_time = planner.find_path(start, goal)
                            if path_nodes is not None:
                                run_success_count += 1
                                true_physical_length = self._calculate_true_euclidean_length(path_nodes)
                                config_results['path_lengths'].append(true_physical_length)
                                config_results['search_times'].append(s_time)
                                if first_path_nodes is None:
                                    first_path_nodes, first_path_edges = path_nodes, path_edges
                        except Exception:
                            pass
                            
                    config_results['success_rates'].append(run_success_count / self.num_paths)
                    
                    if save_samples and not sample_saved and len(nodes) > 0:
                        try:
                            filename = f"prm_samples/map_{self.environment_type}_ang{angle}_rad{radius}.png"
                            renderer = PathPRMRenderer(self.grid_width, self.grid_height, headless=True)
                            renderer.save_path_image(
                                nodes, edges, self.obstacles, 
                                first_path_nodes, first_path_edges, filename,
                                ma_nodes, ma_edges, ma_paths, env=self.environment_type, algorithm="beam"
                            )
                            sample_saved = True
                        except Exception:
                            pass

                if config_results['times']:
                    self.results.append({
                        'Angle Step (deg)': angle,
                        'Min Radius': radius,
                        'Avg Gen Time (s)': np.mean(config_results['times']),
                        'Avg Path Length': np.nanmean(config_results['path_lengths']) if config_results['path_lengths'] else float('nan'),
                        'Success Rate': np.mean(config_results['success_rates']),
                        'Avg Nodes': np.mean(config_results['node_counts'])
                    })
                    print(f"-> 成功率: {np.mean(config_results['success_rates']):.1%}")
                else:
                    print("-> 建图失败")
                    
        df = pd.DataFrame(self.results)
        if not df.empty:
            df.to_csv(csv_filename, index=False)
            print(f"  => 数据已保存至: {csv_filename}")
        return df

# =====================================================================
# 3. 独立学术绘图模块 (统一应用基于 Viridis_r 的学术极简等高线风格)
# =====================================================================
def load_data(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"找不到数据 {filepath}")
    return pd.read_csv(filepath)

def style_academic_contour(ax, x, y, z, title, cbar_label, z_min, z_max):
    """提取的通用绘图引擎：完全还原示例图的高级风格"""
    if z_max == z_min:
        z_max += 1e-5
        
    xi = np.linspace(x.min(), x.max(), 200)
    yi = np.linspace(y.min(), y.max(), 200)
    Xi, Yi = np.meshgrid(xi, yi)
    
    Zi = griddata((x, y), z, (Xi, Yi), method='cubic')
    Zi = np.clip(Zi, z_min, z_max)
    
    # [修改] 强制所有的等高线分层使用绝对范围 z_min 到 z_max，保证三个环境的颜色含义绝对统一
    levels = np.linspace(z_min, z_max, 60)
    c = ax.contourf(Xi, Yi, Zi, levels=levels, cmap='viridis_r', extend='both', zorder=0)
    
    ax.scatter(x, y, c='black', s=25, alpha=0.9, marker='x', zorder=2)
    
    ax.set_title(title, color='black', fontsize=14, pad=12, fontweight='bold')
    ax.set_xlabel("Angle Step (deg)", color='black', fontsize=12)
    ax.set_ylabel("Min Radius", color='black', fontsize=12)
    ax.tick_params(colors='black', labelsize=11)
    
    ax.grid(True, color='silver', linestyle='--', linewidth=1.0, alpha=0.9)
    ax.set_axisbelow(False) 
    
    for spine in ax.spines.values():
        spine.set_color('black')
        spine.set_linewidth(1.5)
        spine.set_zorder(3)
        
    cbar = plt.colorbar(c, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label, color='black', fontsize=12)
    cbar.ax.yaxis.set_tick_params(color='black')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='black')

# --- 图1: 三环境全局大范围成功率图 (1行3列) ---
def plot_standalone_success_rate_all(dfs_dict):
    fig = plt.figure(figsize=(20, 5), facecolor='white')
    fig.suptitle("Global Pathfinding Success Rate Distribution", fontsize=18, fontweight='bold', y=1.05)
    
    for i, (env_name, df) in enumerate(dfs_dict.items()):
        X = df['Angle Step (deg)'].values
        Y = df['Min Radius'].values
        Z_succ = df['Success Rate'].values
        
        ax = fig.add_subplot(1, 3, i + 1)
        # 成功率的统一标尺就是 0.0 到 1.0
        style_academic_contour(ax, X, Y, Z_succ, 
                              f"Env: {env_name.capitalize()}", 
                              "Success Rate", 0.0, 1.0)
    
    plt.tight_layout()
    # [修改] 保存为高质量 svg 矢量图
    save_path = 'plot1_success_rate_all.svg'
    plt.savefig(save_path, format='svg', bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"✅ 已生成图1 (三大环境全局成功率大图): {save_path}")

# --- 图2: 三环境局部综合栅格图 (3行2列) ---
def plot_classic_grid_all(dfs_dict):
    sns.set_theme(style="whitegrid")
    fig = plt.figure(figsize=(12, 14), facecolor='white')
    fig.suptitle("Sweet Spot Analysis - Classic Grid Maps (Node Density & Success Rate)", fontsize=22, fontweight='bold', y=0.98)
    
    # [修改] 提取三大环境的全局节点数最小和最大值
    global_min_nodes = min(df['Avg Nodes'].min() for df in dfs_dict.values())
    global_max_nodes = max(df['Avg Nodes'].max() for df in dfs_dict.values())
    
    for i, (env_name, df) in enumerate(dfs_dict.items()):
        row_offset = i * 2  
        
        # 第一列：节点密度
        ax1 = fig.add_subplot(3, 2, row_offset + 1)
        pivot_nodes = df.pivot(index='Min Radius', columns='Angle Step (deg)', values='Avg Nodes')
        # [修改] 传入 vmin 和 vmax 参数以保证三个子图的数值-颜色映射完全统一
        sns.heatmap(pivot_nodes, annot=True, fmt=".0f", cmap='viridis_r', ax=ax1, 
                    vmin=global_min_nodes, vmax=global_max_nodes, 
                    cbar_kws={'label': 'Node Count'})
        ax1.set_title(f'[{env_name.capitalize()}] Generated Node Density', fontsize=14, pad=10, fontweight='bold')
        ax1.invert_yaxis()

        # 第二列：成功率
        ax2 = fig.add_subplot(3, 2, row_offset + 2)
        pivot_succ = df.pivot(index='Min Radius', columns='Angle Step (deg)', values='Success Rate')
        # [修改] vmin=0, vmax=1 原来就有，保证了成功率颜色是一致的
        sns.heatmap(pivot_succ, annot=True, fmt=".0%", cmap='viridis_r', vmin=0, vmax=1, ax=ax2, cbar_kws={'label': 'Success Rate'})
        ax2.set_title(f'[{env_name.capitalize()}] Pathfinding Success Rate', fontsize=14, pad=10, fontweight='bold')
        ax2.invert_yaxis()

    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    # [修改] 保存为高质量 svg 矢量图
    save_path = 'plot2_classic_grid_all.svg'
    plt.savefig(save_path, format='svg', bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"✅ 已生成图2 (三大环境经典方块栅格图): {save_path}")

# --- 图3: 三环境局部平滑插值热力图 (3行2列) ---
def plot_smooth_contours_all(dfs_dict):
    plt.style.use('default')
    fig = plt.figure(figsize=(14, 16), facecolor='white')
    fig.suptitle("Sweet Spot Analysis - Smooth Contour Maps (Node Density & Success Rate)", fontsize=22, fontweight='bold', color='black', y=0.96)
    
    # [修改] 提取三大环境的全局节点数最小和最大值
    global_min_nodes = min(df['Avg Nodes'].min() for df in dfs_dict.values())
    global_max_nodes = max(df['Avg Nodes'].max() for df in dfs_dict.values())
    
    for i, (env_name, df) in enumerate(dfs_dict.items()):
        row_offset = i * 2  
        X = df['Angle Step (deg)'].values
        Y = df['Min Radius'].values
        
        # 第一列：节点密度
        ax1 = fig.add_subplot(3, 2, row_offset + 1)
        Z_nodes = df['Avg Nodes'].values
        # [修改] 原先使用 Z_nodes.min() 和 Z_nodes.max()，现在替换为统一的 global_min_nodes 和 global_max_nodes
        style_academic_contour(ax1, X, Y, Z_nodes, 
                               f"[{env_name.capitalize()}] Generated Node Density", 
                               "Node Count", global_min_nodes, global_max_nodes)

        # 第二列：成功率
        ax2 = fig.add_subplot(3, 2, row_offset + 2)
        Z_succ = df['Success Rate'].values
        style_academic_contour(ax2, X, Y, Z_succ, 
                               f"[{env_name.capitalize()}] Pathfinding Success Rate", 
                               "Success Rate", 0.0, 1.0)

    plt.tight_layout(rect=[0, 0.02, 1, 0.93])
    # [修改] 保存为高质量 svg 矢量图
    save_path = 'plot3_smooth_contour_all.svg'
    plt.savefig(save_path, format='svg', facecolor='white', edgecolor='none', bbox_inches='tight')
    plt.close(fig)
    print(f"✅ 已生成图3 (三大环境平滑等高热力图): {save_path}")

# =====================================================================
# 4. 主程序控制流程
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="BeamPRM 敏感度测试与多图生成")
    parser.add_argument("--action", type=str, choices=["test", "plot", "both"], default="both", 
                        help="'test':跑测试, 'plot':只画图, 'both':先跑后画(默认)")
    parser.add_argument("--budget", type=int, default=1000)
    args = parser.parse_args()

    envs = ["random", "maze", "indoor"]
    
    test_angles = [3, 5, 15, 20, 25, 30, 40, 45]          
    test_radii_large = [0.2, 0.4, 0.6, 0.8]  
    test_radii_small = [0.2, 0.3, 0.4, 0.5, 0.6, 0.8]

    if args.action in ["test", "both"]:
        print("========== 阶段 1: 批量运行三大环境测试 ==========")
        for env in envs:
            csv_large = f'sensitivity_data_large_{env}.csv'
            csv_small = f'sensitivity_data_small_{env}.csv'
            
            print(f"\n---> 开始环境: {env.upper()}")
            tester = BeamPRMParameterTester(environment_type=env, num_runs=3, num_paths=50)
            
            print(f"  > 1.1 大范围全局参数 (用于图1)")
            tester.run_tests(test_angles, test_radii_large, node_budget=args.budget, save_samples=False, csv_filename=csv_large)

            print(f"  > 1.2 小范围局部参数 (用于图2、图3)")
            tester.run_tests(test_angles, test_radii_small, node_budget=args.budget, save_samples=True, csv_filename=csv_small)

    if args.action in ["plot", "both"]:
        print("\n========== 阶段 2: 合并生成学术大图 ==========")
        try:
            dfs_large = {}
            dfs_small = {}
            for env in envs:
                dfs_large[env] = load_data(f'sensitivity_data_large_{env}.csv')
                dfs_small[env] = load_data(f'sensitivity_data_small_{env}.csv')
            
            plot_standalone_success_rate_all(dfs_large)
            plot_classic_grid_all(dfs_small)
            plot_smooth_contours_all(dfs_small)
            
            print("\n🎉 全部绘图任务圆满完成！输出文件均为高质量 svg 格式。")
            
        except FileNotFoundError as e:
            print(f"❌ 读取数据失败: {e} \n(提示：请先使用 '--action test' 完整生成数据)")

if __name__ == "__main__":
    main()