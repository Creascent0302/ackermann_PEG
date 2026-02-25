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
# 2. 测试与数据生成模块 (分离大范围和小范围)
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
        if self.environment_type == "maze":
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
        print(f"正在生成统一且具有挑战性的 {self.num_paths} 个测试起点与终点对...")
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
                print(f"[{current}/{total_configs}] 参数: Angle={angle:2d}°, Radius={radius:.2f} ", end="", flush=True)
                
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
            print(f"\n✅ 测试完成！数据已保存至: {csv_filename}")
        return df

# =====================================================================
# 3. 独立学术绘图模块 (统一应用基于 Viridis_r 的学术极简等高线风格)
# =====================================================================
def load_data(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"找不到数据 {filepath}，请先使用 '--action test' 生成数据。")
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
    
    # 1. 绘制等高线热力图 (使用viridis_r: 紫高黄低)
    c = ax.contourf(Xi, Yi, Zi, levels=60, cmap='viridis_r', extend='both', zorder=0)
    
    # 2. 绘制黑色采样点 'x'
    ax.scatter(x, y, c='black', s=25, alpha=0.9, marker='x', zorder=2)
    
    # 3. 字体与标题排版
    ax.set_title(title, color='black', fontsize=14, pad=12, fontweight='bold')
    ax.set_xlabel("Angle Step (deg)", color='black', fontsize=12)
    ax.set_ylabel("Min Radius", color='black', fontsize=12)
    ax.tick_params(colors='black', labelsize=11)
    
    # 4. 强制网格线显示在颜色之上
    ax.grid(True, color='silver', linestyle='--', linewidth=1.0, alpha=0.9)
    ax.set_axisbelow(False) # 关键代码：禁止网格线被填充图层遮挡
    
    # 5. 加粗图表边框
    for spine in ax.spines.values():
        spine.set_color('black')
        spine.set_linewidth(1.5)
        spine.set_zorder(3)
        
    # 6. 配置侧边颜色栏
    cbar = plt.colorbar(c, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label, color='black', fontsize=12)
    cbar.ax.yaxis.set_tick_params(color='black')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='black')

# --- 图1: 单独的大范围成功率大图 (全面升级为平滑热力图) ---
def plot_standalone_success_rate(df, env_name):
    X = df['Angle Step (deg)'].values
    Y = df['Min Radius'].values
    Z_succ = df['Success Rate'].values
    
    plt.style.use('default')
    fig = plt.figure(figsize=(9, 7), facecolor='white')
    ax = fig.add_subplot(111)
    
    style_academic_contour(ax, X, Y, Z_succ, 
                          "Global Pathfinding Success Rate Distribution", 
                          "Success Rate", 0.0, 1.0)
    
    plt.tight_layout()
    save_path = f'plot1_success_rate_{env_name}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"已生成图1 (全局大范围成功率等高热力图): {save_path}")

# --- 图2: 小范围局部综合栅格图 (仅统一色彩为 viridis_r) ---
def plot_classic_grid(df, env_name):
    plt.style.use('default')
    sns.set_theme(style="whitegrid")
    fig = plt.figure(figsize=(16, 14), facecolor='white')
    fig.suptitle("Sweet Spot Analysis - Classic Grid Maps", fontsize=24, fontweight='bold', y=0.96)
    
    max_len = df['Avg Path Length'].max()
    penalty_len = max_len * 1.1 if not pd.isna(max_len) else 100

    ax1 = fig.add_subplot(221)
    pivot_nodes = df.pivot(index='Min Radius', columns='Angle Step (deg)', values='Avg Nodes')
    sns.heatmap(pivot_nodes, annot=True, fmt=".0f", cmap='viridis_r', ax=ax1, cbar_kws={'label': 'Node Count'})
    ax1.set_title('Generated Node Density', fontsize=16, pad=10, fontweight='bold')
    ax1.invert_yaxis()

    ax2 = fig.add_subplot(222)
    pivot_time = df.pivot(index='Min Radius', columns='Angle Step (deg)', values='Avg Gen Time (s)')
    sns.heatmap(pivot_time, annot=True, fmt=".2f", cmap='viridis_r', ax=ax2, cbar_kws={'label': 'Time (s)'})
    ax2.set_title('Avg PRM Generation Time (s)', fontsize=16, pad=10, fontweight='bold')
    ax2.invert_yaxis()

    ax3 = fig.add_subplot(223)
    pivot_succ = df.pivot(index='Min Radius', columns='Angle Step (deg)', values='Success Rate')
    sns.heatmap(pivot_succ, annot=True, fmt=".0%", cmap='viridis_r', vmin=0, vmax=1, ax=ax3, cbar_kws={'label': 'Success Rate'})
    ax3.set_title('Pathfinding Success Rate', fontsize=16, pad=10, fontweight='bold')
    ax3.invert_yaxis()

    ax4 = fig.add_subplot(224)
    pivot_len = df.pivot(index='Min Radius', columns='Angle Step (deg)', values='Avg Path Length')
    annot_len = pivot_len.copy()
    annot_len = annot_len.applymap(lambda x: "Fail" if pd.isna(x) else f"{x:.1f}")
    pivot_len_filled = pivot_len.fillna(penalty_len)
    sns.heatmap(pivot_len_filled, annot=annot_len, fmt="", cmap='viridis_r', ax=ax4, cbar_kws={'label': 'Physical Path Length'})
    ax4.set_title('Avg Euclidean Path Length (Fail=No Path)', fontsize=16, pad=10, fontweight='bold')
    ax4.invert_yaxis()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = f'plot2_classic_grid_{env_name}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"已生成图2 (局部甜点区经典方块栅格图): {save_path}")

# --- 图3: 小范围局部平滑插值热力图 (全面匹配示例图质感) ---
def plot_smooth_contours(df, env_name):
    X = df['Angle Step (deg)'].values
    Y = df['Min Radius'].values
    
    plt.style.use('default')
    fig = plt.figure(figsize=(16, 14), facecolor='white')
    fig.suptitle("Sweet Spot Analysis - Smooth Contour Maps", fontsize=24, fontweight='bold', color='black', y=0.96)
    
    ax1 = fig.add_subplot(221)
    Z_nodes = df['Avg Nodes'].values
    style_academic_contour(ax1, X, Y, Z_nodes, "Generated Node Density", "Node Count", Z_nodes.min(), Z_nodes.max())

    ax2 = fig.add_subplot(222)
    Z_time = df['Avg Gen Time (s)'].values
    style_academic_contour(ax2, X, Y, Z_time, "Avg PRM Generation Time (s)", "Time (s)", Z_time.min(), Z_time.max())

    ax3 = fig.add_subplot(223)
    Z_succ = df['Success Rate'].values
    style_academic_contour(ax3, X, Y, Z_succ, "Pathfinding Success Rate", "Success Rate", 0.0, 1.0)

    ax4 = fig.add_subplot(224)
    valid_len_mask = ~np.isnan(df['Avg Path Length'].values)
    max_len = np.nanmax(df['Avg Path Length'].values) if np.any(valid_len_mask) else 100
    min_len = np.nanmin(df['Avg Path Length'].values) if np.any(valid_len_mask) else 0
    penalty_len = max_len * 1.05
    Z_len = np.where(np.isnan(df['Avg Path Length'].values), penalty_len, df['Avg Path Length'].values)
    style_academic_contour(ax4, X, Y, Z_len, "Average Euclidean Path Length", "Path Length", min_len, penalty_len)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = f'plot3_smooth_contour_{env_name}.png'
    plt.savefig(save_path, dpi=300, facecolor='white', edgecolor='none', bbox_inches='tight')
    plt.close(fig)
    print(f"已生成图3 (局部甜点区平滑等高热力图): {save_path}")

# =====================================================================
# 4. 命令行接口 (CLI) 
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="BeamPRM 敏感度测试与多图生成")
    parser.add_argument("--action", type=str, choices=["test", "plot", "both"], default="both", 
                        help="'test':跑测试, 'plot':只画图, 'both':先跑后画(默认)")
    parser.add_argument("--env", type=str, choices=["maze", "indoor"], default="maze")
    parser.add_argument("--budget", type=int, default=1000)
    args = parser.parse_args()

    env = args.env
    csv_large = f'sensitivity_data_large_{env}.csv'
    csv_small = f'sensitivity_data_small_{env}.csv'
    
    test_angles = [1, 3, 6, 9, 12, 15]          
    
    # 第一组：大范围半径 
    test_radii_large = [0.2, 0.4, 0.6, 0.8, 1.0]  
    # 第二组：小范围半径 
    test_radii_small = [0.2, 0.25, 0.3, 0.35, 0.4]

    if args.action in ["test", "both"]:
        print(f"\n[{env.upper()}] === 阶段 1: 测试大范围全局参数 (用于图1) ===")
        tester_large = BeamPRMParameterTester(environment_type=env, num_runs=3, num_paths=50)
        tester_large.run_tests(test_angles, test_radii_large, node_budget=args.budget, save_samples=False, csv_filename=csv_large)

        print(f"\n[{env.upper()}] === 阶段 2: 测试小范围局部参数 (用于图2、图3) ===")
        tester_small = BeamPRMParameterTester(environment_type=env, num_runs=3, num_paths=50)
        tester_small.run_tests(test_angles, test_radii_small, node_budget=args.budget, save_samples=True, csv_filename=csv_small)
    
    if args.action in ["plot", "both"]:
        print(f"\n[{env.upper()}] === 开始生成学术图表 ===")
        try:
            df_large = load_data(csv_large)
            plot_standalone_success_rate(df_large, env)
            
            df_small = load_data(csv_small)
            plot_classic_grid(df_small, env)
            plot_smooth_contours(df_small, env)
            
            print("\n✅ 绘图任务全部完成！")
        except FileNotFoundError as e:
            print(f"❌ {e}")

if __name__ == "__main__":
    main()