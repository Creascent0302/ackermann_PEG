
import pygame
import numpy as np
import math
from collections import defaultdict
import sys
sys.path.append('.')
from config import ENV_CONFIG

class SunshineExplorationPRM:
    """基于阳光算法思路的地图探索点图生成器"""
    
    def __init__(self, grid_width, grid_height, obstacles, target_nodes=200):
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.obstacles = obstacles
        self.target_nodes = target_nodes
        
        # 阳光算法参数
        self.theta_step = math.pi / 24  # 角度步长：7.5度
        self.l_step = ENV_CONFIG['cell_size'] * 0.1  # 光线步长
        self.l_threshold = ENV_CONFIG['cell_size'] * 0.6  # 突变检测阈值
        self.l_forward = ENV_CONFIG['cell_size'] * 0.5  # 前向距离
        
        # 节点管理
        self.exploration_nodes = []  # 探索节点集合
        self.edges = []  # 边集合
        self.seed_points = []  # 种子点
        
        # 碰撞检测
        self.collision_radius = 2 * ENV_CONFIG['agent_collision_radius'] / 3
        self.min_node_distance = ENV_CONFIG['cell_size'] * 0.7  # 节点最小间距
        self.max_connection_distance = ENV_CONFIG['cell_size'] * 2.5  # 最大连接距离
    
    def _is_valid_position(self, x, y):
        """检查位置是否有效"""
        dr = self.collision_radius
        check_points = [
            (x, y), (x+dr, y), (x-dr, y), (x, y+dr), (x, y-dr),
            (x+dr*0.7, y+dr*0.7), (x+dr*0.7, y-dr*0.7),
            (x-dr*0.7, y+dr*0.7), (x-dr*0.7, y-dr*0.7)
        ]
        
        for gx, gy in check_points:
            gx1, gy1 = gx / ENV_CONFIG['cell_size'], gy / ENV_CONFIG['cell_size']
            if gx1 < 0 or gx1 >= self.grid_width or gy1 < 0 or gy1 >= self.grid_height:
                return False
            if (int(gx1), int(gy1)) in self.obstacles:
                return False
        return True
    
    def _generate_strategic_seeds(self):
        """生成战略性种子点：边界、空旷区域中心、通道关键点"""
        seeds = []
        cell_size = ENV_CONFIG['cell_size']
        
        # 1. 边界种子点（地图边缘的有效位置）
        print("生成边界种子点...")
        boundary_seeds = []
        
        # 四个边界
        boundaries = [
            [(i * cell_size, cell_size) for i in range(1, self.grid_width-1, 3)],  # 上边界
            [(i * cell_size, (self.grid_height-2) * cell_size) for i in range(1, self.grid_width-1, 3)],  # 下边界
            [(cell_size, j * cell_size) for j in range(1, self.grid_height-1, 3)],  # 左边界
            [((self.grid_width-2) * cell_size, j * cell_size) for j in range(1, self.grid_height-1, 3)]  # 右边界
        ]
        
        for boundary in boundaries:
            for point in boundary:
                if self._is_valid_position(point[0], point[1]):
                    boundary_seeds.append(point)
        
        seeds.extend(boundary_seeds[:min(20, len(boundary_seeds))])
        
        # # 2. 空旷区域中心点
        # print("寻找空旷区域中心点...")
        # open_area_seeds = self._find_open_area_centers()
        # seeds.extend(open_area_seeds)
        
        # # 3. 障碍物附近的关键导航点
        # print("生成障碍物附近导航点...")
        # obstacle_nav_seeds = self._generate_obstacle_navigation_points()
        # seeds.extend(obstacle_nav_seeds)
        
        self.seed_points = seeds
        print(f"生成了 {len(seeds)} 个战略种子点")
        return seeds
    
    def _find_open_area_centers(self):
        """寻找空旷区域的中心点"""
        centers = []
        cell_size = ENV_CONFIG['cell_size']
        search_radius = 3  # 搜索半径（格子数）
        
        # 在网格上采样潜在中心点
        for i in range(search_radius, self.grid_width - search_radius, search_radius * 2):
            for j in range(search_radius, self.grid_height - search_radius, search_radius * 2):
                center_x, center_y = i * cell_size, j * cell_size
                
                if not self._is_valid_position(center_x, center_y):
                    continue
                
                # 检查周围区域是否足够空旷
                obstacle_count = 0
                total_checks = 0
                
                for di in range(-search_radius, search_radius + 1):
                    for dj in range(-search_radius, search_radius + 1):
                        ni, nj = i + di, j + dj
                        if 0 <= ni < self.grid_width and 0 <= nj < self.grid_height:
                            total_checks += 1
                            if (ni, nj) in self.obstacles:
                                obstacle_count += 1
                
                # 如果障碍物密度较低，认为是空旷区域
                if obstacle_count / total_checks < 0.1:  # 10%以下障碍物密度
                    centers.append((center_x, center_y))
        
        return centers
    
    def _generate_obstacle_navigation_points(self):
        """在障碍物周围生成关键导航点"""
        nav_points = []
        cell_size = ENV_CONFIG['cell_size']
        
        # 为每个障碍物生成周围的导航点
        obstacle_groups = self._group_obstacles()
        
        for obs_group in obstacle_groups:
            # 找到障碍物群的边界
            min_x = min(obs[0] for obs in obs_group) * cell_size
            max_x = max(obs[0] for obs in obs_group) * cell_size
            min_y = min(obs[1] for obs in obs_group) * cell_size
            max_y = max(obs[1] for obs in obs_group) * cell_size
            
            # 在障碍物群周围生成导航点
            margin = cell_size * 1.5
            candidates = [
                (min_x - margin, min_y - margin),  # 左上
                (max_x + margin, min_y - margin),  # 右上
                (min_x - margin, max_y + margin),  # 左下
                (max_x + margin, max_y + margin),  # 右下
                (min_x - margin, (min_y + max_y) / 2),  # 左中
                (max_x + margin, (min_y + max_y) / 2),  # 右中
                ((min_x + max_x) / 2, min_y - margin),  # 上中
                ((min_x + max_x) / 2, max_y + margin)   # 下中
            ]
            
            for point in candidates:
                if (0 <= point[0] < self.grid_width * cell_size and 
                    0 <= point[1] < self.grid_height * cell_size and
                    self._is_valid_position(point[0], point[1])):
                    nav_points.append(point)
        
        return nav_points
    
    def _group_obstacles(self):
        """将相邻的障碍物分组"""
        visited = set()
        groups = []
        
        def dfs(obs, current_group):
            if obs in visited:
                return
            visited.add(obs)
            current_group.append(obs)
            
            # 检查8个邻居
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    neighbor = (obs[0] + dx, obs[1] + dy)
                    if neighbor in self.obstacles and neighbor not in visited:
                        dfs(neighbor, current_group)
        
        for obs in self.obstacles:
            if obs not in visited:
                current_group = []
                dfs(obs, current_group)
                if len(current_group) > 3:  # 只考虑较大的障碍物群
                    groups.append(current_group)
        
        return groups
    
    def _cast_exploration_ray(self, sun_pos, angle, max_length):
        """发射探索光线，返回光线长度"""
        x0, y0 = sun_pos
        dx = math.cos(angle)
        dy = math.sin(angle)
        
        current_length = 0
        while current_length < max_length:
            current_length += self.l_step
            x = x0 + dx * current_length
            y = y0 + dy * current_length
            
            # 边界检查
            if (x < 0 or x >= self.grid_width * ENV_CONFIG['cell_size'] or
                y < 0 or y >= self.grid_height * ENV_CONFIG['cell_size']):
                return max(0, current_length - self.l_step)
            
            # 障碍物检查
            if not self._is_valid_position(x, y):
                return max(0, current_length - self.l_step)
        
        return max_length
    
    def _discover_navigation_points(self, sun_pos):
        """从太阳位置发现导航关键点"""
        navigation_points = []
        max_ray_length = min(self.grid_width, self.grid_height) * ENV_CONFIG['cell_size'] * 0.8
        
        ray_lengths = []
        angles = []
        
        # 收集所有方向的光线长度
        angle = 0
        while angle < 2 * math.pi:
            ray_length = self._cast_exploration_ray(sun_pos, angle, max_ray_length)
            ray_lengths.append(ray_length)
            angles.append(angle)
            angle += self.theta_step
        
        # 检测长度突变点（关键导航位置）
        for i in range(len(ray_lengths)):
            prev_i = (i - 1) % len(ray_lengths)
            next_i = (i + 1) % len(ray_lengths)
            
            current_length = ray_lengths[i]
            prev_length = ray_lengths[prev_i]
            next_length = ray_lengths[next_i]
            
            # 检测突变：当前长度与相邻长度差异显著
            if (abs(current_length - prev_length) > self.l_threshold or 
                abs(current_length - next_length) > self.l_threshold):
                
                # 计算导航点位置
                nav_angle = angles[i]
                nav_distance = min(current_length, prev_length, next_length) + self.l_forward
                
                nav_x = sun_pos[0] + math.cos(nav_angle) * nav_distance
                nav_y = sun_pos[1] + math.sin(nav_angle) * nav_distance
                
                # 验证导航点有效性
                if (0 <= nav_x < self.grid_width * ENV_CONFIG['cell_size'] and
                    0 <= nav_y < self.grid_height * ENV_CONFIG['cell_size'] and
                    self._is_valid_position(nav_x, nav_y)):
                    
                    navigation_points.append((nav_x, nav_y))
        
        return navigation_points
    
    def _filter_duplicate_nodes(self, candidates):
        """过滤重复和过近的节点"""
        filtered = []
        
        for candidate in candidates:
            too_close = False
            
            # 检查与已有节点的距离
            for existing in self.exploration_nodes + filtered:
                distance = np.linalg.norm(np.array(candidate) - np.array(existing))
                if distance < self.min_node_distance:
                    too_close = True
                    break
            
            if not too_close:
                filtered.append(candidate)
        
        return filtered
    
    def _build_navigation_graph(self):
        """构建导航图：连接可视的节点"""
        print("构建导航连接...")
        self.edges = []
        
        for i, node1 in enumerate(self.exploration_nodes):
            for j, node2 in enumerate(self.exploration_nodes):
                if i >= j:
                    continue
                
                distance = np.linalg.norm(np.array(node1) - np.array(node2))
                
                # 距离过滤
                if distance > self.max_connection_distance:
                    continue
                
                # 可视性检查
                if self._is_line_clear(node1, node2):
                    self.edges.append((node1, node2))
        
        print(f"生成了 {len(self.edges)} 条连接边")
    
    def _is_line_clear(self, point1, point2):
        """检查两点间直线路径是否畅通"""
        x1, y1 = point1
        x2, y2 = point2
        
        distance = np.linalg.norm([x2 - x1, y2 - y1])
        if distance == 0:
            return True
        
        num_checks = max(5, int(distance / (ENV_CONFIG['cell_size'] / 3)))
        
        for i in range(num_checks + 1):
            t = i / num_checks
            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)
            
            if not self._is_valid_position(x, y):
                return False
        
        return True
    
    def _enhance_coverage(self):
        """增强覆盖性：在稀疏区域添加节点"""
        print("增强地图覆盖...")
        cell_size = ENV_CONFIG['cell_size']
        
        # 划分地图为网格，检查每个区域的节点密度
        grid_size = 4  # 每个检查区域的大小
        
        for i in range(0, self.grid_width, grid_size):
            for j in range(0, self.grid_height, grid_size):
                region_center = ((i + grid_size/2) * cell_size, 
                               (j + grid_size/2) * cell_size)
                
                if not self._is_valid_position(*region_center):
                    continue
                
                # 检查该区域内的节点密度
                region_radius = grid_size * cell_size
                nodes_in_region = 0
                
                for node in self.exploration_nodes:
                    if np.linalg.norm(np.array(node) - np.array(region_center)) < region_radius:
                        nodes_in_region += 1
                
                # 如果区域内节点过少，添加一个节点
                if nodes_in_region == 0 and len(self.exploration_nodes) < self.target_nodes:
                    self.exploration_nodes.append(region_center)
    
    def generate_exploration_graph(self):
        """生成完整的地图探索导航图"""
        print("开始生成地图探索导航图...")
        
        # 第一阶段：生成战略种子点
        seed_points = self._generate_strategic_seeds()
        
        # 第二阶段：从种子点发现导航关键点
        print("从种子点发现导航关键点...")
        all_candidates = []
        
        for i, seed in enumerate(seed_points):
            if i % 10 == 0:
                print(f"处理种子点 {i+1}/{len(seed_points)}")
            
            navigation_points = self._discover_navigation_points(seed)
            all_candidates.extend(navigation_points)
        
        # 添加种子点本身作为导航节点
        all_candidates.extend(seed_points)
        
        print(f"发现 {len(all_candidates)} 个候选导航点")
        
        # 第三阶段：过滤重复节点
        print("过滤重复节点...")
        self.exploration_nodes = self._filter_duplicate_nodes(all_candidates)
        print(f"过滤后保留 {len(self.exploration_nodes)} 个导航节点")
        
        # 第四阶段：增强覆盖性
        if len(self.exploration_nodes) < self.target_nodes:
            self._enhance_coverage()
            print(f"覆盖增强后共 {len(self.exploration_nodes)} 个节点")
        
        # 第五阶段：构建导航图
        self._build_navigation_graph()
        
        # 第六阶段：连通性优化
        self._optimize_connectivity()
        
        print(f"地图探索导航图生成完成！")
        print(f"最终节点数: {len(self.exploration_nodes)}")
        print(f"最终边数: {len(self.edges)}")
        
        return self.exploration_nodes, self.edges
    
    def _optimize_connectivity(self):
        """优化图的连通性"""
        print("优化图连通性...")
        
        # 找到连接度较低的节点
        node_connections = defaultdict(int)
        for edge in self.edges:
            node_connections[edge[0]] += 1
            node_connections[edge[1]] += 1
        
        # 为连接度低的节点寻找新连接
        for node in self.exploration_nodes:
            if node_connections[node] < 2:  # 连接度过低
                # 寻找最近的可连接节点
                nearest_nodes = []
                for other_node in self.exploration_nodes:
                    if other_node != node:
                        distance = np.linalg.norm(np.array(node) - np.array(other_node))
                        nearest_nodes.append((distance, other_node))
                
                nearest_nodes.sort()
                
                # 尝试连接到最近的几个节点
                connections_added = 0
                for distance, target_node in nearest_nodes:
                    if connections_added >= 2:  # 最多添加2个连接
                        break
                    
                    if (distance < self.max_connection_distance * 1.5 and 
                        self._is_line_clear(node, target_node)):
                        
                        edge = (node, target_node)
                        reverse_edge = (target_node, node)
                        
                        if edge not in self.edges and reverse_edge not in self.edges:
                            self.edges.append(edge)
                            connections_added += 1
    
    def get_graph_statistics(self):
        """获取图的统计信息"""
        node_connections = defaultdict(int)
        for edge in self.edges:
            node_connections[edge[0]] += 1
            node_connections[edge[1]] += 1
        
        avg_connections = sum(node_connections.values()) / len(self.exploration_nodes) if self.exploration_nodes else 0
        isolated_nodes = sum(1 for node in self.exploration_nodes if node_connections[node] == 0)
        
        return {
            'total_nodes': len(self.exploration_nodes),
            'total_edges': len(self.edges),
            'avg_connections': avg_connections,
            'isolated_nodes': isolated_nodes,
            'seed_points': len(self.seed_points)
        }


class SunshineExplorationRenderer:
    """地图探索导航图可视化"""
    
    def __init__(self, grid_width, grid_height, cell_size=20):
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.cell_size = cell_size
        self.screen_width = grid_width * cell_size
        self.screen_height = grid_height * cell_size
        pygame.init()
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Sunshine Exploration Navigation Graph")
        self.clock = pygame.time.Clock()

    def render(self, nodes, edges, obstacles, seed_points=None):
        """渲染探索导航图"""
        self.screen.fill((240, 248, 255))  # 淡蓝色背景

        # 绘制障碍物
        for obs in obstacles:
            x, y = obs
            rect = pygame.Rect(int(x * self.cell_size), int(y * self.cell_size), 
                             self.cell_size, self.cell_size)
            pygame.draw.rect(self.screen, (70, 70, 70), rect)

        # 绘制连接边
        for edge in edges:
            node1, node2 = edge
            x1, y1 = node1
            x2, y2 = node2
            screen_x1 = int(x1 * self.cell_size / ENV_CONFIG['cell_size'])
            screen_y1 = int(y1 * self.cell_size / ENV_CONFIG['cell_size'])
            screen_x2 = int(x2 * self.cell_size / ENV_CONFIG['cell_size'])
            screen_y2 = int(y2 * self.cell_size / ENV_CONFIG['cell_size'])
            
            pygame.draw.line(self.screen, (100, 149, 237), 
                           (screen_x1, screen_y1), (screen_x2, screen_y2), 2)

        # 绘制种子点
        if seed_points:
            for point in seed_points:
                x, y = point
                screen_x = int(x * self.cell_size / ENV_CONFIG['cell_size'])
                screen_y = int(y * self.cell_size / ENV_CONFIG['cell_size'])
                pygame.draw.circle(self.screen, (255, 140, 0), (screen_x, screen_y), 6)

        # 绘制导航节点
        for node in nodes:
            x, y = node
            screen_x = int(x * self.cell_size / ENV_CONFIG['cell_size'])
            screen_y = int(y * self.cell_size / ENV_CONFIG['cell_size'])
            pygame.draw.circle(self.screen, (34, 139, 34), (screen_x, screen_y), 4)

        pygame.display.flip()

    def run(self, exploration_prm):
        """运行可视化"""
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_s:
                        pygame.image.save(self.screen, "exploration_graph.png")
                        print("截图保存为 exploration_graph.png")
                    elif event.key == pygame.K_i:
                        stats = exploration_prm.get_graph_statistics()
                        print("\n=== 导航图统计信息 ===")
                        for key, value in stats.items():
                            print(f"{key}: {value}")
            
            self.render(exploration_prm.exploration_nodes, exploration_prm.edges, 
                       exploration_prm.obstacles, exploration_prm.seed_points)
            self.clock.tick(30)
        
        pygame.quit()


if __name__ == "__main__":
    # 配置参数
    grid_width = ENV_CONFIG['gridnum_width']
    grid_height = ENV_CONFIG['gridnum_height']
    cell_size = ENV_CONFIG['cell_size']
    
    # 生成简单的块状障碍物
    obstacles = set()
    
    # 生成随机块状障碍物
    num_blocks = 8  # 障碍物块数量
    
    for _ in range(num_blocks):
        # 随机选择块的起始位置
        start_x = np.random.randint(2, grid_width - 8)
        start_y = np.random.randint(2, grid_height - 8)
        
        # 随机块大小
        block_width = np.random.randint(1, 4)
        block_height = np.random.randint(1, 4)
        
        # 生成矩形块
        for i in range(block_width):
            for j in range(block_height):
                x = start_x + i
                y = start_y + j
                if 0 <= x < grid_width and 0 <= y < grid_height:
                    obstacles.add((x, y))
    
    # 添加一些边界障碍物（可选）
    # 左上角块
    for i in range(4):
        for j in range(4):
            obstacles.add((i, j))
    
    # 右下角块
    for i in range(grid_width - 4, grid_width):
        for j in range(grid_height - 4, grid_height):
            if 0 <= i < grid_width and 0 <= j < grid_height:
                obstacles.add((i, j))
    
    # 中央一个较大的障碍物块
    center_x, center_y = grid_width // 2, grid_height // 2
    for i in range(-3, 4):
        for j in range(-2, 3):
            x, y = center_x + i, center_y + j
            if 0 <= x < grid_width and 0 <= y < grid_height:
                obstacles.add((x, y))
    
    obstacles = list(obstacles)
    
    # 创建基于阳光算法的地图探索PRM
    exploration_prm = SunshineExplorationPRM(
        grid_width, grid_height, obstacles, 
        target_nodes=250
    )
    
    # 生成探索导航图
    nodes, edges = exploration_prm.generate_exploration_graph()
    
    # 显示统计信息
    stats = exploration_prm.get_graph_statistics()
    print("\n=== 最终导航图统计 ===")
    for key, value in stats.items():
        print(f"{key}: {value:.2f}" if isinstance(value, float) else f"{key}: {value}")
    
    # 可视化
    renderer = SunshineExplorationRenderer(grid_width, grid_height)
    print("\n按键说明:")
    print("S - 保存截图")
    print("I - 显示统计信息")
    renderer.run(exploration_prm)
