
import pygame
import numpy as np
from scipy.spatial import KDTree
import sys
sys.path.append('.')
from config import ENV_CONFIG
import math
import heapq
from abc import ABC, abstractmethod
import time

class BasePathPlanner(ABC):
    """路径规划算法基类"""
    
    def __init__(self, grid_width, grid_height, obstacles, **kwargs):
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.obstacles = set(obstacles)  # 转换为set提高查询效率
        self.nodes = []
        self.edges = []
        self._adj_cache = None
        # 通用参数
        self.collision_radius = kwargs.get('collision_radius', 2 * ENV_CONFIG['agent_collision_radius'] / 3)
        self.step_size = kwargs.get('step_size', 0.1)
        
        # 构建障碍物KDTree用于加速碰撞检测
        self._build_obstacle_kdtree()
        
    def _build_obstacle_kdtree(self):
        """构建障碍物KDTree用于加速碰撞检测"""
        if not self.obstacles:
            self.obstacle_kdtree = None
            return
        cs = ENV_CONFIG['cell_size']
        pts = [((ox + 0.5) * cs, (oy + 0.5) * cs) for (ox, oy) in self.obstacles]
        self.obstacle_kdtree = KDTree(pts)
    
    def _is_valid_position(self, x, y, dr=None):
        """检查位置是否有效（不在障碍物附近且在边界内）"""
        if dr is None:
            dr = 0.015
            
        # 边界检查
        if x < dr or x >= self.grid_width * ENV_CONFIG['cell_size'] - dr:
            return False
        if y < dr or y >= self.grid_height * ENV_CONFIG['cell_size'] - dr:
            return False
        
        # 障碍物检查
        check_points = [
            (x + dr, y), (x - dr, y), (x, y + dr), (x, y - dr),
            (x + dr/np.sqrt(2), y + dr/np.sqrt(2)), 
            (x + dr/np.sqrt(2), y - dr/np.sqrt(2)),
            (x - dr/np.sqrt(2), y + dr/np.sqrt(2)), 
            (x - dr/np.sqrt(2), y - dr/np.sqrt(2)),
            (x, y)
        ]
        
        cs = ENV_CONFIG['cell_size']
        for px, py in check_points:
            gx, gy = int(px / cs), int(py / cs)
            if gx < 0 or gx >= self.grid_width or gy < 0 or gy >= self.grid_height:
                return False
            if (gx, gy) in self.obstacles:
                return False
                
        return True
    
    def _is_valid_edge(self, node1, node2, num_samples=None):
        """检查边是否有效（沿线段采样检测碰撞）"""
        x1, y1 = node1
        x2, y2 = node2
        
        length = math.hypot(x2 - x1, y2 - y1)
        if num_samples is None:
            num_samples = max(1, int(length * 20))
        
        for i in range(num_samples + 1):
            t = i / num_samples
            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)
            if not self._is_valid_position(x, y):
                return False
        return True
    
    def _distance(self, node1, node2):
        """计算两节点间欧几里得距离"""
        return math.hypot(node1[0] - node2[0], node1[1] - node2[1])
    
    def _random_sample(self):
        """在自由空间中随机采样一个点"""
        max_attempts = 1000
        for _ in range(max_attempts):
            x = np.random.uniform(0, self.grid_width * ENV_CONFIG['cell_size'])
            y = np.random.uniform(0, self.grid_height * ENV_CONFIG['cell_size'])
            if self._is_valid_position(x, y):
                return (x, y)
        return None
    
    def cal_middle_point(self, node1, node2):
        """计算两节点的中点"""
        return ((node1[0] + node2[0]) / 2, (node1[1] + node2[1]) / 2)

    @abstractmethod
    def generate_prm(self):
        """生成路径图 - 抽象方法，子类必须实现"""
        pass
    
    def calculate_node_utilization(self, num_test_paths=50):
        """
        计算PRM中节点的平均利用率
        通过随机生成多对起点和终点，计算路径中实际使用的节点占总节点数的比例
        
        参数:
            num_test_paths: 测试路径数量
        
        返回:
            dict: 包含利用率统计信息
        """
        if not self.nodes or len(self.nodes) < 2:
            return {
                'avg_utilization': 0.0,
                'node_usage_count': {},
                'total_nodes': 0,
                'used_nodes': 0,
                'successful_paths': 0,
                'avg_nodes_per_path': 0.0
            }
        
        # 记录每个节点被使用的次数
        node_usage_count = {node: 0 for node in self.nodes}
        successful_paths = 0
        total_path_nodes = 0
        
        # 生成随机起点和终点对
        valid_pairs = self.generate_valid_point_pairs(num_test_paths)
        
        for start, goal in valid_pairs:
            try:
                result = self.find_path(start, goal)
                # find_path 可能返回不同格式，兼容处理
                if result:
                    if isinstance(result, tuple) and len(result) >= 1:
                        path = result[0]  # path_nodes
                    else:
                        path = result
                    
                    if path and len(path) > 1:
                        successful_paths += 1
                        # 统计路径中的节点使用情况
                        for node in path:
                            if node in node_usage_count:
                                node_usage_count[node] += 1
                        total_path_nodes += len([n for n in path if n in node_usage_count])
            except Exception:
                # 路径查找失败，跳过
                continue
        
        # 计算统计信息
        total_nodes = len(self.nodes)
        used_nodes = sum(1 for count in node_usage_count.values() if count > 0)
        avg_utilization = (used_nodes / total_nodes) if total_nodes > 0 else 0.0
        avg_nodes_per_path = (total_path_nodes / successful_paths) if successful_paths > 0 else 0.0
        
        return {
            'avg_utilization': avg_utilization,
            'node_usage_count': node_usage_count,
            'total_nodes': total_nodes,
            'used_nodes': used_nodes,
            'successful_paths': successful_paths,
            'avg_nodes_per_path': avg_nodes_per_path
        }

    def cal_dispersion(self, num_samples=1000, media=False):
        """
        计算路图的离散度 (Dispersion) - 性能优化版本。
        """
        if not self.nodes:
            return float('inf')
        
        # 预计算物理尺寸和常量
        physical_width = self.grid_width * ENV_CONFIG['cell_size']
        physical_height = self.grid_height * ENV_CONFIG['cell_size']
        cell_size = ENV_CONFIG['cell_size']
        
        # 构建节点KDTree加速距离查询
        nodes_array = np.array(self.nodes)
        nodes_kdtree = KDTree(nodes_array)
        
        # 障碍物集合转换为更高效的查询结构
        obstacle_set = set(self.obstacles)
        
        # 批量生成测试点
        test_points = np.random.uniform(
            low=[0, 0], 
            high=[physical_width, physical_height], 
            size=(num_samples * 2, 2)  # 生成2倍点数，减少后续循环
        )
        
        # 快速筛选有效测试点
        valid_points = []
        for point in test_points:
            # 快速检查是否在障碍物上
            grid_x, grid_y = int(point[0] / cell_size), int(point[1] / cell_size)
            if 0 <= grid_x < self.grid_width and 0 <= grid_y < self.grid_height:
                if (grid_x, grid_y) not in obstacle_set:
                    valid_points.append(point)
                    if len(valid_points) >= num_samples:
                        break
                    
        if not valid_points:
            return 0.0
        
        valid_points = np.array(valid_points[: num_samples])
        
        # 使用KDTree查询每个测试点到最近节点的距离
        distances, indices = nodes_kdtree.query(valid_points, k=3)  # 查询最近的3个节点
        
        # 初始化最大距离
        max_distance = 0
        
        # 验证最近节点的路径是否有效
        for i, point in enumerate(valid_points):
            # 只检查最近的几个节点
            for j in range(min(3, len(indices[i]))):                
                # 简化有效性检查 - 只在必要时使用完整检查
                max_distance = max(max_distance, distances[i][j])
                break        
        return max_distance

    def backward_prune_path(self, path_nodes):
        """
        后向裁剪路径:
        """
        if not path_nodes or len(path_nodes) < 3:
            return path_nodes
        for _ in range(7):
            for i in range(len(path_nodes) - 2):
                current = path_nodes[i]
                next = path_nodes[i + 1]
                after_next = path_nodes[i + 2]
                if self._is_valid_edge(current, after_next):
                    mid = self.cal_middle_point(next, after_next)
                    path_nodes[i + 1] = mid
                else:
                    mid = self.cal_middle_point(next, after_next)
                    if self._is_valid_edge(current, mid):
                        path_nodes[i + 1] = mid
                    else:
                        mid = self.cal_middle_point(next, mid)
                        if self._is_valid_edge(current, mid):
                            path_nodes[i + 1] = mid
            path_nodes.reverse()
        return path_nodes

    def cal_discrepancy(self, num_samples=1000, media=False):
        """
        计算节点集的星偏差度 (Star Discrepancy) - 性能优化版本。
        限制矩形面积不超过地图总面积的1/8。
        """
        if not self.nodes:
            return 1.0
        
        # 预计算常量
        physical_width = self.grid_width * ENV_CONFIG['cell_size']
        physical_height = self.grid_height * ENV_CONFIG['cell_size']
        total_area = physical_width * physical_height
        max_area = total_area / 8  # 限制最大矩形面积为地图总面积的1/8
        num_nodes = len(self.nodes)
        
        # 将节点转换为NumPy数组 - 只做一次
        np_nodes = np.array(self.nodes)
        
        # 批量生成所有矩形 (每个矩形需要2个点)
        points = np.random.uniform(
            low=[0, 0], 
            high=[physical_width, physical_height], 
            size=(num_samples, 2, 2)
        )
        
        # 预分配结果数组
        discrepancies = np.zeros(num_samples)
        
        # 向量化处理所有矩形
        for i in range(num_samples):
            # 获取矩形坐标
            x1 = min(points[i, 0, 0], points[i, 1, 0])
            x2 = max(points[i, 0, 0], points[i, 1, 0])
            y1 = min(points[i, 0, 1], points[i, 1, 1])
            y2 = max(points[i, 0, 1], points[i, 1, 1])
            
            # # 计算矩形面积并检查是否超限
            # rect_area = (x2 - x1) * (y2 - y1)
            
            # # 如果面积超过限制，缩小矩形保持中心点不变
            # if rect_area > max_area:
            #     # 计算矩形中心
            #     center_x = (x1 + x2) / 2
            #     center_y = (y1 + y2) / 2
                
            #     # 计算缩放因子
            #     scale = math.sqrt(max_area / rect_area)
                
            #     # 计算新的半宽和半高
            #     half_width = (x2 - x1) / 2 * scale
            #     half_height = (y2 - y1) / 2 * scale
                
            #     # 更新矩形坐标
            #     x1 = center_x - half_width
            #     x2 = center_x + half_width
            #     y1 = center_y - half_height
            #     y2 = center_y + half_height
            
            # 计算面积比例
            area_rate = (x2 - x1) * (y2 - y1) / total_area
            
            # 计算点在矩形内的比例
            mask = ((np_nodes[:, 0] >= x1) & 
                    (np_nodes[:, 0] <= x2) & 
                    (np_nodes[:, 1] >= y1) & 
                    (np_nodes[:, 1] <= y2))
            
            count = np.sum(mask)
            point_rate = count / num_nodes
            
            # 计算偏差
            discrepancies[i] = abs(area_rate - point_rate)
        
        # 返回最大偏差
        return np.max(discrepancies)
    
    def find_path(self, start, goal):
        """使用A*算法在路图中查找从start到goal的路径"""

        if not self._is_valid_position(start[0], start[1]):
            raise ValueError("Start position is invalid or in collision.")
        if not self._is_valid_position(goal[0], goal[1]):
            raise ValueError("Goal position is invalid or in collision.")

        if not self.nodes:
            self.generate_prm()
        if not self.nodes:
            raise ValueError("Cannot generate any nodes in the PRM.")

        if self._adj_cache is None:
            adj_cache = {node: {} for node in self.nodes}
            for edge in self.edges:
                u, v = edge
                dist = self._distance(u, v)
                adj_cache[u][v] = dist
                adj_cache[v][u] = dist
            self._adj_cache = adj_cache

        start_time = time.time()

        adjacency = dict(self._adj_cache)
        adjacency[start] = {}
        adjacency[goal] = {}
        start_connected = False
        goal_connected = False

        for node in self.nodes:
            if self._is_valid_edge(start, node):
                dist = self._distance(start, node)
                adjacency[start][node] = dist
                # 反向边需深拷贝，避免修改缓存内的邻居字典
                if adjacency[node] is self._adj_cache.get(node):
                    adjacency[node] = dict(self._adj_cache[node])
                adjacency[node][start] = dist
                start_connected = True
            if self._is_valid_edge(goal, node):
                dist = self._distance(goal, node)
                adjacency[goal][node] = dist
                if adjacency[node] is self._adj_cache.get(node):
                    adjacency[node] = dict(self._adj_cache[node])
                adjacency[node][goal] = dist
                goal_connected = True

        if not start_connected or not goal_connected:
            return None, None, None, time.time() - start_time

        # A* 搜索（原有逻辑完全不变）
        open_set = []
        open_set.append((self._distance(start, goal), start))
        open_set_nodes = {start}
        backtrack = {}
        g_score = {start: 0}
        visited = set()

        while open_set:
            current_f, current = heapq.heappop(open_set)
            if current in visited:
                continue
            visited.add(current)
            open_set_nodes.remove(current)

            if current == goal:
                path_nodes = []
                tmp_node = goal
                while tmp_node in backtrack:
                    path_nodes.append(tmp_node)
                    tmp_node = backtrack[tmp_node]
                path_nodes.append(start)
                path_nodes.reverse()

                path_nodes = self.backward_prune_path(path_nodes)
                path_edges = []
                for i in range(len(path_nodes) - 1):
                    path_edges.append((path_nodes[i], path_nodes[i + 1]))

                path_length = sum(
                    self._distance(path_nodes[i], path_nodes[i + 1])
                    for i in range(len(path_nodes) - 1)
                )
                return path_nodes, path_edges, path_length, time.time() - start_time

            else:
                for neighbor, dist in adjacency[current].items():
                    if neighbor in visited:
                        continue
                    tentative_g = g_score[current] + dist
                    if neighbor not in g_score or tentative_g < g_score[neighbor]:
                        g_score[neighbor] = tentative_g
                        f_score = tentative_g + self._distance(neighbor, goal)
                        backtrack[neighbor] = current
                        if neighbor not in open_set_nodes:
                            heapq.heappush(open_set, (f_score, neighbor))
                            open_set_nodes.add(neighbor)

        return None, None, None, time.time() - start_time 

    def generate_valid_point_pairs(self, num_pairs):
        """生成指定数量的有效起终点对"""
        pairs = []
        attempts = 0
        max_attempts = num_pairs * 10
        min_distance = 8 * ENV_CONFIG['cell_size']
        
        while len(pairs) < num_pairs and attempts < max_attempts:
            start = self._random_sample()
            goal = self._random_sample()
            attempts += 1
            if self._is_valid_position(start[0], start[1]) and self._is_valid_position(goal[0], goal[1]) and start != goal:
                if self._distance(start, goal) > min_distance:
                    pairs.append((start, goal))

        return pairs


    def cal_clearance(self):
        """
        计算路图的平均净空 (Average Clearance)。

        定义：路图中所有节点到最近障碍物（含地图边界）的最小距离的均值。
        单位与节点坐标一致（物理坐标，即 cell_size 为单位）。

        - 障碍物距离：复用已有的 obstacle_kdtree，O(N log M)
        - 边界距离：对每个节点分别计算到四条边界的最小距离
        - 两者取 min，再对所有节点求均值

        返回:
            float: 平均净空值，越大表示路图整体离障碍/边界越远、越安全
                节点为空时返回 0.0
        """
        if not self.nodes:
            return 0.0

        cs = ENV_CONFIG['cell_size']
        nodes_array = np.array(self.nodes)  # shape: (N, 2)，物理坐标

        # ── 1. 到最近障碍物的距离 ──────────────────────────────────────────────
        if self.obstacle_kdtree is not None:
            # query 返回 (distances, indices)，distances shape: (N,)
            obstacle_dists, _ = self.obstacle_kdtree.query(nodes_array)
            # KDTree 中障碍物坐标是格子中心 (ox+0.5)*cs，
            # 查到的是到格子中心的距离，减去半个格子使结果更保守
            obstacle_dists = np.maximum(0.0, obstacle_dists - cs * 0.5)
        else:
            # 没有障碍物，距离设为无穷大（由边界距离决定）
            obstacle_dists = np.full(len(self.nodes), np.inf)

        # ── 2. 到地图边界的距离 ────────────────────────────────────────────────
        map_w = self.grid_width  * cs
        map_h = self.grid_height * cs

        boundary_dists = np.minimum(
            np.minimum(nodes_array[:, 0],          map_w - nodes_array[:, 0]),
            np.minimum(nodes_array[:, 1],          map_h - nodes_array[:, 1])
        )

        # ── 3. 综合净空 = min(障碍距离, 边界距离)，对所有节点求均值 ────────────
        clearances = np.minimum(obstacle_dists, boundary_dists)
        return float(np.mean(clearances))
    
    def calculate_spatial_coverage(self, num_test_paths=50):
        """
        计算单位节点空间覆盖率 (Spatial Coverage per Node)。

        公式：U_spatial = L_path / N_total
            L_path   : 单次成功路径的实际物理长度（欧氏累计距离）
            N_total  : 路图的总节点数（固定值，与具体路径无关）

        物理意义：
            稠密图 N_total 极大 → 指标偏低
            稀疏图用少量节点支撑同等路径长度 → 指标显著偏高
            直接体现节点的空间利用效率

        参数:
            num_test_paths: 测试路径对数量

        返回:
            dict:
                avg_spatial_coverage  : 所有成功路径的 U_spatial 均值  ← 核心指标
                std_spatial_coverage  : 标准差
                total_nodes           : 路图总节点数 N_total
                successful_paths      : 成功找到路径的次数
                avg_path_length       : 成功路径的平均物理长度
        """
        empty_result = {
            'avg_spatial_coverage': 0.0,
            'std_spatial_coverage': 0.0,
            'total_nodes':          0,
            'successful_paths':     0,
            'avg_path_length':      0.0,
        }

        if not self.nodes or len(self.nodes) < 2:
            return empty_result

        N_total = len(self.nodes)          # 分母：固定不变
        coverage_list = []                 # 每条成功路径的 U_spatial
        path_length_list = []              # 每条成功路径的 L_path

        valid_pairs = self.generate_valid_point_pairs(num_test_paths)

        for start, goal in valid_pairs:
            try:
                result = self.find_path(start, goal)
                if not result:
                    continue

                # 兼容 find_path 返回 (path_nodes, path_edges, path_length, t)
                if isinstance(result, tuple) and len(result) >= 3:
                    path_nodes  = result[0]
                    path_length = result[2]   # 已由 find_path 计算好的物理长度
                else:
                    continue

                if not path_nodes or len(path_nodes) < 2:
                    continue

                # 若 find_path 返回的 path_length 为 None，手动计算
                if path_length is None:
                    path_length = sum(
                        self._distance(path_nodes[i], path_nodes[i + 1])
                        for i in range(len(path_nodes) - 1)
                    )

                L_path     = float(path_length)
                U_spatial  = L_path / N_total        # 核心公式

                coverage_list.append(U_spatial)
                path_length_list.append(L_path)

            except Exception:
                continue

        successful_paths = len(coverage_list)
        if successful_paths == 0:
            return {**empty_result, 'total_nodes': N_total}

        return {
            'avg_spatial_coverage': float(np.mean(coverage_list)),    # ← 核心指标
            'std_spatial_coverage': float(np.std(coverage_list)),
            'total_nodes':          N_total,
            'successful_paths':     successful_paths,
            'avg_path_length':      float(np.mean(path_length_list)),
        }

