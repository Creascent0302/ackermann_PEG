import pygame
import numpy as np
from scipy.spatial import KDTree
import sys
sys.path.append('.')
from config import ENV_CONFIG
import math
import heapq
from map_generator import generate_indoor_obstacles, generate_maze_obstacles
from base_generator import BasePathPlanner
import time
import os
from collections import defaultdict
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
import random

from scipy.spatial import Delaunay
from scipy.signal import convolve2d
import matplotlib.pyplot as plt

class BeamPRM(BasePathPlanner):
    """概率路图生成器（精简：仅节点 + 边，去除守卫/连接器分类）"""
    def __init__(self, grid_width, grid_height, obstacles,
                num_nodes=100,
                connection_radius=0.8,
                init_seed_count=15,
                beam_angle_step_deg=3,    # 新增: 光束角度分辨率(°)
                beam_ray_step=0.08,       # 新增: 光束射线前进步长(米)
                min_connection_radius=0.4
                ):
        super().__init__(grid_width, grid_height, obstacles)
        self.num_nodes = num_nodes
        self.connection_radius = connection_radius
        self.init_seed_count = init_seed_count
        self.num_nodes = num_nodes
        self.connection_radius = connection_radius
        self.min_connection_radius = min_connection_radius
        self.nodes = []          # 统一节点列表
        self.edges = []
        self.collision_radius = 2 * ENV_CONFIG['agent_collision_radius'] / 3
        self.init_seed_count = init_seed_count
        # 新增: 障碍物KDTree和节点清距/中轴集合
        self._build_obstacle_kdtree()
        self.node_clearance = {}         # {node: clearance}
        self.medial_axis_nodes = set()   # 近似中轴节点集合
        self.medial_axis_edges = set()  # 新增: 中轴骨架边集合(无向, 存放排序后的tuple)
        self.medial_axis_paths = []  # 新增: 中轴骨架路径列表(每条为节点序列)
        self.node_explore_cones = {}  # 新增: 节点允许探索角域列表 [(start,end)], 角度制
        self.medial_axis_all_nodes = set()  # 新增: 中轴全集(包含简化后边端点)

        # --- Beam 采样关键参数(集中管理) ---
        self.beam_angle_step_deg = beam_angle_step_deg
        self.beam_ray_step = beam_ray_step

    def _compute_clearance(self, node):
        """
        计算节点到“最近碰撞实体”的距离：
        - 障碍物：取到最近障碍物中心距离减半格(近似到方格边界)，若结果<0置0
        - 边界：到四个外边界的最小距离
        返回 min(障碍物距离, 边界距离)
        """
        cs = ENV_CONFIG['cell_size']
        x, y = node
        # 边界距离
        width_m = self.grid_width * cs
        height_m = self.grid_height * cs
        boundary_dist = min(x, y, width_m - x, height_m - y)

        # 障碍物距离
        if self.obstacle_kdtree is None:
            obstacle_dist = float('inf')
        else:
            dist_center, _ = self.obstacle_kdtree.query(np.array(node))
            # 近似从中心到格子边界的距离
            obstacle_dist = max(dist_center - 0.5 * cs, 0.0)

        return min(obstacle_dist, boundary_dist)

    def check_near_radius(self, position):
        """
        若与现有任一节点距离小于最小连接半径则判定为过近，返回 True（表示应放弃该采样）。
        返回 False 表示距离足够可以继续后续验证。
        """
        if position[0] is None:
            return True
        px, py = position
        pos_vec = np.array([px, py])
        for n in self.nodes:
            if np.linalg.norm(pos_vec - np.array(n)) < self.min_connection_radius:
                return True
        return False

    def _build_node_kdtree(self):
        """构建节点的 KD 树，加速近邻搜索。"""
        if not self.nodes:
            self.node_kdtree = None
            return
        self.node_kdtree = KDTree(np.array(self.nodes))

    def check_near_radius_fast(self, position):
        """使用 KD 树加速最近邻节点距离检查。"""
        if position[0] is None:
            return True
    
        if not hasattr(self, 'node_kdtree') or self.node_kdtree is None or len(self.nodes) == 0:
            return self.check_near_radius(position)  # 回退到原方法
        
        # 使用 KD 树查询
        dist, _ = self.node_kdtree.query(np.array(position), k=1)
        return dist < self.min_connection_radius

    def _bearing_deg(self, a, b):
        """从点 a 指向点 b 的方位角(0-360)."""
        dx = b[0] - a[0]
        dy = b[1] - a[1]
        ang = math.degrees(math.atan2(dy, dx)) % 360.0
        return ang

    def _merge_intervals(self, intervals):
        """合并已按 start 排序的不跨 0 的区间列表。"""
        if not intervals:
            return []
        intervals.sort(key=lambda x: x[0])
        merged = [list(intervals[0])]
        for s, e in intervals[1:]:
            ps, pe = merged[-1]
            if s <= pe + 1e-6:  # 可合并
                merged[-1][1] = max(pe, e)
            else:
                merged.append([s, e])
        # 压缩成 (0,360) 若覆盖全向
        total = sum(e - s for s, e in merged)
        if total >= 360 - 1e-3:
            return [[0.0, 360.0]]
        return [(s, e) for s, e in merged]

    def _remove_explore_sector(self, node, center_deg, width_deg=90.0):
        """从 node 的允许探索角域中删除以 center_deg 为中心、宽 width_deg 的扇区(支持跨0)."""
        if width_deg <= 0:
            return
        if node not in self.node_explore_cones or not self.node_explore_cones[node]:
            # 若未初始化，视为全向
            self.node_explore_cones[node] = [(0.0, 360.0)]
        intervals = self.node_explore_cones[node]
        if intervals == [(0.0, 360.0)] and width_deg >= 360 - 1e-6:
            self.node_explore_cones[node] = []
            return

        half = width_deg / 2.0
        s = (center_deg - half) % 360.0
        e = (center_deg + half) % 360.0
        del_spans = []
        if s <= e:
            del_spans.append((s, e))
        else:
            del_spans.append((s, 360.0))
            del_spans.append((0.0, e))

        kept = []
        for (a, b) in intervals:
            segs = [(a, b)]
            for ds, de in del_spans:
                next_segs = []
                for x, y in segs:
                    if de <= x or ds >= y:            # 无重叠
                        next_segs.append((x, y))
                    else:
                        # 有重叠四种裁剪
                        if ds <= x and de >= y:       # 全覆盖 -> 删除
                            continue
                        if ds <= x < de < y:          # 覆盖左端
                            next_segs.append((de, y))
                        elif x < ds < y <= de:        # 覆盖右端
                            next_segs.append((x, ds))
                        elif x < ds and de < y:       # 中间挖空
                            next_segs.append((x, ds))
                            next_segs.append((de, y))
                segs = next_segs
            kept.extend(segs)

        self.node_explore_cones[node] = self._merge_intervals(kept)

    def _bresenham_line(self, x0, y0, x1, y1):
        """Bresenham 算法生成从 (x0,y0) 到 (x1,y1) 的格子坐标列表"""
        cs = ENV_CONFIG['cell_size']
        gx0, gy0 = int(x0 / cs), int(y0 / cs)
        gx1, gy1 = int(x1 / cs), int(y1 / cs)
        
        dx = abs(gx1 - gx0)
        dy = abs(gy1 - gy0)
        sx = 1 if gx0 < gx1 else -1
        sy = 1 if gy0 < gy1 else -1
        err = dx - dy
        
        points = []
        x, y = gx0, gy0
        
        while True:
            points.append((x, y))
            if x == gx1 and y == gy1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
        
        return points
    
    def _fast_raycast(self, ex, ey, theta, max_range, ray_step):
        """优化的 raycasting：使用 Bresenham 算法快速检测碰撞"""
        dx, dy = math.cos(theta), math.sin(theta)
        end_x = ex + max_range * dx
        end_y = ey + max_range * dy
        
        # 使用 Bresenham 获取路径上的格子
        grid_points = self._bresenham_line(ex, ey, end_x, end_y)
        
        cs = ENV_CONFIG['cell_size']
        dr = 0.015
        last_valid = 0.0
        
        for i, (gx, gy) in enumerate(grid_points):
            # 检查边界
            if gx < 0 or gx >= self.grid_width or gy < 0 or gy >= self.grid_height:
                break
            # 检查障碍物
            if (gx, gy) in self.obstacles:
                break
            
            # 计算实际距离
            px = (gx + 0.5) * cs
            py = (gy + 0.5) * cs
            last_valid = math.hypot(px - ex, py - ey)
        
        return last_valid

    def _get_allowed_angles(self, node, step_deg=2):
        """返回该节点允许的角度数组（整数或浮点）。"""
        intervals = self.node_explore_cones.get(node)
        if not intervals:
            # 无记录则默认全向
            return np.arange(0, 360, step_deg)
        if len(intervals) == 1 and abs(intervals[0][0]) < 1e-6 and abs(intervals[0][1]-360) < 1e-6:
            return np.arange(0, 360, step_deg)
        angles = []
        for s, e in intervals:
            # 包含终点向下取整避免重复
            a_start = int(round(s / step_deg)) * step_deg
            if a_start < s - 1e-6:
                a_start += step_deg
            a = a_start
            while True:
                if a > e + 1e-6:
                    break
                angles.append(a % 360)
                a += step_deg
        if not angles:
            return np.arange(0, 360, step_deg)
        return np.array(sorted(set(angles)))

    def beam_sampling(self, explore_node):
        """
        从 explore_node 全向发射光束:
        使用实例参数:
        beam_angle_step_deg: 光束角度步长
        beam_ray_step: 沿光束方向离散检测步长
        优化: 使用 Bresenham 算法加速 raycasting
        """
        ex, ey = explore_node
        max_range = math.hypot(self.grid_width * ENV_CONFIG['cell_size'],
                            self.grid_height * ENV_CONFIG['cell_size'])
        ray_step = self.beam_ray_step
        angles = self._get_allowed_angles(explore_node, step_deg=self.beam_angle_step_deg)
        
        # 优化: 向量化计算所有角度
        distances = []
        for ang in angles:
            theta = math.radians(ang)
            # 使用优化的 raycasting
            last_valid = self._fast_raycast(ex, ey, theta, max_range, ray_step)
            distances.append(last_valid)

        new_nodes = []
        seen = set()
        n = len(distances)
        if n == 0:
            return new_nodes
        for i in range(n):
            j = (i + 1) % n
            d1, d2 = distances[i], distances[j]
            short, long = (d1, d2) if d1 <= d2 else (d2, d1)
            if long <= 0:
                continue
            threshold = min(short / 5.0, 0.2)
            if long - short > threshold:
                # 长光束候选
                long_idx = i if distances[i] == long else j
                theta_long = math.radians(angles[long_idx])
                sample_dist_long = (short + long) / 2.0
                sx_long = ex + sample_dist_long * math.cos(theta_long)
                sy_long = ey + sample_dist_long * math.sin(theta_long)
                key_long = (round(sx_long, 2), round(sy_long, 2))
                if key_long not in seen:
                    seen.add(key_long)
                    new_nodes.append(key_long)
                # 短光束候选（短方向一半位置）
                if short > 0:
                    short_idx = i if distances[i] == short else j
                    theta_short = math.radians(angles[short_idx])
                    sample_dist_short = short / 2.0
                    sx_short = ex + sample_dist_short * math.cos(theta_short)
                    sy_short = ey + sample_dist_short * math.sin(theta_short)
                    key_short = (round(sx_short, 2), round(sy_short, 2))
                    if key_short not in seen:
                        seen.add(key_short)
                        new_nodes.append(key_short)
        return new_nodes  # 可能为空列表

    def batch_edge_check(self, candidates, existing_nodes):
        """批量边有效性检查，可用于优化连接操作。"""
        valid_edges = []
        # 预先计算所有候选点到现有节点的距离矩阵
        candidates_array = np.array(candidates)
        nodes_array = np.array(existing_nodes)
    
        # 使用广播计算距离矩阵
        dists = np.sqrt(np.sum((candidates_array[:, np.newaxis, :] - nodes_array[np.newaxis, :, :]) ** 2, axis=2))
    
        # 对每个候选点
        for i, candidate in enumerate(candidates):
            # 找出距离符合要求的节点
            valid_indices = np.where((dists[i] <= self.connection_radius) &
                                    (dists[i] > self.min_connection_radius))[0]
        
            # 对这些节点检查边有效性
            for idx in valid_indices:
                other = existing_nodes[idx]
                if self._is_valid_edge(candidate, other):
                    valid_edges.append((candidate, other))
                
        return valid_edges

    def generate_nodes(self):
        """
        批量光束采样 + 探索队列（多源起始）：
        - 先生成多个起始种子（不互相过近）
        - 将全部种子压入 frontier
        - 之后循环从 frontier 选度数最低节点扩展
        """
        self.nodes, self.edges = [], []
        self.node_clearance = {}
        frontier = []
        seed_attempt = 0
        while len(frontier) < self.init_seed_count and len(self.nodes) < self.num_nodes:
            seed_attempt += 1
            if seed_attempt > self.init_seed_count * 200:
                print("起始种子生成受限，提前停止。")
                break
            x = np.random.uniform(0, self.grid_width * ENV_CONFIG['cell_size'])
            y = np.random.uniform(0, self.grid_height * ENV_CONFIG['cell_size'])
            candidate = (x, y)
            if not self._is_valid_position(*candidate):
                continue
            # === 新增: 十字方向包围裁剪 ===
            if self._is_cardinal_enclosed(x, y):
                continue
            if self.check_near_radius(candidate):
                continue
            # 加入节点
            self.nodes.append(candidate)
            self.node_clearance[candidate] = self._compute_clearance(candidate)
            frontier.append(candidate)
            # 初始全向
            self.node_explore_cones[candidate] = [(0.0, 360.0)]
            # 连接已有种子并删除方向扇区
            for other in self.nodes[:-1]:
                dist = np.linalg.norm(np.array(candidate) - np.array(other))
                if dist <= self.connection_radius and dist > self.min_connection_radius and self._is_valid_edge(candidate, other):
                    if (candidate, other) not in self.edges and (other, candidate) not in self.edges:
                        self.edges.append((candidate, other))
                        ang_c = self._bearing_deg(candidate, other)
                        ang_o = self._bearing_deg(other, candidate)
                        self._remove_explore_sector(candidate, ang_c, 150)
                        self._remove_explore_sector(other, ang_o, 150)

        if not frontier:
            print("未能生成任何起始种子，终止。")
            return

        # 2. 探索循环
        max_idle_expansions = self.num_nodes * 3
        idle_expansions = 0
        while len(self.nodes) < self.num_nodes and frontier and idle_expansions < max_idle_expansions:
            explore_node = self.select_explore_node(frontier)
            if explore_node is None:
                break
            if len(self.nodes) % 10 == 0:
                print(f"探索节点 {explore_node} | 已生成 {len(self.nodes)}/{self.num_nodes}")
            candidates = self.beam_sampling(explore_node)

            added_this_round = 0
            for candidate in candidates:
                if len(self.nodes) >= self.num_nodes:
                    break
                if self.check_near_radius(candidate):
                    continue
                if not self._is_valid_position(*candidate):
                    continue
                # === 新增: 十字方向包围裁剪 ===
                if self._is_cardinal_enclosed(candidate[0], candidate[1]):
                    continue
                self.nodes.append(candidate)
                self.node_clearance[candidate] = self._compute_clearance(candidate)
                frontier.append(candidate)
                # 新节点初始全向
                self.node_explore_cones[candidate] = [(0.0, 360.0)]
                added_this_round += 1
                for other in self.nodes[:-1]:
                    dist = np.linalg.norm(np.array(candidate) - np.array(other))
                    if dist <= self.connection_radius and dist > self.min_connection_radius and self._is_valid_edge(candidate, other):
                        if (candidate, other) not in self.edges and (other, candidate) not in self.edges:
                            self.edges.append((candidate, other))
                            ang_c = self._bearing_deg(candidate, other)
                            ang_o = self._bearing_deg(other, candidate)
                            self._remove_explore_sector(candidate, ang_c, 90)
                            self._remove_explore_sector(other, ang_o, 90)
            idle_expansions = idle_expansions + 1 if added_this_round == 0 else 0

        if len(self.nodes) < self.num_nodes and not frontier:
            print("frontier 为空，无法继续扩展。")
        if idle_expansions >= max_idle_expansions:
            print("多次探索无新增节点，提前终止。")
        # 结束
        return

    def _compute_degrees(self):
        """计算当前图中各节点度数"""
        deg = {n: 0 for n in self.nodes}
        for a, b in self.edges:
            if a in deg: deg[a] += 1
            if b in deg: deg[b] += 1
        return deg

    def select_explore_node(self, frontier):
        """
        从 frontier 中选择下一个待探索节点：
        优先度数（连接数）较少的节点；若度数相同随机打破平局。
        """
        if not frontier:
            return None
        deg = self._compute_degrees()
        # 通过添加随机扰动做次级排序，避免总是选择同一节点
        #chosen = min(frontier, key=lambda n: (deg.get(n, 0), np.random.random()))
        chosen = min(frontier, key=lambda n: (deg.get(n, 0)))
        frontier.remove(chosen)
        return chosen

    def connect_edges(self):
        """重建所有边并按方向删除探索角域。"""
        self.edges = []
        for i, a in enumerate(self.nodes):
            for j, b in enumerate(self.nodes):
                if i >= j:
                    continue
                dist = np.linalg.norm(np.array(a) - np.array(b))
                if dist <= self.connection_radius * 0.8 and self._is_valid_edge(a, b):
                    self.edges.append((a, b))
                    if a not in self.node_explore_cones:
                        self.node_explore_cones[a] = [(0.0, 360.0)]
                    if b not in self.node_explore_cones:
                        self.node_explore_cones[b] = [(0.0, 360.0)]
                    ang_ab = self._bearing_deg(a, b)
                    ang_ba = self._bearing_deg(b, a)
                    self._remove_explore_sector(a, ang_ab, 120)
                    self._remove_explore_sector(b, ang_ba, 120)

    def generate_prm(self):
        # 导入优化库(仅在需要时)
        import time
    
        self.generate_nodes()
        time_a = time.time()
        self.prune_graph_components()  # 新增: 保留最大连通子图并去除孤立节点
        self.identify_medial_axis()
        self.connect_medial_shortest_paths(threshold=4.0)
        self.simplify_medial_axis()
        self._finalize_medial_axis()  # 新增: 统一收集所有中轴端点
        time_b = time.time()
        print(f"minus time: {time_b - time_a:.2f} 秒")
        return (self.nodes,
                self.edges,
                self.medial_axis_nodes,
                self.medial_axis_all_nodes,   # 新增返回项
                self.medial_axis_edges,
                self.medial_axis_paths)

    def _finalize_medial_axis(self):
        """
        汇总中轴节点全集:
        medial_axis_all_nodes = 显式中轴节点 ∪ 所有中轴边端点
        确保调用方可一次性获取构成骨架的全部节点。
        """
        all_nodes = set(self.medial_axis_nodes)
        for a, b in self.medial_axis_edges:
            all_nodes.add(a); all_nodes.add(b)
        self.medial_axis_all_nodes = all_nodes

    def _has_line_of_sight(self, node1, node2):
        """判断两点之间是否无遮挡（复用边有效性判定）。"""
        distance_constraint = 1.5
        x1, y1 = node1
        x2, y2 = node2
        length = math.hypot(x2 - x1, y2 - y1)
        if length > distance_constraint:
            return False
        num_points = max(1, int(length * 20))  # 采样密度
        for i in range(num_points + 1):
            t = i / num_points
            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)
            if not self._is_valid_position(x, y,dr=0.015):
                return False
        return True

    def identify_medial_axis(self):
        """
        优化策略:
        1. 将所有节点按 clearance 从大到小排序
        2. 选取第一个作为中轴
        3. 依次尝试剩余节点: 使用 KD-Tree 只检查局部邻域内的可见性
           若在局部邻域内与已选节点可视则跳过，否则加入中轴集合
        4. 构建中轴边: 所有成对可视的中轴节点间的连接
        
        复杂度优化: O(N²) -> O(N log N)
        """
        if not self.nodes:
            self.medial_axis_nodes = set()
            self.medial_axis_edges = set()
            return

        # 全局排序 (clearance 降序, 次级按 x,y 稳定)
        ordered = sorted(
            self.nodes,
            key=lambda n: (self.node_clearance.get(n, 0.0), -n[0], -n[1]),
            reverse=True
        )

        selected = []
        # 优化: 使用自适应的搜索半径 (基于地图尺寸)
        cs = ENV_CONFIG['cell_size']
        search_radius = min(self.grid_width, self.grid_height) * cs * 0.3  # 局部邻域为地图尺寸的 30%
        
        for node in ordered:
            # 优化: 只检查局部邻域内的已选节点
            if selected:
                # 构建已选节点的 KD-Tree
                selected_kdtree = KDTree(selected)
                # 查找局部邻域内的节点
                indices = selected_kdtree.query_ball_point(node, search_radius)
                # 只检查局部邻域内的可见性
                is_redundant = False
                for idx in indices:
                    if self._has_line_of_sight(node, selected[idx]):
                        is_redundant = True
                        break
                if is_redundant:
                    continue
            selected.append(node)

        self.medial_axis_nodes = set(selected)

        # 构建中轴边 (优化: 只检查局部邻域)
        maa = set()
        if len(self.medial_axis_nodes) > 1:
            sel_list = list(self.medial_axis_nodes)
            sel_kdtree = KDTree(sel_list)
            
            for i, node_a in enumerate(sel_list):
                # 只检查局部邻域内的节点对
                indices = sel_kdtree.query_ball_point(node_a, search_radius)
                for j in indices:
                    if j > i:  # 避免重复
                        node_b = sel_list[j]
                        if self._has_line_of_sight(node_a, node_b):
                            edge = (node_a, node_b) if node_a <= node_b else (node_b, node_a)
                            maa.add(edge)
        
        self.medial_axis_edges = maa

    def _adjust_path_to_medial(self, path, adjacency_set, medial_set):
        """
        调整路径: 若内部普通节点 x 前后节点 (prev,next) 都与同一中轴节点 m 相邻，
        用 m 替换 x，使路径更贴合中轴。循环直到不再变化；选择清距最大的候选 m。
        """
        path = list(path)
        changed = True
        while changed:
            changed = False
            for i in range(1, len(path) - 1):
                x = path[i]
                if x in medial_set:
                    continue
                prev_node = path[i - 1]
                next_node = path[i + 1]
                # 在 x 的邻居里找同时与 prev 和 next 相连且是中轴的节点
                candidates = [
                    m for m in adjacency_set.get(x, [])
                    if m in medial_set and
                    prev_node in adjacency_set.get(m, set()) and
                    next_node in adjacency_set.get(m, set())
                ]
                if candidates:
                    m_best = max(candidates, key=lambda n: self.node_clearance.get(n, 0.0))
                    path[i] = m_best
                    changed = True
                    break
        # 去除相邻重复
        dedup = [path[0]]
        for n in path[1:]:
            if n != dedup[-1]:
                dedup.append(n)
        return dedup

    def connect_medial_shortest_paths(self, threshold=4.0):
        """
        对每个中轴节点运行限距 Dijkstra:
        若到另一中轴节点的最短路径长度 < threshold:
            - 记录该最短路径(去重)
            - 将路径上相邻节点加入 medial_axis_edges
        """
        if not self.medial_axis_nodes:
            self.medial_axis_paths = []
            self.medial_axis_edges = set()
            return

        # 建图 (邻接 + 权重)
        adj = {n: [] for n in self.nodes}
        for a, b in self.edges:
            w = math.hypot(b[0]-a[0], b[1]-a[1])
            adj[a].append((b, w))
            adj[b].append((a, w))

        medial_list = sorted(self.medial_axis_nodes)
        medial_index = {n: i for i, n in enumerate(medial_list)}

        path_set = set()
        self.medial_axis_paths = []
        new_edges = set()

        for src in medial_list:
            # Dijkstra 限距
            dist = {src: 0.0}
            parent = {}
            hq = [(0.0, src)]
            while hq:
                d, u = heapq.heappop(hq)
                if d > threshold:
                    continue
                for v, w in adj.get(u, []):
                    nd = d + w
                    if nd >= threshold + 1e-9:
                        continue
                    if nd + 1e-9 < dist.get(v, float('inf')):
                        dist[v] = nd
                        parent[v] = u
                        heapq.heappush(hq, (nd, v))

            # 处理目标中轴节点 (只保留 src 索引小于 tgt 防重复)
            for tgt, dval in dist.items():
                if tgt == src or tgt not in self.medial_axis_nodes:
                    continue
                if medial_index[src] > medial_index[tgt]:
                    continue
                # 回溯最短路径
                path = [tgt]
                cur = tgt
                while cur != src:
                    cur = parent.get(cur)
                    if cur is None:
                        path = []
                        break
                    path.append(cur)
                if not path:
                    continue
                path.reverse()
                key_path = tuple(path)
                if key_path in path_set:
                    continue
                path_set.add(key_path)
                self.medial_axis_paths.append(path)
                # 加入路径边
                for i in range(len(path)-1):
                    a, b = path[i], path[i+1]
                    new_edges.add((a, b) if a <= b else (b, a))

        self.medial_axis_edges = new_edges

    def simplify_medial_axis(self):
        """
        中轴简化规则:
        对每个中轴节点 m:
        - 查找其在 PRM 中的普通邻居节点集合 N_normal(m) (非中轴节点)
        - 若 N_normal(m) 中任意两节点 (u,v) 之间有中轴边相连
        - 则删除该中轴边 (u,v)，并添加 (m,u) 和 (m,v) 作为中轴边
        作用: 将普通节点间的冗余中轴边重定向到中轴节点，形成以中轴为中心的星状结构。
        """
        if not self.medial_axis_nodes or not self.medial_axis_edges:
            return

        medial_set = set(self.medial_axis_nodes)
    
        # PRM 全图邻接关系
        full_adj = {n: set() for n in self.nodes}
        for a, b in self.edges:
            if a in full_adj and b in full_adj:
                full_adj[a].add(b)
                full_adj[b].add(a)

        # 规范化边
        def norm_edge(a, b):
            return (a, b) if a <= b else (b, a)

        # 当前中轴边集合
        medial_edges = {norm_edge(a, b) for (a, b) in self.medial_axis_edges}
    
        to_remove = set()
        to_add = set()
    
        # 遍历每个中轴节点
        for m in medial_set:
            # 获取普通邻居 (在 PRM 中相邻但不是中轴节点)
            normal_neighbors = [n for n in full_adj.get(m, []) if n not in medial_set]
        
            # 检查普通邻居对之间是否有中轴边
            for i in range(len(normal_neighbors)):
                for j in range(i+1, len(normal_neighbors)):
                    u, v = normal_neighbors[i], normal_neighbors[j]
                    e_uv = norm_edge(u, v)
                
                    # 如果普通邻居间有中轴边，移除它并添加到中轴
                    if e_uv in medial_edges:
                        to_remove.add(e_uv)
                    
                        # 添加中轴到两个普通节点的中轴边
                        e_mu = norm_edge(m, u)
                        e_mv = norm_edge(m, v)
                    
                        if e_mu not in medial_edges:
                            to_add.add(e_mu)
                    
                        if e_mv not in medial_edges:
                            to_add.add(e_mv)

        # 应用变更
        if to_remove or to_add:
            medial_edges.difference_update(to_remove)
            medial_edges.update(to_add)
        
            self.medial_axis_edges = medial_edges
            # 重建路径为简单的边列表
            self.medial_axis_paths = [[a, b] for (a, b) in self.medial_axis_edges]

    def prune_graph_components(self):
        """
        PRM 生成后裁剪:
        1) 删除度=0 节点
        2) 仅保留最大连通分量 (若并列取平均清距最大)
        """
        if not self.nodes:
            return
        # 建邻接
        adj = {n: set() for n in self.nodes}
        for a, b in self.edges:
            if a in adj and b in adj:
                adj[a].add(b)
                adj[b].add(a)

        # 去除度=0 节点
        nodes_with_edges = {n for n, neigh in adj.items() if neigh}
        if not nodes_with_edges:
            # 全孤立 => 清空
            self.nodes = []
            self.edges = []
            self.node_clearance = {}
            return

        # 仅保留含边节点的子图
        # 重新构建邻接 (仅带边节点)
        adj_reduced = {n: set() for n in nodes_with_edges}
        for a, b in self.edges:
            if a in nodes_with_edges and b in nodes_with_edges:
                adj_reduced[a].add(b)
                adj_reduced[b].add(a)

        # 连通分量
        visited = set()
        comps = []
        from collections import deque
        for n in nodes_with_edges:
            if n in visited:
                continue
            q = deque([n])
            visited.add(n)
            comp = []
            while q:
                u = q.popleft()
                comp.append(u)
                for v in adj_reduced[u]:
                    if v not in visited:
                        visited.add(v)
                        q.append(v)
            comps.append(comp)

        if len(comps) == 1:
            keep = set(comps[0])
        else:
            # 选最大 |comp|, 并列取平均 clearance 最大
            def comp_key(c):
                avg_clr = sum(self.node_clearance.get(x, 0.0) for x in c) / max(1, len(c))
                return (len(c), avg_clr)
            keep = set(max(comps, key=comp_key))

        # 过滤 nodes / edges / clearance
        self.nodes = [n for n in self.nodes if n in keep]
        self.edges = [ (a, b) for (a, b) in self.edges if a in keep and b in keep ]
        self.node_clearance = {n: self.node_clearance[n] for n in keep if n in self.node_clearance}

    def _is_cardinal_enclosed(self, x, y):
        """
        十字方向包围判定:
        取四个偏移点 (±0.5*cs,0),(0,±0.5*cs)
        统计落入障碍/越界的方向个数 blocked
        条件:
            - blocked >= 3  -> 判定被包围
            - blocked == 2 且这两个方向不是一对相反 -> 判定被包围
        返回 True 表示该节点应被丢弃
        """
        cs = ENV_CONFIG['cell_size']
        dirs = [(1,0),(-1,0),(0,1),(0,-1)]
        grid_w, grid_h = self.grid_width, self.grid_height
        obs_set = set(self.obstacles)
        blocked_dirs = []
        for dx, dy in dirs:
            px = x +  cs * dx
            py = y +  cs * dy
            gx = int(px / cs)
            gy = int(py / cs)
            # 越界视为障碍
            if gx < 0 or gx >= grid_w or gy < 0 or gy >= grid_h or (gx, gy) in obs_set:
                blocked_dirs.append((dx, dy))
        if len(blocked_dirs) >= 3:
            return True
        if len(blocked_dirs) == 2:
            d1, d2 = blocked_dirs
            # 相反: dx1 == -dx2 且 dy1 == -dy2
            if not (d1[0] == -d2[0] and d1[1] == -d2[1]):
                return True
        return False

    def find_path(self, start, goal):
        """使用A*算法在路图中查找从start到goal的路径，因为引入了中轴骨架和后向算法，因此重写函数覆盖基类"""     
        start_time = time.time()   
        if not self._is_valid_position(start[0], start[1]):
            raise ValueError("Start position is invalid or in collision.")
        if not self._is_valid_position(goal[0], goal[1]):
            raise ValueError("Goal position is invalid or in collision.")

        if not self.nodes:
            self.generate_prm()
        if not self.nodes:
            raise ValueError("Cannot generate any nodes in the PRM.")
        if not self.medial_axis_all_nodes:
            raise ValueError("Medial axis is empty, cannot find path.")
        
        adjacency = {node: {} for node in self.medial_axis_all_nodes}
        for edge in self.medial_axis_edges:
            u, v = edge
            dist = self._distance(u, v)
            adjacency[u][v] = dist
            adjacency[v][u] = dist

        adjacency[start] = {}
        adjacency[goal] = {}
        start_connected = False
        goal_connected = False
        for node in self.medial_axis_all_nodes:
            if self._is_valid_edge(start, node):
                dist = self._distance(start, node)
                adjacency[start][node] = dist
                adjacency[node][start] = dist
                start_connected = True
            if self._is_valid_edge(goal, node):
                dist = self._distance(goal, node)
                adjacency[goal][node] = dist
                adjacency[node][goal] = dist
                goal_connected = True
        
        if not start_connected:
            # 从稠密图中寻找连接
            for node in self.nodes:
                if self._is_valid_edge(start, node):
                    node_connected = False
                    for m in self.medial_axis_all_nodes:
                        if self._is_valid_edge(node, m):
                            dist = self._distance(node, m)
                            if node not in adjacency:
                                adjacency[node] = {}
                            adjacency[node][m] = dist
                            adjacency[m][node] = dist
                            node_connected = True
                    if node_connected:
                        dist = self._distance(start, node)
                        adjacency[start][node] = dist
                        adjacency[node][start] = dist
                        start_connected = True
                        break
        if not goal_connected:
            # 从稠密图中寻找连接
            for node in self.nodes:
                if self._is_valid_edge(goal, node):
                    node_connected = False
                    for m in self.medial_axis_all_nodes:
                        if self._is_valid_edge(node, m):
                            dist = self._distance(node, m)
                            if node not in adjacency:
                                adjacency[node] = {}
                            adjacency[node][m] = dist
                            adjacency[m][node] = dist
                            node_connected = True
                    if node_connected:
                        dist = self._distance(goal, node)
                        adjacency[goal][node] = dist
                        adjacency[node][goal] = dist
                        goal_connected = True
                        break

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
                
                path_length = sum(self._distance(path_nodes[i], path_nodes[i + 1]) for i in range(len(path_nodes) - 1))

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

    def cal_dispersion(self, num_samples=1000, media=False):
        """
        计算路图的离散度 (Dispersion) - 性能优化版本。
        """
        if not media:
            if not self.nodes:
                return float('inf')
        else:
            if not self.medial_axis_all_nodes:
                return float('inf')

        # 预计算物理尺寸和常量
        physical_width = self.grid_width * ENV_CONFIG['cell_size']
        physical_height = self.grid_height * ENV_CONFIG['cell_size']
        cell_size = ENV_CONFIG['cell_size']
        
        # 构建节点KDTree加速距离查询
        nodes_array = np.array(self.nodes) if not media else np.array(list(self.medial_axis_all_nodes))
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

    def calculate_node_utilization(self, num_test_paths=50):
        """
        计算PRM中节点的平均利用率
        通过随机生成多对起点和终点，计算路径中实际使用的节点占总节点数的比例
        
        对于 BeamPRM，由于路径规划使用的是 medial_axis_all_nodes（骨架节点），
        所以统计的是骨架节点的利用率
        
        参数:
            num_test_paths: 测试路径数量
        
        返回:
            dict: 包含利用率统计信息
                - avg_utilization: 平均利用率 (使用的节点数 / 总节点数)
                - node_usage_count: 每个节点被使用的次数
                - total_nodes: 总节点数
                - used_nodes: 至少被使用一次的节点数
                - successful_paths: 成功找到路径的数量
                - avg_nodes_per_path: 平均每条路径使用的节点数
        """
        # 使用骨架节点进行统计（BeamPRM的路径规划基于骨架）
        nodes_to_check = self.medial_axis_all_nodes if hasattr(self, 'medial_axis_all_nodes') and self.medial_axis_all_nodes else self.nodes
        
        if not nodes_to_check or len(nodes_to_check) < 2:
            return {
                'avg_utilization': 0.0,
                'node_usage_count': {},
                'total_nodes': 0,
                'used_nodes': 0,
                'successful_paths': 0,
                'avg_nodes_per_path': 0.0
            }
        
        # 记录每个节点被使用的次数
        node_usage_count = {node: 0 for node in nodes_to_check}
        successful_paths = 0
        total_path_nodes = 0
        
        # 生成随机起点和终点对
        valid_pairs = self.generate_valid_point_pairs(num_test_paths)
        
        for start, goal in valid_pairs:
            try:
                result = self.find_path(start, goal)
                # find_path 返回 (path_nodes, path_edges, path_length, search_time)
                if result and result[0] and len(result[0]) > 1:
                    path = result[0]  # path_nodes
                    successful_paths += 1
                    # 统计路径中的节点使用情况（排除起点和终点，因为它们不在nodes中）
                    for node in path:
                        if node in node_usage_count:
                            node_usage_count[node] += 1
                    total_path_nodes += len([n for n in path if n in node_usage_count])
            except Exception:
                # 路径查找失败，跳过
                continue
        
        # 计算统计信息
        total_nodes = len(nodes_to_check)
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
    
    def cal_discrepancy(self, num_samples=1000, media=False):
        """
        计算节点集的星偏差度 (Star Discrepancy) - 性能优化版本。
        限制矩形面积不超过地图总面积的1/8。
        """
        if not media:
            if not self.nodes:
                return 1.0
        else:
            if not self.medial_axis_all_nodes:
                return 1.0

        # 预计算常量
        physical_width = self.grid_width * ENV_CONFIG['cell_size']
        physical_height = self.grid_height * ENV_CONFIG['cell_size']
        total_area = physical_width * physical_height
        max_area = total_area / 8  # 限制最大矩形面积为地图总面积的1/8
        num_nodes = len(self.nodes) if not media else len(self.medial_axis_all_nodes)

        # 将节点转换为NumPy数组 - 只做一次
        np_nodes = np.array(self.nodes) if not media else np.array(list(self.medial_axis_all_nodes))

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
            
            # 计算矩形面积并检查是否超限
            rect_area = (x2 - x1) * (y2 - y1)
            
            # 如果面积超过限制，缩小矩形保持中心点不变
            if rect_area > max_area:
                # 计算矩形中心
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                
                # 计算缩放因子
                scale = math.sqrt(max_area / rect_area)
                
                # 计算新的半宽和半高
                half_width = (x2 - x1) / 2 * scale
                half_height = (y2 - y1) / 2 * scale
                
                # 更新矩形坐标
                x1 = center_x - half_width
                x2 = center_x + half_width
                y1 = center_y - half_height
                y2 = center_y + half_height
            
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
    
class DeltaPRM(BasePathPlanner):
    """
    delta-PRM算法
    使用最小距离阈值(delta)来确保采样点不会过于密集
    使用最大连接半径来限制边的连接距离
    """

    def __init__(self, grid_width, grid_height, obstacles, 
                 num_nodes=500, 
                 delta_radius=0.15,         # 最小连接距离
                 connection_radius=1.6,     # 最大连接距离
                 max_failures=30,  # 最大连续失败次数
                 **kwargs):
        super().__init__(grid_width, grid_height, obstacles, **kwargs)
        self.num_nodes = num_nodes
        self.delta_radius = delta_radius  # 最小距离阈值
        self.connection_radius = connection_radius  # 最大连接半径
        self.max_failures = max_failures
        self.node_kdtree = None  # 用于快速查找近邻节点

    def _is_delta_valid(self, node):
        """
        检查节点是否满足delta条件
        如果node距离任何现有节点小于delta_radius，则返回False
        """
        if not self.nodes:
            return True
            
        # 如果已经构建了KD树，使用它来加速查询
        if self.node_kdtree is not None:
            dist, _ = self.node_kdtree.query(np.array(node))
            return dist >= self.delta_radius
        
        # 否则使用暴力搜索
        for existing_node in self.nodes:
            if self._distance(node, existing_node) < self.delta_radius:
                return False
        return True
    
    def _update_kdtree(self):
        """更新KD树以加速邻近搜索"""
        if len(self.nodes) > 10:  # 当节点数量足够时才构建KD树
            self.node_kdtree = KDTree(np.array(self.nodes))
        else:
            self.node_kdtree = None
            
    def _find_connections(self, node):
        """为节点寻找连接半径内的有效连接"""
        connections = []
        
        # 使用KD树加速搜索，如果可用
        if self.node_kdtree is not None:
            indices = self.node_kdtree.query_ball_point(
                np.array(node), self.connection_radius
            )
            for i in indices:
                neighbor = self.nodes[i]
                if neighbor != node and self._is_valid_edge(node, neighbor):
                    connections.append(neighbor)
        else:
            # 暴力搜索
            for neighbor in self.nodes:
                if neighbor != node and self._distance(node, neighbor) <= self.connection_radius:
                    if self._is_valid_edge(node, neighbor):
                        connections.append(neighbor)
                        
        return connections

    def generate_prm(self):
        """生成delta-PRM路径图"""
        print("开始生成delta-PRM...")
        start_time = time.time()
    
        self.nodes = []
        self.edges = []
        self.node_kdtree = None
    
        print(f"目标采样: {self.num_nodes} 个节点...")
        sampled_count = 0
        consecutive_failures = 0
        total_attempts = 0
        
        # 主循环：直到达到目标节点数或连续失败次数过多
        while sampled_count < self.num_nodes and consecutive_failures < self.max_failures:
            total_attempts += 1
            
            # 1. 随机采样一个点
            sample = self._random_sample()
            if sample is None:
                continue
                
            # 2. 检查delta有效性（与现有点的最小距离约束）
            if not self._is_delta_valid(sample):
                consecutive_failures += 1
                continue
                
            # 3. 通过所有检查，将新点添加到图中
            self.nodes.append(sample)
            sampled_count += 1
            consecutive_failures = 0  # 重置连续失败计数
            self._update_kdtree()
            # 4. 定期更新KD树以加速后续搜索
            if sampled_count % 50 == 0:
                print(f"已采样 {sampled_count}/{self.num_nodes} 节点，连续失败: {consecutive_failures}")
            
            # 5. 为新节点寻找连接
            connections = self._find_connections(sample)
            for neighbor in connections:
                self.edges.append((sample, neighbor))

        if consecutive_failures >= self.max_failures:
            print(f"达到最大连续失败次数：{self.max_failures}次")
        # 最后一次构建KD树，确保完整性
        self._update_kdtree()
        
        # 找到最大连通分量并丢弃孤立点
        self._keep_largest_component()
    
        end_time = time.time()
        print(f"delta-PRM生成完成！")
        print(f"节点数: {len(self.nodes)}")
        print(f"边数: {len(self.edges)}")
        print(f"生成时间: {end_time - start_time:.2f} 秒")
        print(f"总尝试次数: {total_attempts}, 成功率: {sampled_count/max(1,total_attempts)*100:.1f}%")
    
        return self.nodes, self.edges
        
    def _keep_largest_component(self):
        """保留最大连通分量，删除孤立点"""
        if not self.nodes:
            return
            
        # 构建邻接表
        adj = defaultdict(list)
        for a, b in self.edges:
            adj[a].append(b)
            adj[b].append(a)
            
        # 找出所有连通分量
        visited = set()
        components = []
        
        for node in self.nodes:
            if node in visited:
                continue
                
            # BFS找出一个连通分量
            component = []
            queue = [node]
            visited.add(node)
            
            while queue:
                current = queue.pop(0)
                component.append(current)
                
                for neighbor in adj[current]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
                        
            components.append(component)
            
        # 如果没有连通分量，返回
        if not components:
            return
            
        # 找出最大的连通分量
        largest_component = max(components, key=len)
        largest_component_set = set(largest_component)
        
        # 更新节点和边
        self.nodes = [n for n in self.nodes if n in largest_component_set]
        self.edges = [(a, b) for a, b in self.edges 
                     if a in largest_component_set and b in largest_component_set]
                             
class PRMStar(BasePathPlanner):
    """
    渐进最优概率路径图算法 (PRM*)
    该算法使用一个随采样点数量动态调整的连接半径，以理论上保证路径的渐进最优性。
    """

    def __init__(self, grid_width, grid_height, obstacles, num_nodes=500, gamma_prm_star=1.5, **kwargs):
        """
        初始化 PRM* 算法
        :param num_nodes: 采样的节点数量
        :param gamma_prm_star: PRM*算法中用于计算可变半径的常数。这个值需要根据环境进行调整。
                               一个经验法则是，它应该略大于空间维度的2倍。
        """
        super().__init__(grid_width, grid_height, obstacles, **kwargs)
        self.num_nodes = num_nodes
        self.gamma_prm_star = gamma_prm_star
        # 空间的维度，对于2D平面是2
        self.space_dimension = 2.0
    
    def generate_prm(self):
        """
        生成PRM*路径图。
        该方法采用增量式构建，每采样一个新节点，就立刻根据可变半径尝试连接它。
        """
        print("开始生成 PRM* (可变半径)...")
        start_time = time.time()
    
        self.nodes = []
        self.edges = []
    
        print(f"采样并增量连接 {self.num_nodes} 个节点...")
        sampled_count = 0
        attempts = 0
        max_attempts = self.num_nodes * 10
    
        while sampled_count < self.num_nodes and attempts < max_attempts:
            attempts += 1
            # 1. 在自由空间中随机采样一个新节点
            new_node = self._random_sample()
            if new_node is None:
                continue
        
            # 2. 将新节点添加到图中
            self.nodes.append(new_node)
            sampled_count += 1
            
            n = len(self.nodes)

            # 3. PRM* 核心: 计算可变连接半径
            # 半径 r = gamma * (log(n)/n)^(1/d)
            # 当n=1时, log(n)=0, 半径为0, 不进行连接
            if n > 1:
                radius = self.gamma_prm_star * (math.log(n) / n)**(1.0 / self.space_dimension)
            
                # 4. 寻找新节点在可变半径内的邻居
                # 注意: 为了最高效率, 这里的节点搜索也应该用KDTree, 但为保持与原代码结构一致, 此处使用遍历
                neighbors = []
                for existing_node in self.nodes[:-1]: # 遍历除新节点外的所有已有节点
                    if self._distance(new_node, existing_node) <= radius:
                        neighbors.append(existing_node)
            
                # 5. 尝试连接新节点与它的邻居
                for neighbor in neighbors:
                    if self._is_valid_edge(new_node, neighbor):
                        self.edges.append((new_node, neighbor))

            if sampled_count % 50 == 0 and sampled_count > 0:
                print(f"已处理 {sampled_count}/{self.num_nodes} 个节点，图中有 {len(self.edges)} 条边")
    
        end_time = time.time()
        print(f"PRM* 生成完成！")
        print(f"实际采样节点数: {len(self.nodes)}")
        print(f"生成边数: {len(self.edges)}")
        print(f"生成时间: {end_time - start_time:.2f} 秒")
    
        return self.nodes, self.edges

class UnionFind:
    """并查集，用于高效维护连通组件"""
    def __init__(self):
        self.parent = {}
        self.num_components = 0
    
    def make_set(self, x):
        if x not in self.parent:
            self.parent[x] = x
            self.num_components += 1
    
    def find(self, x):
        if x not in self.parent:
            self.make_set(x)
            return x
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x != root_y:
            self.parent[root_y] = root_x
            self.num_components -= 1
            return True
        return False

class SPARS(BasePathPlanner):
    def __init__(self, grid_width, grid_height, obstacles, num_nodes=500, max_failures=20, visibility_radius = 1.0, stretch_factor = 3.0, connection_radius = 0.6, delta = 0.1, **kwargs):
        super().__init__(grid_width, grid_height, obstacles, **kwargs)
        self.num_nodes = num_nodes
        self.visibility_radius = visibility_radius
        self.t_value = stretch_factor
        self.max_failures = max_failures
        self.dense_connection_radius = 0.6 # 稠密图连接半径
        self.delta = delta

        self.sparse_nodes = []
        self.sparse_adj = defaultdict(dict)  # 稀疏图邻接表
        self.components = UnionFind()  # 稀疏图连通分量管理器
        self.dense_nodes = []
        self.dense_adj = defaultdict(dict)   # 稠密图邻接表

        self.representatives = {}  # 稠密图节点到稀疏图代表节点的映射

    def _find_nearby_guards(self, node):
        """在稀疏图中寻找距离node小于visibility_radius的最近节点"""
        guards = []
        for point in self.sparse_nodes:
            dist = self._distance(node, point)
            if dist <= self.visibility_radius and self._is_valid_edge(node, point):
                guards.append(point)
        return guards
    
    def add_sparse_node(self, node):
        if node not in self.sparse_nodes:
            self.sparse_nodes.append(node)
            self.components.make_set(node)
            self.representatives[node] = node

    def add_sparse_edge(self, u, v):
        if u != v and v not in self.sparse_adj[u]:
            dist = self._distance(u, v)
            self.sparse_adj[u][v] = dist
            self.sparse_adj[v][u] = dist
            self.components.union(u, v)
        
    def generate_prm(self):
        """主函数：生成SPARS路径图"""
        print(f"开始生成SPARS路径图 (max_samples={self.num_nodes}, stretch_factor={self.t_value})")
        failures = 0

        for _ in range(self.num_nodes):
            if failures >= self.max_failures:
                print(f"达到最大失败次数{self.max_failures}，停止采样。")
                break
            sample = self._random_sample()
            if sample is None:
                continue
            # 随机采样生成稠密图节点并和可达节点连接
            self.dense_nodes.append(sample)
            for node_d in self.dense_nodes[:-1]:
                dist = self._distance(sample, node_d)
                if dist <= self.dense_connection_radius and self._is_valid_edge(sample, node_d):
                    self.dense_adj[sample][node_d] = dist
                    self.dense_adj[node_d][sample] = dist
            
            guards = self._find_nearby_guards(sample)
            # 情况零，检查是否离已有 guard 太近
            flag = False
            for g in guards:
                if self._distance(sample, g) <= self.delta:
                    failures += 1
                    flag = True
                    break
            if flag:
                continue

            # 情况一，没有可以连接的 guard ，说明增加了覆盖度，加到稀疏图中
            if not guards:
                self.add_sparse_node(sample)
                self.representatives[sample] = sample
                # print(f"添加新的稀疏节点 {sample}，当前稀疏图节点数: {len(self.sparse_nodes)}")
                failures = 0
                continue
            else:
                # 情况二，是否连接两个不连通区域，使用并查集判断
                guard = min(guards, key=lambda g: self._distance(sample, g))
                self.representatives[sample] = guard
                
                visible_guards = {self.components.find(g) for g in guards}
                if len(visible_guards) > 1:
                    self.add_sparse_node(sample)
                    for g in guards:
                        self.add_sparse_edge(g, sample)
                    failures = 0
                    continue
                # print(f"情况二：样本点 {sample} 连接到守卫图")
                # 情况三，检查是否需要添加中间节点以确保接口连接
                for point in self.dense_adj.get(sample, {}):
                    # 检查临近节点的守卫节点
                    if point not in self.representatives:
                        continue
                    rep_point = self.representatives[point]

                    if guard == rep_point:
                        continue
                    else:
                        if rep_point in self.sparse_adj.get(guard, {}):
                            # 看看守卫节点和临近节点的守卫节点是否能直接相连，如果能直接相连那就说明没用
                            continue
                        else:
                            if self._is_valid_edge(guard, rep_point):
                                self.add_sparse_edge(guard, rep_point)
                            else:
                                midpoint = ((sample[0] + point[0]) / 2, (sample[1] + point[1]) / 2)
                                if self._is_valid_edge(midpoint, guard) and self._is_valid_edge(midpoint, rep_point):
                                    self.add_sparse_node(midpoint)
                                    self.add_sparse_edge(midpoint, guard)
                                    self.add_sparse_edge(midpoint, rep_point)
                                else:
                                    self.add_sparse_node(sample)
                                    self.add_sparse_node(point)
                                    self.add_sparse_edge(sample, guard)
                                    self.add_sparse_edge(sample, point)
                                    self.add_sparse_edge(point, rep_point)
                                failures = 0 # 更新了稀疏图，失败计数置零
                                break
  
            failures += 1
        print(f"SPARS生成完成！")
        print(f"稀疏图节点数: {len(self.sparse_nodes)}")
        print(f"稀疏图边数: {sum(len(v) for v in self.sparse_adj.values()) // 2}")
        print(f"稠密图节点数: {len(self.dense_nodes)}")
        print(f"稠密图边数: {sum(len(v) for v in self.dense_adj.values()) // 2}")
        self.nodes = self.sparse_nodes
        self.edges = []
        for u in self.sparse_adj:
            for v in self.sparse_adj[u]:
                if (v, u) not in self.edges:
                    self.edges.append((u, v))
        return self.nodes, self.edges                             

class SPARS2(BasePathPlanner):
    def __init__(self, grid_width, grid_height, obstacles, 
                 num_nodes=2000,
                 max_failures=500,
                 visibility_radius=2.0,
                 stretch_factor=3.0,
                 delta=0.2,
                 k=4,
                 **kwargs):
        super().__init__(grid_width, grid_height, obstacles, **kwargs)

        self.M = max_failures
        self.Delta = visibility_radius
        self.t_value = stretch_factor
        self.delta = delta
        self.k = k
        self.max_samples = num_nodes 

        self.sparse_nodes = []
        self.sparse_adj = defaultdict(dict)
        self.components = UnionFind()
        self.interface_data = defaultdict(dict)

    def _find_visible_guards(self, node, radius):
        guards = []
        r_sq = radius ** 2
        
        for g in self.sparse_nodes:
            d_sq = (node[0] - g[0])**2 + (node[1] - g[1])**2
            if d_sq <= r_sq:
                if self._is_valid_edge(node, g):
                    guards.append(g)
        return guards

    def generate_prm(self):
        print(f"开始生成 SPARS2 (Delta={self.Delta}, t={self.t_value}, M={self.M})")
        start_time = time.time()
        
        self.sparse_nodes = []
        self.sparse_adj = defaultdict(dict)
        self.components = UnionFind()
        self.interface_data = defaultdict(dict)
        
        failures = 0
        total_attempts = 0

        while failures < self.M and total_attempts < self.max_samples:
            total_attempts += 1
            if total_attempts % 100 == 0:
                print(f"采样: {total_attempts}/{self.max_samples}, 节点: {len(self.sparse_nodes)}")

            rho = self._random_sample()
            if rho is None: continue

            visible_guards = self._find_visible_guards(rho, self.Delta)

            # 覆盖准则
            if not visible_guards:
                self._add_node(rho)
                failures = 0
                continue
            
            # 找到最近的哨兵 v
            v = min(visible_guards, key=lambda n: self._distance(rho, n))
            # 连通性准则
            comp_ids = {self.components.find(g) for g in visible_guards}
            
            if len(comp_ids) > 1:
                # 连接不同的连通分量
                self._add_node(rho)
                for g in visible_guards:
                    if self.components.find(g) != self.components.find(rho):
                        self._add_edge(rho, g)
                failures = 0
                continue

            # 接口处理 (Interface)
            if len(visible_guards) >= 2:
                sorted_guards = sorted(visible_guards, key=lambda n: self._distance(rho, n))
                v1, v2 = sorted_guards[0], sorted_guards[1]
                
                if v1 != v2 and not self._has_edge(v1, v2):
                    # 只有当成功添加了边/路径才重置 failures
                    if self._close_interface(rho, v1, v2):
                        failures = 0

            # 稀疏图路径优化
            if rho not in self.sparse_nodes:
                Sigma, R, graph_changed = self._get_close_reps(rho, v)
                
                if R:
                    added_change = False
                    
                    # 更新接口信息
                    for r, sigma in zip(R, Sigma):
                        self._update_points(rho, sigma, v, r)
                        self._update_points(sigma, rho, r, v)
                    if self._test_add_path(v):
                        added_change = True
                    
                    for r in R:
                        if self._test_add_path(r):
                            added_change = True
                    
                    if added_change or graph_changed:
                        failures = 0
                    else:
                        failures += 1
                else:
                    failures += 1
            else:
                failures = 0

        self._sync_to_base()
        print(f"完成。耗时: {time.time()-start_time:.2f}s, 节点: {len(self.nodes)}, 边: {len(self.edges)}")
        return self.nodes, self.edges

    def _get_close_reps(self, rho, v):
        Sigma = []
        R = []
        for _ in range(self.k):
            sigma = self._sample_near(rho, self.delta)
            if not sigma: continue

            if self._is_valid_edge(rho, sigma):
                N_sigma = self._find_visible_guards(sigma, self.Delta)
                
                if not N_sigma:
                    self._add_node(sigma)
                    return [], [], True
                
                v_sigma = min(N_sigma, key=lambda n: self._distance(sigma, n))
                
                if v != v_sigma:
                    Sigma.append(sigma)
                    R.append(v_sigma)
        return Sigma, R, False

    def _update_points(self, rho, sigma, v, r):
        v_neighbors = list(self.sparse_adj.get(v, {}).keys())
        for r_prime in v_neighbors:
            if r_prime == r: continue
            if self._has_edge(r, r_prime): continue

            key = frozenset({r, r_prime})
            
            data = self.interface_data[v].get(key, {
                'dist': float('inf'),
                'point_map': {} 
            }).copy()
            
            data['point_map'] = data['point_map'].copy()
            
            stored_map = data['point_map']
            other_side = stored_map.get(r_prime)
            
            new_dist = float('inf')
            if other_side:
                p_prime = other_side[0]
                new_dist = self._distance(rho, p_prime)
            
            if other_side is None or new_dist < data['dist']:
                stored_map[r] = (rho, sigma)
                data['point_map'] = stored_map
                
                if other_side:
                    data['dist'] = new_dist
                    data['p'] = rho 
                    data['p_prime'] = other_side[0]
                    data['xi'] = sigma
                    data['xi_prime'] = other_side[1]
                
                self.interface_data[v][key] = data

    def _test_add_path(self, v):
        if v not in self.interface_data: return False
        success = False

        # 转换为 list 避免迭代时修改字典错误
        for key, data in list(self.interface_data[v].items()):
            r_list = list(key)
            if len(r_list) != 2: continue
            r, r_prime = r_list[0], r_list[1]
            
            if self._has_edge(r, r_prime): continue
            
            if data['dist'] == float('inf') or 'p' not in data: continue

            rho, rho_prime = data['p'], data['p_prime']
            sigma, sigma_prime = data['xi'], data['xi_prime']
            d_physical = data['dist']

            # t-spanner 检查
            d_graph = self._dijkstra(r, r_prime)
            
            if d_graph > self.t_value * d_physical:
                # --- 开始事务性检查 ---
                
                # 1. 尝试直接加边 r -> r'
                if self._is_valid_edge(r, r_prime):
                    self._add_edge(r, r_prime)
                    success = True
                    continue # 成功则跳过后续复杂路径

                # 2. 尝试添加复杂路径: r -> (sigma) -> rho -> v -> rho' -> (sigma') -> r'
                # 我们先收集所有需要的边，验证全部通过后再添加
                
                edges_to_add = []
                nodes_to_add = set()

                # 段 1: r 到 rho
                seg1 = self._get_safe_connection(r, rho, sigma)
                if seg1 is None: continue # 路径不通，放弃
                edges_to_add.extend(seg1)

                # 段 2: rho 到 v (直接连)
                if not self._is_valid_edge(rho, v): continue
                edges_to_add.append((rho, v))

                # 段 3: v 到 rho' (直接连)
                if not self._is_valid_edge(v, rho_prime): continue
                edges_to_add.append((v, rho_prime))

                # 段 4: rho' 到 r'
                seg4 = self._get_safe_connection(rho_prime, r_prime, sigma_prime)
                if seg4 is None: continue
                edges_to_add.extend(seg4)

                # --- 所有检查通过，开始写入图 ---
                self._add_node(rho)
                self._add_node(rho_prime)
                
                # 将路径涉及的中间点加入节点集合
                for u, w in edges_to_add:
                    self._add_edge(u, w)
                
                success = True
                
        return success


    def _get_safe_connection(self, start, end, helper):
        """
        尝试连接 start 和 end。
        优先直连；如果直连碰撞，尝试通过 helper 连接。
        如果在必须使用 helper 时 helper 的路径也碰撞，则返回 None。
        返回: [(u1, v1), (u2, v2)...] 边列表
        """
        # 方案 A: 直连
        if self._is_valid_edge(start, end):
            return [(start, end)]
        
        # 方案 B: 通过 helper
        # 必须确保 start->helper 和 helper->end 都是有效的！
        if self._is_valid_edge(start, helper) and self._is_valid_edge(helper, end):
            return [(start, helper), (helper, end)]
        
        # 方案 C: 彻底失败
        return None

    def _close_interface(self, rho, v1, v2):
        """尝试连接接口 v1-v2，通过 rho 或直连"""
        # 1. 尝试直连
        if self._is_valid_edge(v1, v2):
            self._add_edge(v1, v2)
            return True
        
        # 2. 尝试通过 rho 桥接
        # 必须检查 v1->rho 和 rho->v2 是否真的无碰撞
        if self._is_valid_edge(v1, rho) and self._is_valid_edge(rho, v2):
            self._add_node(rho)
            self._add_edge(v1, rho)
            self._add_edge(rho, v2)
            return True
            
        return False

    def _add_node(self, node):
        if node not in self.sparse_nodes:
            self.sparse_nodes.append(node)
            self.components.make_set(node)

    def _add_edge(self, u, v):
        if u == v: return
        self._add_node(u)
        self._add_node(v)
        dist = self._distance(u, v)
        self.sparse_adj[u][v] = dist
        self.sparse_adj[v][u] = dist
        self.components.union(u, v)
        
    def _has_edge(self, u, v):
        return v in self.sparse_adj.get(u, {})

    def _sample_near(self, node, radius):
        r = radius * math.sqrt(random.random())
        theta = random.random() * 2 * math.pi
        x = node[0] + r * math.cos(theta)
        y = node[1] + r * math.sin(theta)
        if self._is_valid_position(x, y):
            return (x, y)
        return None

    def _dijkstra(self, start, goal):
        if start == goal: return 0.0
        # 优化：如果 start 或 goal 孤立，直接返回 inf
        if start not in self.sparse_adj or goal not in self.sparse_adj:
            return float('inf')

        pq = [(0.0, start)]
        dists = {start: 0.0}
        
        while pq:
            d, u = heapq.heappop(pq)
            if u == goal: return d
            
            if d > dists.get(u, float('inf')): continue
            
            for v, weight in self.sparse_adj.get(u, {}).items():
                new_dist = d + weight
                if new_dist < dists.get(v, float('inf')):
                    dists[v] = new_dist
                    heapq.heappush(pq, (new_dist, v))
        return float('inf')

    def _sync_to_base(self):
        self.nodes = self.sparse_nodes
        self.edges = []
        seen = set()
        for u, neighbors in self.sparse_adj.items():
            for v in neighbors:
                # 保证边只添加一次
                edge_key = tuple(sorted((u, v)))
                if edge_key not in seen:
                    self.edges.append((u, v))
                    seen.add(edge_key)

class GSRM(BasePathPlanner):
    def __init__(self, grid_width, grid_height, obstacles, 
                 upscale_factor=6,    
                 iterations=100,     
                 dt=2.0,
                 Du=0.32, Dv=0.08,      
                 A=0.035, 
                 B=0.063,
                 min_node_dist=0.5,
                 peak_min_distance=8,
                 **kwargs):
        super().__init__(grid_width, grid_height, obstacles, connection_radius=2.0, **kwargs)
        
        self.grid_w = int(grid_width)
        self.grid_h = int(grid_height)
        self.upscale = upscale_factor
        self.sim_w = self.grid_w * self.upscale
        self.sim_h = self.grid_h * self.upscale
        
        print(f"GSRM Reset: Grid {self.grid_w}x{self.grid_h} -> Sim {self.sim_w}x{self.sim_h}")
        print(f"Params: F(A)={A}, k(B)={B}, Du={Du}, Dv={Dv}")
        
        self.N = iterations
        self.dt = dt
        self.Du = Du
        self.Dv = Dv
        self.A = A
        self.B = B
        self.min_node_dist = min_node_dist
        self.peak_min_distance = peak_min_distance
        self.laplacian_kernel = np.array([[0.05, 0.20, 0.05],
                                          [0.20, -1.0, 0.20],
                                          [0.05, 0.20, 0.05]])
        self.nodes = []
        self.edges = []

    def generate_prm(self):
        start_time = time.time()
        self.obstacle_mask = self._create_upscaled_mask()

        self.U = np.random.uniform(0.8, 1.0, (self.sim_h, self.sim_w))
        self.V = np.random.uniform(0.0, 0.2, (self.sim_h, self.sim_w))        

        print("执行仿真中...")
        self._simulate()

        self._save_debug_viz("gsrm_debug_check.png")

        self.nodes = self._extract_nodes(threshold_ratio=0.35)
        if len(self.nodes) < 20:
            print("Too few nodes, trying lower threshold...")
            self.nodes = self._extract_nodes(threshold_ratio=0.25)
        
        print(f"Final nodes: {len(self.nodes)}")

        if len(self.nodes) > 1:
            self._construct_geometry()

        print(f"Time: {time.time()-start_time:.2f}s")
        return self.nodes, self.edges

    def _extract_nodes(self, threshold_ratio=0.4):
        from scipy.ndimage import maximum_filter
        
        max_v = np.max(self.V)
        if max_v < 0.05:
            return []
        
        threshold = max_v * threshold_ratio
        neighborhood_size = self.peak_min_distance
        
        local_max = maximum_filter(self.V, size=neighborhood_size, mode='constant', cval=0.0)
        is_peak = (self.V == local_max) & (self.V > threshold) & (~self.obstacle_mask)
        peak_coords = np.argwhere(is_peak)
        
        if len(peak_coords) == 0:
            return []
        
        peak_values = self.V[is_peak]
        peak_coords = peak_coords[np.argsort(-peak_values)]
        
        nodes = []
        for sim_cy, sim_cx in peak_coords:
            grid_x_pri = sim_cx / self.upscale
            grid_y_pri = sim_cy / self.upscale
            grid_x = grid_x_pri.item()
            grid_y = grid_y_pri.item()
            if self._is_valid_position(grid_x, grid_y):
                nodes.append((grid_x * ENV_CONFIG['cell_size'],
                              grid_y * ENV_CONFIG['cell_size']))
        
        min_dist = self.min_node_dist * ENV_CONFIG['cell_size']
        kept = []
        for node in nodes:
            if all(np.hypot(node[0]-k[0], node[1]-k[1]) >= min_dist for k in kept):
                kept.append(node)
        
        print(f"Extracted {len(kept)} nodes")
        return kept

    def _simulate(self):
        U = self.U
        V = self.V
        mask = self.obstacle_mask
        kernel = self.laplacian_kernel
        Du, Dv = self.Du, self.Dv
        A, B = self.A, self.B
        dt = self.dt

        for i in range(self.N):
            U[mask] = 0.0
            V[mask] = 0.0
            
            Lu = convolve2d(U, kernel, mode='same', boundary='fill', fillvalue=0)
            Lv = convolve2d(V, kernel, mode='same', boundary='fill', fillvalue=0)
            
            uvv = U * (V * V)
            U += (Du * Lu - uvv + A * (1.0 - U)) * dt
            V += (Dv * Lv + uvv - (A + B) * V) * dt
            
            np.clip(U, 0.0, 1.0, out=U)
            np.clip(V, 0.0, 1.0, out=V)

    def _create_upscaled_mask(self):
        mask = np.zeros((self.sim_h, self.sim_w), dtype=bool)
        for (ox, oy) in self.obstacles:
            r_start = int(oy * self.upscale)
            r_end = int((oy + 1) * self.upscale)
            c_start = int(ox * self.upscale)
            c_end = int((ox + 1) * self.upscale)
            
            r_start = max(0, r_start)
            r_end = min(self.sim_h, r_end)
            c_start = max(0, c_start)
            c_end = min(self.sim_w, c_end)
            
            mask[r_start:r_end, c_start:c_end] = True
        return mask

    def _construct_geometry(self):
        dummy_indices = np.argwhere(self.obstacle_mask)
        if len(dummy_indices) > 0:
            count = min(len(dummy_indices), len(self.nodes)*2)
            choices = np.random.choice(len(dummy_indices), count, replace=False)
            dummies = [((c/self.upscale) * ENV_CONFIG['cell_size'], (r/self.upscale) * ENV_CONFIG['cell_size']) for r, c in dummy_indices[choices]]
        else:
            dummies = []

        all_points = np.array(self.nodes + dummies)
        if len(all_points) < 4: return

        try:
            tri = Delaunay(all_points)
        except:
            return

        seen = set()
        num_real = len(self.nodes)
        
        for simplex in tri.simplices:
            for i in range(3):
                u, v = simplex[i], simplex[(i+1)%3]
                if u < num_real and v < num_real:
                    if u > v: u, v = v, u
                    if (u, v) in seen: continue
                    seen.add((u, v))
                    
                    if self._is_valid_edge(self.nodes[u], self.nodes[v]):
                        self.edges.append((self.nodes[u], self.nodes[v]))

    def _is_valid_edge(self, node1, node2, num_samples=None):
        x1 = node1[0] / ENV_CONFIG['cell_size']
        y1 = node1[1] / ENV_CONFIG['cell_size']
        x2 = node2[0] / ENV_CONFIG['cell_size']
        y2 = node2[1] / ENV_CONFIG['cell_size']
        
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

    def _is_valid_position(self, x, y):
        ix = int(x)
        iy = int(y)
        if (ix, iy) in self.obstacles: return False
        if ix < 0 or ix >= self.grid_w or iy < 0 or iy >= self.grid_h: return False
        return True
    def _save_debug_viz(self, filename):
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.title("Map & Obstacles (Mask)")
        plt.imshow(self.obstacle_mask, cmap='gray')
        plt.subplot(1, 2, 2)
        plt.title(f"Chemical V (Max={self.V.max():.2f})")
        plt.imshow(self.V, cmap='inferno')
        plt.colorbar()
        plt.savefig(filename)
        plt.close()

    def generate_valid_point_pairs(self, num_pairs):
        """生成指定数量的有效起终点对，返回世界坐标"""
        pairs = []
        attempts = 0
        max_attempts = num_pairs * 100          # ← ★ 修复②：原来 *10 在稀疏场景极易不够
        min_distance = 8 * ENV_CONFIG['cell_size']
        
        while len(pairs) < num_pairs and attempts < max_attempts:
            start = self._random_sample()
            goal  = self._random_sample()
            attempts += 1
            
            # ← ★ 修复①：_random_sample 无合法点时返回 None，必须跳过
            if start is None or goal is None:
                continue
            
            start_grid_x = start[0] / ENV_CONFIG['cell_size']
            start_grid_y = start[1] / ENV_CONFIG['cell_size']
            goal_grid_x  = goal[0]  / ENV_CONFIG['cell_size']
            goal_grid_y  = goal[1]  / ENV_CONFIG['cell_size']
            
            if (self._is_valid_position(start_grid_x, start_grid_y) and 
                    self._is_valid_position(goal_grid_x, goal_grid_y) and 
                    start != goal):
                
                if self._distance(start, goal) > min_distance:
                    pairs.append((start, goal))

        return pairs

    def find_path(self, start, goal):
        """使用A*算法在路图中查找从start到goal的路径"""     
        start_time = time.time()   

        start_grid_x = start[0] / ENV_CONFIG['cell_size']
        start_grid_y = start[1] / ENV_CONFIG['cell_size']
        goal_grid_x  = goal[0]  / ENV_CONFIG['cell_size']
        goal_grid_y  = goal[1]  / ENV_CONFIG['cell_size']

        if not self._is_valid_position(start_grid_x, start_grid_y):
            raise ValueError(f"Start position is invalid or in collision.")
        if not self._is_valid_position(goal_grid_x, goal_grid_y):
            raise ValueError(f"Goal position is invalid or in collision.")

        if not self.nodes:
            self.generate_prm()
        if not self.nodes:
            raise ValueError("Cannot generate any nodes in the PRM.")

        adjacency = {node: {} for node in self.nodes}
        for edge in self.edges:
            u, v = edge
            dist = self._distance(u, v)
            adjacency[u][v] = dist
            adjacency[v][u] = dist
        
        adjacency[start] = {}
        adjacency[goal]  = {}
        start_connected  = False
        goal_connected   = False
        for node in self.nodes:
            if self._is_valid_edge(start, node):
                dist = self._distance(start, node)
                adjacency[start][node] = dist
                adjacency[node][start] = dist
                start_connected = True
            if self._is_valid_edge(goal, node):
                dist = self._distance(goal, node)
                adjacency[goal][node]  = dist
                adjacency[node][goal]  = dist
                goal_connected = True
        
        if not start_connected or not goal_connected:
            return None, None, None, time.time() - start_time

        open_set       = [(self._distance(start, goal), start)]
        open_set_nodes = {start}
        backtrack      = {}
        g_score        = {start: 0}
        visited        = set()

        while open_set:
            current_f, current = heapq.heappop(open_set)
            if current in visited:
                continue
            visited.add(current)
            open_set_nodes.discard(current)

            if current == goal:
                path_nodes = []
                tmp_node = goal
                while tmp_node in backtrack:
                    path_nodes.append(tmp_node)
                    tmp_node = backtrack[tmp_node]
                path_nodes.append(start)
                path_nodes.reverse()
                
                path_nodes  = self.backward_prune_path(path_nodes)
                path_edges  = [(path_nodes[i], path_nodes[i+1]) for i in range(len(path_nodes)-1)]
                path_length = sum(self._distance(path_nodes[i], path_nodes[i+1]) for i in range(len(path_nodes)-1))
                return path_nodes, path_edges, path_length, time.time() - start_time

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
    
    def backward_prune_path(self, path_nodes):
        if not path_nodes or len(path_nodes) < 3:
            return path_nodes
            
        pruned_path = [path_nodes[0]]
        current_idx = 0
        
        while current_idx < len(path_nodes) - 1:
            furthest_idx = current_idx + 1
            for i in range(len(path_nodes) - 1, current_idx, -1):
                if self._is_valid_edge(path_nodes[current_idx], path_nodes[i]):
                    furthest_idx = i
                    break
            
            pruned_path.append(path_nodes[furthest_idx])
            current_idx = furthest_idx
            
        return pruned_path  
    
def save_pdf_image(screen, filepath):
    """将pygame screen保存为PDF文件"""    
    # 使用PIL将PNG转换为PDF
    try:
        from PIL import Image
        w, h = screen.get_size()
        raw_data = pygame.image.tostring(screen, 'RGB')
        pil_image = Image.frombytes('RGB', (w, h), raw_data)
        pil_image.save(filepath, "PDF", quality=95)
    except Exception as e:
        print(f"保存PDF时出错: {e}")


class PRMRenderer:
    """使用Pygame渲染PRM - 支持保存PDF"""
    def __init__(self, grid_width, grid_height, cell_size=15, headless=False):
        self.grid_width = grid_width    
        self.grid_height = grid_height
        self.cell_size = cell_size
        self.screen_width = grid_width * cell_size
        self.screen_height = grid_height * cell_size
        self.headless = headless
        
        # 确保pygame已初始化
        if not pygame.get_init():
            pygame.init()
        
        if headless:
            # 无头模式：创建内存Surface
            self.screen = pygame.Surface((self.screen_width, self.screen_height))
        else:
            # 可视模式：创建实际窗口
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption("PRM Visualization")
            
        self.clock = pygame.time.Clock()

    def render(self, nodes, edges, obstacles, medial_axis_nodes=None, medial_axis_edges=None, medial_axis_paths=None, env="random", algorithm="beam"):
        """渲染PRM"""
        self.screen.fill((255, 255, 255))  # 背景

        # 绘制障碍物
        for obs in obstacles:
            x, y = obs
            rect = pygame.Rect(int(x * self.cell_size), int(y * self.cell_size), self.cell_size, self.cell_size)
            pygame.draw.rect(self.screen, (0, 49, 83), rect)  # 障碍物

        medial_axis_nodes = medial_axis_nodes or set()
        medial_axis_edges = medial_axis_edges or set()
        medial_axis_paths = medial_axis_paths or []

        # 普通边
        for edge in edges:
            a, b = edge
            key = edge if a <= b else (b, a)
            if key in medial_axis_edges:
                continue
            x1, y1 = a
            x2, y2 = b
            line_width=1
            if algorithm == "spars":
                if env == "random":
                    line_width = 2
                else:
                    line_width = 3
            pygame.draw.line(self.screen, (158, 176, 204) if algorithm == "beam" else (40, 100, 180),
                            (int(x1 * self.cell_size / ENV_CONFIG['cell_size']),
                            int(y1 * self.cell_size / ENV_CONFIG['cell_size'])),
                            (int(x2 * self.cell_size / ENV_CONFIG['cell_size']),
                            int(y2 * self.cell_size / ENV_CONFIG['cell_size'])), line_width)

        # 中轴骨架路径
        drawn_seg = set()
        for path in medial_axis_paths:
            if len(path) < 2:
                continue
            for i in range(len(path)-1):
                a, b = path[i], path[i+1]
                seg_key = (a, b) if a <= b else (b, a)
                if seg_key in drawn_seg:
                    continue
                drawn_seg.add(seg_key)
                x1, y1 = a
                x2, y2 = b
                pygame.draw.line(self.screen, (40, 100, 180),
                                (int(x1 * self.cell_size / ENV_CONFIG['cell_size']),
                                int(y1 * self.cell_size / ENV_CONFIG['cell_size'])),
                                (int(x2 * self.cell_size / ENV_CONFIG['cell_size']),
                                int(y2 * self.cell_size / ENV_CONFIG['cell_size'])), 3)

        # # 若没有路径则用边
        # if not medial_axis_paths:
        for a, b in medial_axis_edges:
            x1, y1 = a
            x2, y2 = b
            pygame.draw.line(self.screen, (40, 100, 180),
                            (int(x1 * self.cell_size / ENV_CONFIG['cell_size']),
                            int(y1 * self.cell_size / ENV_CONFIG['cell_size'])),
                            (int(x2 * self.cell_size / ENV_CONFIG['cell_size']),
                            int(y2 * self.cell_size / ENV_CONFIG['cell_size'])), 2 if env == "random" else 3)

        # 绘制节点
        for node in nodes:
            x, y = node
            pos = (int(x * self.cell_size / ENV_CONFIG['cell_size']),
                  int(y * self.cell_size / ENV_CONFIG['cell_size']))
            if node in medial_axis_nodes:
                color = (255, 91, 0)    # 红色: 中轴骨架节点
                if env == "maze" or env == "indoor":
                    radius = self.cell_size // 3
                else:
                    radius = self.cell_size // 4
            else:
                color = (255, 185, 153) if algorithm == "beam" else (255, 91, 0)  # 红色: 普通
                radius = self.cell_size // 3 if algorithm == "spars" and env != "random" else self.cell_size // 4
            pygame.draw.circle(self.screen, color, pos, radius)
        if not self.headless:
            pygame.display.flip()
        return self.screen

    def save_image(self, nodes, edges, obstacles, filepath, medial_axis_nodes=None, medial_axis_edges=None, medial_axis_paths=None, env="random", algorithm="beam"):
        """渲染并保存图像"""
        screen = self.render(nodes, edges, obstacles, medial_axis_nodes, medial_axis_edges, medial_axis_paths, env, algorithm)
        save_pdf_image(screen, filepath)
        print(f"图像已保存到: {filepath}")

    def run(self, nodes, edges, obstacles, medial_axis_nodes=None, medial_axis_edges=None, medial_axis_paths=None, env="random", algorithm="beam"):
        """运行渲染器（交互模式）"""
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            self.render(nodes, edges, obstacles, medial_axis_nodes, medial_axis_edges, medial_axis_paths, env, algorithm)
            self.clock.tick(60)
        pygame.quit()


if __name__ == "__main__":
    # 示例使用
    ENVIRONMENT_TYPE = "maze" # <-- 在这里切换环境！  
    np.random.seed(42)  # 固定随机种子以获得可重复结果
    import random
    random.seed(42)

    if ENVIRONMENT_TYPE == "maze":  
        # 迷宫环境特定配置  
        random.seed(114)
        ENV_CONFIG['gridnum_width'] = 49
        ENV_CONFIG['gridnum_height'] = 49
        grid_width = ENV_CONFIG['gridnum_width']  
        grid_height = ENV_CONFIG['gridnum_height']  
        obstacles = generate_maze_obstacles(grid_width, grid_height)  
        num_nodes = 400  
        connection_radius = 0.6  
    
    elif ENVIRONMENT_TYPE == "indoor":  
        # 室内环境特定配置  
        ENV_CONFIG['gridnum_width'] = 51  
        ENV_CONFIG['gridnum_height'] = 51  
        grid_width = ENV_CONFIG['gridnum_width']  
        grid_height = ENV_CONFIG['gridnum_height']  
        obstacles = generate_indoor_obstacles(grid_width, grid_height)  
        num_nodes = 350  
        connection_radius = 0.8  
    
    elif ENVIRONMENT_TYPE == "random":  
        # 原始的随机环境  
        random.seed(520)
        ENV_CONFIG['gridnum_width'] = 40  
        ENV_CONFIG['gridnum_height'] = 40  
        grid_width = ENV_CONFIG['gridnum_width']  
        grid_height = ENV_CONFIG['gridnum_height']  
        total_cells = grid_width * grid_height  
        num_obstacles = int(total_cells * 0.25)  # 25% 障碍物  
        obstacles = []  
        np.random.seed(43) # 使用不同的种子以获得不同的随机布局  
        while len(obstacles) < num_obstacles:  
            x = np.random.randint(0, grid_width)  
            y = np.random.randint(0, grid_height)  
            if (x, y) not in obstacles:  
                obstacles.append((x, y))  
        num_nodes = 320  
        connection_radius = 0.6  

    import time
    start_time = time.time()

    # 将默认测试算法改为 gsrm
    generator_name = "gsrm" # "delta" / "star" / "beam" / "spars"/ "gsrm" / "odrm"

    # 初始化特定于骨架图的变量，防止非 beam 算法在后续渲染时抛出未定义异常
    medial_axis_nodes = set()
    medial_axis_edges = set()
    medial_axis_paths = []

    # 统一使用 generator 作为实例变量名
    if generator_name == "delta":
        generator = DeltaPRM(grid_width, grid_height, obstacles, num_nodes=2000, delta_radius=0.15, connection_radius=1.6,max_failures=100)
        (nodes, edges) = generator.generate_prm()

    elif generator_name == "star":
        generator = PRMStar(grid_width, grid_height, obstacles, num_nodes=400, connection_radius=1)
        (nodes, edges) = generator.generate_prm()

    elif generator_name == "beam":
        if ENVIRONMENT_TYPE == "random":
            generator = BeamPRM(grid_width, grid_height, obstacles,
                                    num_nodes=1000,
                                    connection_radius=1.2,
                                    beam_angle_step_deg=3,
                                    beam_ray_step=0.08,
                                    min_connection_radius=0.3)
        elif ENVIRONMENT_TYPE == "maze":
            generator = BeamPRM(grid_width, grid_height, obstacles,
                                    num_nodes=1000,
                                    connection_radius=1.5,
                                    beam_angle_step_deg=25,
                                    beam_ray_step=0.2,
                                    min_connection_radius=0.4)
        elif ENVIRONMENT_TYPE == "indoor":
            generator = BeamPRM(grid_width, grid_height, obstacles,
                                    num_nodes=400,
                                    connection_radius=2,
                                    beam_angle_step_deg=25,
                                    beam_ray_step=0.2,
                                    min_connection_radius=0.4)
        else:
            generator = BeamPRM(grid_width, grid_height, obstacles,
                                    num_nodes=1000,
                                    connection_radius=1.5,
                                    beam_angle_step_deg=10,
                                    beam_ray_step=0.15,
                                    min_connection_radius=0.4)

        (nodes,
        edges,
        medial_axis_nodes,
        medial_axis_all_nodes,
        medial_axis_edges,
        medial_axis_paths) = generator.generate_prm()

    elif generator_name == "spars":
        generator = SPARS2(grid_width, grid_height, obstacles, num_nodes=3000, max_failures=200, delta=0.2, visibility_radius=1.6, connection_radius=1.2)
        (nodes, edges) = generator.generate_prm()
        
    elif generator_name == "gsrm":
        generator = GSRM(grid_width, grid_height, obstacles,iterations=10)
        (nodes, edges) = generator.generate_prm()

    end_time = time.time()
    print(f"PRM 生成耗时: {end_time - start_time:.2f} 秒")
    print(len(nodes), "nodes generated")
    print(len(edges), "edges generated")

    # 1. 新增：用一个列表收集所有规划成功的路径
    found_paths = []
    
    point_pairs = generator.generate_valid_point_pairs(num_pairs=5)
    for start, goal in point_pairs:
        print(f"Valid pair: Start {start} -> Goal {goal}")
        path_nodes, path_edges, path_length, search_time = generator.find_path(start, goal)
        
        if path_nodes and len(path_nodes) > 1:
            found_paths.append(path_nodes) # 记录成功找到的路径
            print(f"  Path length: {path_length:.2f}, Search time: {search_time:.2f}s, Nodes in path: {len(path_nodes)}, Edges in path: {len(path_edges)}")
            # print(f"  Path nodes: {path_nodes}")
            # print(f"  Path edges: {path_edges}")
        else:
            print("  Path search failed (no valid path found).")

    # 2. 合并：把搜索到的路径传给渲染参数进行加粗高亮渲染
    all_paths_to_draw = medial_axis_paths + found_paths

    # 3. 渲染：强制开启 headless=True，防止服务器挂机死锁
    renderer = PRMRenderer(grid_width, grid_height, headless=True)
    
    renderer.save_image(
        nodes, 
        edges, 
        obstacles, 
        medial_axis_nodes=medial_axis_nodes, 
        medial_axis_edges=medial_axis_edges, 
        medial_axis_paths=all_paths_to_draw, # 传入所有需要高亮的粗线路径
        env=ENVIRONMENT_TYPE, 
        algorithm=generator_name, 
        filepath=f"./pursuer_strategies/PRM/results/{ENVIRONMENT_TYPE}/{generator_name}_{ENVIRONMENT_TYPE}_path.pdf"
    )