import pygame
import numpy as np
from scipy.spatial import KDTree
import sys
sys.path.append('.')
from config import ENV_CONFIG
import math
import heapq  # 新增: 最短路
from map_generator import generate_indoor_obstacles, generate_maze_obstacles
from base_generator import BasePathPlanner
import time
import os
from collections import defaultdict

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
        其它逻辑保持不变。
        """
        ex, ey = explore_node
        max_range = math.hypot(self.grid_width * ENV_CONFIG['cell_size'],
                            self.grid_height * ENV_CONFIG['cell_size'])
        ray_step = self.beam_ray_step
        angles = self._get_allowed_angles(explore_node, step_deg=self.beam_angle_step_deg)
        distances = []
        for ang in angles:
            theta = math.radians(ang)
            dx, dy = math.cos(theta), math.sin(theta)
            t = 0.0
            last_valid = 0.0
            while t <= max_range:
                px = ex + t * dx
                py = ey + t * dy
                if not self._is_valid_position(px, py):
                    break
                last_valid = t
                t += ray_step
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
        新策略:
        1. 将所有节点按 clearance 从大到小排序
        2. 选取第一个作为中轴
        3. 依次尝试剩余节点: 若它与已选任一中轴节点具有 line-of-sight(无遮挡直线) 则跳过
        否则加入中轴集合
        4. 构建中轴边: 所有成对可视的中轴节点间的连接
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
        for node in ordered:
            # 与已选任意节点可视则跳过
            if any(self._has_line_of_sight(node, m) for m in selected):
                continue
            selected.append(node)

        self.medial_axis_nodes = set(selected)

        # 构建中轴边 (仅当可视)
        maa = set()
        sel_list = list(self.medial_axis_nodes)
        for i in range(len(sel_list)):
            for j in range(i + 1, len(sel_list)):
                a, b = sel_list[i], sel_list[j]
                if self._has_line_of_sight(a, b):
                    maa.add((a, b) if a <= b else (b, a))
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

class ClassicalPRM(BasePathPlanner):
    """经典概率路径图算法"""

    def __init__(self, grid_width, grid_height, obstacles, num_nodes=500, connection_radius=0.8, **kwargs):
        super().__init__(grid_width, grid_height, obstacles, **kwargs)
        self.num_nodes = num_nodes
        self.connection_radius = connection_radius
    
    def _find_k_nearest_neighbors(self, node, k=10):
        """找到节点的k个最近邻居"""
        if len(self.nodes) <= 1:
            return []
        
        distances = [(self._distance(node, other), other)
                    for other in self.nodes if other != node]
        distances.sort()
    
        return [neighbor for _, neighbor in distances[:min(k, len(distances))]]

    def _find_radius_neighbors(self, node):
        """找到连接半径内的所有邻居节点"""
        neighbors = []
        for other in self.nodes:
            if other != node and self._distance(node, other) <= self.connection_radius:
                neighbors.append(other)
        return neighbors

    def generate_prm(self):
        """生成经典PRM路径图"""
        print("开始生成Classical PRM...")
        start_time = time.time()
    
        self.nodes = []
        self.edges = []
    
        # 第一阶段：采样节点
        print(f"采样 {self.num_nodes} 个节点...")
        sampled = 0
        attempts = 0
        max_attempts = self.num_nodes * 10
    
        while sampled < self.num_nodes and attempts < max_attempts:
            attempts += 1
            sample = self._random_sample()
            if sample is not None:
                self.nodes.append(sample)
                sampled += 1
                if sampled % 50 == 0:
                    print(f"已采样 {sampled}/{self.num_nodes} 节点")
    
        print(f"实际采样到 {len(self.nodes)} 个有效节点")
    
        # 第二阶段：连接节点
        print("连接节点...")
        edge_attempts = 0
        successful_edges = 0
    
        for i, node in enumerate(self.nodes):
            # 使用半径连接策略
            neighbors = self._find_radius_neighbors(node)
        
            for neighbor in neighbors:
                # 避免重复连接
                if (node, neighbor) not in self.edges and (neighbor, node) not in self.edges:
                    edge_attempts += 1
                    if self._is_valid_edge(node, neighbor):
                        self.edges.append((node, neighbor))
                        successful_edges += 1
        
            if i % 50 == 0 and i > 0:
                print(f"已处理 {i}/{len(self.nodes)} 个节点，生成 {successful_edges} 条边")
    
        end_time = time.time()
        print(f"Classical PRM生成完成！")
        print(f"节点数: {len(self.nodes)}")
        print(f"边数: {len(self.edges)}")
        print(f"生成时间: {end_time - start_time:.2f} 秒")
        print(f"边连接成功率: {successful_edges/max(1,edge_attempts)*100:.1f}%")
    
        return self.nodes, self.edges

class PRMStar(BasePathPlanner):
    """
    渐进最优概率路径图算法 (PRM*)
    该算法使用一个随采样点数量动态调整的连接半径，以理论上保证路径的渐进最优性。
    """

    def __init__(self, grid_width, grid_height, obstacles, num_nodes=500, gamma_prm_star=15.0, **kwargs):
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
        self.rank = {}
        self.num_components = 0
    
    def make_set(self, x):
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
            self.num_components += 1
    
    def find(self, x):
        if x not in self.parent:
            return None
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # 路径压缩
        return self.parent[x]
    
    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)
        
        if root_x is None or root_y is None or root_x == root_y:
            return False
        
        # 按秩合并
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1
        
        self.num_components -= 1
        return True
    
    def connected(self, x, y):
        return self.find(x) == self.find(y)

class SPARS(BasePathPlanner):
    """
    优化的SPARS算法实现
    基于论文: "Sparse Roadmap Spanners" by Dobson et al. (2013)
    """
    
    def __init__(self, grid_width, grid_height, obstacles, 
                 max_samples=5000, target_guards=500, 
                 delta=0.8, stretch_factor=3.0, **kwargs):
        super().__init__(grid_width, grid_height, obstacles, **kwargs)
        
        self.max_samples = max_samples
        self.target_guards = target_guards
        self.delta = delta  # 可见范围D
        self.stretch_factor = stretch_factor
        self.dense_radius = delta * 0.3  # 稠密图连接半径d
        
        # 数据结构
        self.dense_nodes = []
        self.dense_adj = defaultdict(set)
        self.dense_kdtree = None
        
        self.sparse_nodes = []
        self.sparse_adj = defaultdict(set)  
        self.sparse_kdtree = None
        
        # 索引和映射
        self.sparse_node_to_idx = {}
        self.representatives = {}  # dense_node -> sparse_node
        self.guard_types = {}
        self.components = UnionFind()
        
        # 优化参数
        self.kdtree_rebuild_threshold = 100  # 提高阈值
        self.nodes_since_rebuild = 0
        self.batch_size = 10  # 批处理大小
        
        # 统计
        self.stats = {
            'coverage_guards': 0, 'connectivity_guards': 0,
            'interface_guards': 0, 'quality_guards': 0,
            'total_samples': 0, 'failed_samples': 0
        }
    
    def _maybe_rebuild_kdtrees(self):
        """优化的KDTree重建"""
        if self.nodes_since_rebuild >= self.kdtree_rebuild_threshold:
            if len(self.dense_nodes) > 10:
                self.dense_kdtree = KDTree(np.array(self.dense_nodes))
            if len(self.sparse_nodes) > 5:
                self.sparse_kdtree = KDTree(np.array(self.sparse_nodes))
                self.sparse_node_to_idx = {node: i for i, node in enumerate(self.sparse_nodes)}
            self.nodes_since_rebuild = 0
    
    def _add_to_dense(self, q):
        """优化的稠密图添加"""
        self.dense_nodes.append(q)
        self.nodes_since_rebuild += 1
        
        # 快速邻居查找
        if self.dense_kdtree is not None and len(self.dense_nodes) > 1:
            try:
                indices = self.dense_kdtree.query_ball_point(q, self.dense_radius)
                for idx in indices[:20]:  # 限制邻居数量
                    if idx < len(self.dense_nodes) - 1:  # 排除自己
                        neighbor = self.dense_nodes[idx]
                        if self._distance(q, neighbor) <= self.dense_radius:
                            if self._is_valid_edge(q, neighbor, num_samples=5):  # 减少采样
                                self.dense_adj[q].add(neighbor)
                                self.dense_adj[neighbor].add(q)
            except:
                # KDTree查询失败，重建
                self.nodes_since_rebuild = self.kdtree_rebuild_threshold
        else:
            # 后备：线性搜索（限制数量）
            for neighbor in self.dense_nodes[-20:]:
                if neighbor != q and self._distance(q, neighbor) <= self.dense_radius:
                    if self._is_valid_edge(q, neighbor, num_samples=5):
                        self.dense_adj[q].add(neighbor)
                        self.dense_adj[neighbor].add(q)
    
    def _add_sparse_guard(self, q, guard_type):
        """添加稀疏守卫"""
        if q in self.sparse_nodes:
            return
            
        self.sparse_nodes.append(q)
        self.guard_types[q] = guard_type
        self.stats[f'{guard_type}_guards'] += 1
        self.components.make_set(q)
        self.nodes_since_rebuild += 1
        
        # 连接到邻居
        self._connect_sparse_neighbors(q)
    
    def _connect_sparse_neighbors(self, q):
        """优化的稀疏邻居连接"""
        if not self.sparse_nodes:
            return
            
        # 使用更小的搜索半径
        search_radius = min(1.5 * self.delta, 2.0)
        
        if self.sparse_kdtree is not None and len(self.sparse_nodes) > 1:
            try:
                indices = self.sparse_kdtree.query_ball_point(q, search_radius)
                for idx in indices[:15]:  # 限制连接数
                    if idx < len(self.sparse_nodes) - 1:
                        neighbor = self.sparse_nodes[idx]
                        dist = self._distance(q, neighbor)
                        if dist <= self.delta * 2 and neighbor not in self.sparse_adj[q]:
                            if self._is_valid_edge(q, neighbor, num_samples=8):
                                self.sparse_adj[q].add(neighbor)
                                self.sparse_adj[neighbor].add(q)
                                self.components.union(q, neighbor)
            except:
                self.nodes_since_rebuild = self.kdtree_rebuild_threshold
        else:
            # 后备方案
            for neighbor in self.sparse_nodes[-15:]:
                if neighbor != q:
                    dist = self._distance(q, neighbor)
                    if dist <= self.delta * 2:
                        if self._is_valid_edge(q, neighbor, num_samples=8):
                            self.sparse_adj[q].add(neighbor)
                            self.sparse_adj[neighbor].add(q)
                            self.components.union(q, neighbor)
    
    def _find_visible_guards(self, q):
        """优化的可见守卫查找"""
        if not self.sparse_nodes:
            return []
        
        visible_guards = []
        
        if self.sparse_kdtree is not None and len(self.sparse_nodes) > 5:
            try:
                indices = self.sparse_kdtree.query_ball_point(q, self.delta * 1.1)
                for idx in indices[:10]:  # 限制检查数量
                    if idx < len(self.sparse_nodes):
                        guard = self.sparse_nodes[idx]
                        dist = self._distance(q, guard)
                        if dist <= self.delta:
                            if self._is_valid_edge(q, guard, num_samples=6):
                                visible_guards.append(guard)
            except:
                self.nodes_since_rebuild = self.kdtree_rebuild_threshold
        else:
            # 后备方案
            for guard in self.sparse_nodes[-10:]:
                dist = self._distance(q, guard)
                if dist <= self.delta:
                    if self._is_valid_edge(q, guard, num_samples=6):
                        visible_guards.append(guard)
        
        return visible_guards
    
    def _check_connectivity_guard(self, q, visible_guards):
        """检查连通性守卫"""
        if len(visible_guards) < 2:
            return False
        
        # 检查是否连接不同组件
        components_set = set()
        for guard in visible_guards:
            root = self.components.find(guard)
            if root:
                components_set.add(root)
        
        return len(components_set) >= 2
    
    def _check_interface_guard(self, q):
        """简化的接口守卫检查"""
        visible_guards = self._find_visible_guards(q)
        if len(visible_guards) != 1:
            return False, None
        
        main_guard = visible_guards[0]
        
        # 检查稠密邻居
        dense_neighbors = list(self.dense_adj.get(q, set()))[:5]  # 限制检查数
        for neighbor in dense_neighbors:
            neighbor_guards = self._find_visible_guards(neighbor)
            if len(neighbor_guards) == 1:
                neighbor_guard = neighbor_guards[0]
                if (neighbor_guard != main_guard and 
                    neighbor_guard not in self.sparse_adj.get(main_guard, set())):
                    return True, (main_guard, neighbor_guard)
        
        return False, None
    
    def _check_quality_guard(self, q):
        """简化的质量守卫检查"""
        visible_guards = self._find_visible_guards(q)
        if len(visible_guards) < 1:
            return False
        
        # 简化检查逻辑
        dense_neighbors = list(self.dense_adj.get(q, set()))
        if len(dense_neighbors) >= 3:
            guard_count = set()
            for neighbor in dense_neighbors[:3]:
                n_guards = self._find_visible_guards(neighbor)
                for guard in n_guards[:1]:  # 只取第一个
                    guard_count.add(guard)
            return len(guard_count) >= 2
        
        return False
    
    def generate_prm(self):
        """优化的SPARS路径图生成"""
        print(f"开始生成优化SPARS，目标：{self.target_guards}守卫")
        start_time = time.time()
        
        consecutive_failures = 0
        max_failures = 1500  # 减少最大失败次数
        batch_count = 0
        
        while (self.stats['total_samples'] < self.max_samples and 
               len(self.sparse_nodes) < self.target_guards and
               consecutive_failures < max_failures):
            
            # 批处理重建KDTree
            if batch_count % self.batch_size == 0:
                self._maybe_rebuild_kdtrees()
            
            # 采样
            q = self._random_sample()
            if q is None:
                consecutive_failures += 1
                self.stats['failed_samples'] += 1
                continue
            
            self.stats['total_samples'] += 1
            batch_count += 1
            
            # 添加到稠密图
            self._add_to_dense(q)
            
            # 寻找可见守卫
            visible_guards = self._find_visible_guards(q)
            
            # 情况1: Coverage
            if len(visible_guards) == 0:
                self._add_sparse_guard(q, 'coverage')
                self.representatives[q] = q
                consecutive_failures = 0
                continue
            
            # 情况2: Connectivity  
            if self._check_connectivity_guard(q, visible_guards):
                self._add_sparse_guard(q, 'connectivity')
                # 连接到可见守卫
                for guard in visible_guards:
                    if guard not in self.sparse_adj[q]:
                        self.sparse_adj[q].add(guard)
                        self.sparse_adj[guard].add(q)
                        self.components.union(q, guard)
                self.representatives[q] = q
                consecutive_failures = 0
                continue
            
            # 分配最近代表
            closest_guard = min(visible_guards, key=lambda g: self._distance(q, g))
            self.representatives[q] = closest_guard
            
            # 情况3: Interface (简化版)
            is_interface, interface_info = self._check_interface_guard(q)
            if is_interface and interface_info:
                guard1, guard2 = interface_info
                self._add_sparse_guard(q, 'interface')
                # 尝试直接连接两个守卫
                if self._distance(guard1, guard2) <= 2 * self.delta:
                    if self._is_valid_edge(guard1, guard2, num_samples=10):
                        self.sparse_adj[guard1].add(guard2)
                        self.sparse_adj[guard2].add(guard1)
                        self.components.union(guard1, guard2)
                self.representatives[q] = q
                consecutive_failures = 0
                continue
            
            # 情况4: Quality (简化版)
            if self._check_quality_guard(q):
                self._add_sparse_guard(q, 'quality')
                # 连接到最近守卫
                if closest_guard not in self.sparse_adj[q]:
                    self.sparse_adj[q].add(closest_guard)
                    self.sparse_adj[closest_guard].add(q)
                    self.components.union(q, closest_guard)
                self.representatives[q] = q
                consecutive_failures = 0
                continue
            
            consecutive_failures += 1
        
        # 构建边列表
        edges = []
        for node in self.sparse_nodes:
            for neighbor in self.sparse_adj[node]:
                if node < neighbor:
                    edges.append((node, neighbor))
        
        self.nodes = self.sparse_nodes
        self.edges = edges
        
        end_time = time.time()
        self.stats['generation_time'] = end_time - start_time
        self.stats['final_nodes'] = len(self.sparse_nodes)
        self.stats['final_edges'] = len(edges)
        
        print(f"优化SPARS完成：{len(self.sparse_nodes)}守卫, {len(edges)}边, "
              f"用时{self.stats['generation_time']:.2f}s")
        print(f"守卫统计：C={self.stats['coverage_guards']}, "
              f"N={self.stats['connectivity_guards']}, "
              f"I={self.stats['interface_guards']}, "
              f"Q={self.stats['quality_guards']}")
        print(f"组件数：{self.components.num_components}")
        
        return self.nodes, self.edges, self.guard_types, self.stats


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

    def render(self, nodes, edges, obstacles, medial_axis_nodes=None, medial_axis_edges=None, medial_axis_paths=None):
        """渲染PRM"""
        self.screen.fill((229, 221, 215))  # 背景

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
            pygame.draw.line(self.screen, (40, 100, 180),
                            (int(x1 * self.cell_size / ENV_CONFIG['cell_size']),
                            int(y1 * self.cell_size / ENV_CONFIG['cell_size'])),
                            (int(x2 * self.cell_size / ENV_CONFIG['cell_size']),
                            int(y2 * self.cell_size / ENV_CONFIG['cell_size'])), 1)

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
                pygame.draw.line(self.screen, (0, 140, 0),
                                (int(x1 * self.cell_size / ENV_CONFIG['cell_size']),
                                int(y1 * self.cell_size / ENV_CONFIG['cell_size'])),
                                (int(x2 * self.cell_size / ENV_CONFIG['cell_size']),
                                int(y2 * self.cell_size / ENV_CONFIG['cell_size'])), 3)

        # 若没有路径则用边
        if not medial_axis_paths:
            for a, b in medial_axis_edges:
                x1, y1 = a
                x2, y2 = b
                pygame.draw.line(self.screen, (0, 140, 0),
                                (int(x1 * self.cell_size / ENV_CONFIG['cell_size']),
                                int(y1 * self.cell_size / ENV_CONFIG['cell_size'])),
                                (int(x2 * self.cell_size / ENV_CONFIG['cell_size']),
                                int(y2 * self.cell_size / ENV_CONFIG['cell_size'])), 3)

        # 绘制节点
        for node in nodes:
            x, y = node
            pos = (int(x * self.cell_size / ENV_CONFIG['cell_size']),
                  int(y * self.cell_size / ENV_CONFIG['cell_size']))
            if node in medial_axis_nodes:
                color = (0, 180, 0)    # 绿色: 中轴
                radius = self.cell_size // 3
            else:
                color = (255, 91, 0)    # 红色: 普通
                radius = self.cell_size // 4
            pygame.draw.circle(self.screen, color, pos, radius)
        if not self.headless:
            pygame.display.flip()
        return self.screen

    def save_image(self, nodes, edges, obstacles, filepath, medial_axis_nodes=None, medial_axis_edges=None, medial_axis_paths=None):
        """渲染并保存图像"""
        screen = self.render(nodes, edges, obstacles, medial_axis_nodes, medial_axis_edges, medial_axis_paths)
        save_pdf_image(screen, filepath)
        print(f"图像已保存到: {filepath}")

    def run(self, nodes, edges, obstacles, medial_axis_nodes=None, medial_axis_edges=None, medial_axis_paths=None):
        """运行渲染器（交互模式）"""
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            self.render(nodes, edges, obstacles, medial_axis_nodes, medial_axis_edges, medial_axis_paths)
            self.clock.tick(30)
        pygame.quit()


if __name__ == "__main__":
    # 示例使用
    ENVIRONMENT_TYPE = "maze" # <-- 在这里切换环境！  
    np.random.seed(42)  # 固定随机种子以获得可重复结果
    import random
    random.seed(42)

    if ENVIRONMENT_TYPE == "maze":  
        # 迷宫环境特定配置  
        ENV_CONFIG['gridnum_width'] = 49
        ENV_CONFIG['gridnum_height'] = 49
        grid_width = ENV_CONFIG['gridnum_width']  
        grid_height = ENV_CONFIG['gridnum_height']  
        obstacles = generate_maze_obstacles(grid_width, grid_height)  
        num_nodes = 400  
        connection_radius = 0.6  
    
    elif ENVIRONMENT_TYPE == "indoor":  
        # 室内环境特定配置  
        ENV_CONFIG['gridnum_width'] = 50  
        ENV_CONFIG['gridnum_height'] = 50  
        grid_width = ENV_CONFIG['gridnum_width']  
        grid_height = ENV_CONFIG['gridnum_height']  
        obstacles = generate_indoor_obstacles(grid_width, grid_height)  
        num_nodes = 350  
        connection_radius = 0.8  
    
    elif ENVIRONMENT_TYPE == "random":  
        # 原始的随机环境  
        ENV_CONFIG['gridnum_width'] = 40  
        ENV_CONFIG['gridnum_height'] = 30  
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

    generator_name = "spars" # "classical" / "star" / "beam" / "spars" / "fmt" / "bit"

    if generator_name == "classical":
        prm_generator = ClassicalPRM(grid_width, grid_height, obstacles, num_nodes=1000, connection_radius=0.6)
        (nodes, edges) = prm_generator.generate_prm()

    elif generator_name == "star":
        prm_generator = PRMStar(grid_width, grid_height, obstacles, num_nodes=1000, connection_radius=1)
        (nodes, edges) = prm_generator.generate_prm()

    elif generator_name == "beam":
        #maze:550,1.5,25,0.2,0.4
        #indoor:380,2,25(30),0.2,0.4
        #random:700,1.2,3,0.08,0.3
        # 可按需传入 beam_angle_step_deg / beam_ray_step 覆盖默认:
        if ENVIRONMENT_TYPE == "random":
            prm_generator = BeamPRM(grid_width, grid_height, obstacles,
                                    num_nodes=1000,
                                    connection_radius=1.2,
                                    beam_angle_step_deg=3,
                                    beam_ray_step=0.08,
                                    min_connection_radius=0.3)
        elif ENVIRONMENT_TYPE == "maze":
            prm_generator = BeamPRM(grid_width, grid_height, obstacles,
                                    num_nodes=1000,
                                    connection_radius=1.5,
                                    beam_angle_step_deg=25,
                                    beam_ray_step=0.2,
                                    min_connection_radius=0.4)
        elif ENVIRONMENT_TYPE == "indoor":
            prm_generator = BeamPRM(grid_width, grid_height, obstacles,
                                    num_nodes=1000,
                                    connection_radius=2,
                                    beam_angle_step_deg=25,
                                    beam_ray_step=0.2,
                                    min_connection_radius=0.4)
        else:
            prm_generator = BeamPRM(grid_width, grid_height, obstacles,
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
        medial_axis_paths) = prm_generator.generate_prm()

    elif generator_name == "spars":
        spars = SPARS(grid_width, grid_height, obstacles,
                    max_samples=6000, target_guards=800,
                    delta=connection_radius, stretch_factor=5)
        (nodes, edges, guard_types, stats) = spars.generate_prm()

    end_time = time.time()
    print(f"PRM 生成耗时: {end_time - start_time:.2f} 秒")
    print(len(nodes), "nodes generated")
    print(len(edges), "edges generated")
    # print(len(medial_axis_nodes), "selected medial axis nodes")
    # print(len(medial_axis_all_nodes), "all medial axis nodes (including edge endpoints)")
    # print(len(medial_axis_edges), "medial axis edges")
    # print(len(medial_axis_paths), "medial axis paths")
    renderer = PRMRenderer(grid_width, grid_height)
    # 仍可用原集合渲染(不需要 all_nodes 渲染则保持不变)
    #renderer.run(nodes, edges, obstacles, medial_axis_nodes, medial_axis_edges, medial_axis_paths)
    renderer.run(nodes, edges, obstacles)

