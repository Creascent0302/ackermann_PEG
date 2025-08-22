import pygame
import numpy as np
from scipy.spatial import KDTree
import sys
sys.path.append('.')
from config import ENV_CONFIG
import math

class PRMGenerator:
    """概率路图生成器（精简：仅节点 + 边，去除守卫/连接器分类）"""
    def __init__(self, grid_width, grid_height, obstacles, num_nodes=100, connection_radius=0.8):
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.obstacles = obstacles
        self.num_nodes = num_nodes 
        self.connection_radius = connection_radius
        self.min_connection_radius = 0.24
        self.nodes = []          # 统一节点列表
        self.edges = []
        self.collision_radius = 2 * ENV_CONFIG['agent_collision_radius'] / 3

    def _is_valid_position(self, x, y):
        """检查位置是否有效（不在障碍物附近且在边界内）"""

        #dr = self.collision_radius
        dr = 0.015
        check_list = [(x+dr,y),(x+dr/np.sqrt(2),y+dr/np.sqrt(2)),
                      (x+dr/np.sqrt(2),y-dr/np.sqrt(2)),(x-dr,y),
                        (x-dr/np.sqrt(2),y+dr/np.sqrt(2)),(x-dr/np.sqrt(2),y-dr/np.sqrt(2)),
                        (x,y+dr),(x,y-dr)]
        collision_check_x, collision_check_y = zip(*check_list)

        for gx in collision_check_x:
            for gy in collision_check_y:
                gx1, gy1 = gx / ENV_CONFIG['cell_size'], gy / ENV_CONFIG['cell_size']
                if gx1 < 0 or gx1 >= self.grid_width or gy1 < 0 or gy1 >= self.grid_height:
                    return False
                if (int(gx1), int(gy1)) in self.obstacles:
                    return False
        return True

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

    def beam_sampling(self, explore_node):
        """
        从 explore_node 全向发射光束：
        - 计算每 3° 光束最大可行距离
        - 遍历相邻光束对，若差值 > min(short/5,1) 视为临界
        - 为每个临界对生成两个候选节点：
            1) 长光束方向距离 (short+long)/2
            2) 短光束方向距离 short/2（short>0 时）
        返回所有新候选节点列表（可能为空，已去重）
        """
        ex, ey = explore_node
        max_range = math.hypot(self.grid_width * ENV_CONFIG['cell_size'],
                               self.grid_height * ENV_CONFIG['cell_size'])
        step = 0.08
        angles = np.arange(0, 360, 2)
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
                t += step
            distances.append(last_valid)

        new_nodes = []
        seen = set()
        n = len(distances)
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

    def generate_nodes(self):
        """
        批量光束采样 + 探索队列：
        - 使用 frontier 维护待探索节点列表（FIFO）
        - 每次从 frontier 取出一个节点做一次全向光束采样
        - 新节点通过筛选后加入 self.nodes 与 frontier，并立即尝试建立边
        """
        self.nodes, self.edges = [], []
        # 1. 初始化起点
        while True:
            start_x = np.random.uniform(0, self.grid_width * ENV_CONFIG['cell_size'])
            start_y = np.random.uniform(0, self.grid_height * ENV_CONFIG['cell_size'])
            if self._is_valid_position(start_x, start_y):
                start_node = (start_x, start_y)
                self.nodes.append(start_node)
                break
        frontier = [start_node]

        # 2. 探索循环
        max_idle_expansions = self.num_nodes * 3  # 没有新增节点的探索上限
        idle_expansions = 0

        while len(self.nodes) < self.num_nodes and frontier and idle_expansions < max_idle_expansions:
            explore_node = self.select_explore_node(frontier)
            if explore_node is None:
                break
            print(f"探索节点 {explore_node} | 当前已生成 {len(self.nodes)}/{self.num_nodes}")
            candidates = self.beam_sampling(explore_node)

            added_this_round = 0
            for candidate in candidates:
                if len(self.nodes) >= self.num_nodes:
                    break
                if self.check_near_radius(candidate):
                    continue
                if not self._is_valid_position(*candidate):
                    continue

                # 添加节点
                self.nodes.append(candidate)
                frontier.append(candidate)  # 加入待探索
                added_this_round += 1

                # 建立边
                for other in self.nodes[:-1]:
                    dist = np.linalg.norm(np.array(candidate) - np.array(other))
                    if dist <= self.connection_radius\
                            and dist > self.min_connection_radius\
                            and self._is_valid_edge(candidate, other):
                        if (candidate, other) not in self.edges and (other, candidate) not in self.edges:
                            self.edges.append((candidate, other))

            if added_this_round == 0:
                idle_expansions += 1
            else:
                idle_expansions = 0  # 有新增则重置

        if len(self.nodes) < self.num_nodes and not frontier:
            print("frontier 为空，无法继续扩展。")
        if idle_expansions >= max_idle_expansions:
            print("多次探索无新增节点，提前终止。")

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
        """可选：重建所有边"""
        self.edges = []
        for i, a in enumerate(self.nodes):
            for j, b in enumerate(self.nodes):
                if i >= j:
                    continue
                dist = np.linalg.norm(np.array(a) - np.array(b))
                if dist <= self.connection_radius * 0.8 \
                and self._is_valid_edge(a, b):
                        #and dist > self.min_connection_radius * 1.2 \
                        
                    self.edges.append((a, b))

    def generate_prm(self):
        self.generate_nodes()
        return self.nodes, self.edges

    def _is_valid_edge(self, node1, node2):
        """
        检查两节点间连线是否可行：对线段做等距采样，若任一点无效则整条边不可用。
        """
        x1, y1 = node1
        x2, y2 = node2
        length = math.hypot(x2 - x1, y2 - y1)
        num_points = max(1, int(length * 20))  # 采样密度：每单位长度约10点
        for i in range(num_points + 1):
            t = i / num_points
            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)
            if not self._is_valid_position(x, y):
                return False
        return True


class PRMRenderer:
    """使用Pygame渲染PRM"""
    def __init__(self, grid_width, grid_height, cell_size=20):
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.cell_size = cell_size
        self.screen_width = grid_width * cell_size
        self.screen_height = grid_height * cell_size
        pygame.init()
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("PRM Visualization")
        self.clock = pygame.time.Clock()

    def render(self, nodes, edges, obstacles):
        """渲染PRM"""
        self.screen.fill((255, 255, 255))  # 白色背景

        # 绘制障碍物
        for obs in obstacles:
            x, y = obs
            rect = pygame.Rect(int(x * self.cell_size), int(y * self.cell_size), self.cell_size, self.cell_size)
            pygame.draw.rect(self.screen, (0, 0, 0), rect)  # 黑色障碍物

        # 绘制边
        for edge in edges:
            node1, node2 = edge
            x1, y1 = node1
            x2, y2 = node2
            pygame.draw.line(self.screen, (0, 0, 255), 
                             (int(x1 * self.cell_size / ENV_CONFIG['cell_size']), 
                              int(y1 * self.cell_size / ENV_CONFIG['cell_size'])),
                             (int(x2 * self.cell_size / ENV_CONFIG['cell_size']),
                              int(y2 * self.cell_size / ENV_CONFIG['cell_size'])), 1)

        # 绘制守卫节点
        for node in nodes:
            x, y = node
            pygame.draw.circle(self.screen, (255, 0, 0), 
                               (x * self.cell_size / ENV_CONFIG['cell_size'], 
                                y * self.cell_size / ENV_CONFIG['cell_size']),
                               self.cell_size // 4)
        pygame.display.flip()

    def run(self, nodes, edges, obstacles):
        """运行渲染器"""
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            self.render(nodes, edges, obstacles)
            self.clock.tick(30)
        pygame.quit()


if __name__ == "__main__":
    # 示例使用
    grid_width = ENV_CONFIG['gridnum_width']
    grid_height = ENV_CONFIG['gridnum_height']
    cell_size = ENV_CONFIG['cell_size']
    total_cells = grid_width * grid_height
    num_obstacles = int(total_cells * 0.2)  # 20% 障碍物
    obstacles = []
    while len(obstacles) < num_obstacles:
        x = np.random.randint(0, grid_width)
        y = np.random.randint(0, grid_height)
        obstacles.append((x, y))
    prm_generator = PRMGenerator(grid_width, grid_height, obstacles, num_nodes=300, connection_radius=0.8)
    nodes, edges = prm_generator.generate_prm()
    print(len(nodes), "nodes generated")
    print(len(edges), "edges generated")
    renderer = PRMRenderer(grid_width, grid_height)
    renderer.run(nodes, edges, obstacles)
