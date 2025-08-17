import pygame
import numpy as np
from scipy.spatial import KDTree
import sys
sys.path.append('.')
from config import ENV_CONFIG

class PRMGenerator:
    """概率路图生成器，支持守卫节点和连接器节点"""
    def __init__(self, grid_width, grid_height, obstacles, num_nodes=100, connection_radius=1.0):
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.obstacles = obstacles  # 障碍物为整数坐标
        self.num_nodes = num_nodes
        self.connection_radius = connection_radius
        self.min_connection_radius = 0.2
        self.guards = []  # 守卫节点
        self.connectors = []  # 连接器节点
        self.edges = []
        self.collision_radius = 2 * ENV_CONFIG['agent_collision_radius'] / 3

    def _is_valid_position(self, x, y):
        """检查位置是否有效（不在障碍物附近且在边界内）"""

        dr = self.collision_radius
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


    def _is_valid_edge(self, start, end):
        """检查边缘是否有效（不与障碍物相交）"""
        # 使用Bresenham算法检查路径上的每个点
        x0, y0 = start
        x1, y1 = end
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        while True:
            if (int(x0), int(y0)) in self.obstacles:
                return False
            if x0 == x1 and y0 == y1:
                break
            err2 = err * 2
            if err2 > -dy:
                err -= dy
                x0 += sx
            if err2 < dx:
                err += dx
                y0 += sy
        return True

    def haltonian_sampling(self):
        """使用Halton序列生成随机位置（浮点数，保留两位小数）"""
        from scipy.stats import qmc
        sampler = qmc.Halton(d=2, scramble=True)
        sample = sampler.random(1)[0]
        x = round(sample[0] * self.grid_width * ENV_CONFIG['cell_size'], 2)
        y = round(sample[1] * self.grid_height * ENV_CONFIG['cell_size'], 2)
        return (x, y)

    def check_near_radius(self, position):
        """检查位置是否在连接器节点附近"""
        if position[0] == None:
            return True
        for node in self.connectors:
            if np.linalg.norm(np.array(position) - np.array(node)) < self.min_connection_radius:
                return True
        for node in self.guards:
            if np.linalg.norm(np.array(position) - np.array(node)) < self.min_connection_radius:
                return True
        return False
    def bridge_sampling(self):
        """使用桥接采样生成位置（整数坐标）"""
        x1 = np.random.randint(0, self.grid_width * ENV_CONFIG['cell_size'])
        y1 = np.random.randint(0, self.grid_height * ENV_CONFIG['cell_size'])
        x2 = np.random.randint(0, self.grid_width * ENV_CONFIG['cell_size'])
        y2 = np.random.randint(0, self.grid_height * ENV_CONFIG['cell_size'])
        if self._is_valid_position(x1, y1)== False and self._is_valid_position(x2, y2)==False:
            return ((x1+x2)/2, (y1+y2)/2)
        else:
            return (None, None)

    def random_sampling(self):  
        """随机采样生成位置（整数坐标）"""

        x = np.random.randint(0, self.grid_width* ENV_CONFIG['cell_size'])
        y = np.random.randint(0, self.grid_height* ENV_CONFIG['cell_size'])
        return (x, y)

    def generate_nodes(self):
        """生成守卫节点和连接器节点"""
        self.guards = []
        self.connectors = []
        current_number = 0
        # 生成节点
        attempt_count = 0
        while current_number < self.num_nodes:
            if current_number % 10 == 0:
                print(f"生成节点: {current_number}/{self.num_nodes}")
            if attempt_count < self.num_nodes / 3:
                x, y = self.haltonian_sampling()  # 使用Halton序列生成位置 
            elif attempt_count < 1.22 * self.num_nodes :
                x, y = self.bridge_sampling()  # 使用桥接采样生成位置
            else:
                x, y = self.random_sampling()  # 使用随机采样生成位置

            if self.check_near_radius((x, y)):
                continue
            if self._is_valid_position(x, y):
                # 检查是否满足守卫节点条件
                guard_count = sum((self._is_valid_edge((x, y), guard) \
                                   and 0.8*self.connection_radius >= np.linalg.norm(np.array((x, y)) - np.array(guard)) \
                                    for guard in self.guards))
                if guard_count == 0:
                    self.guards.append((x, y))
                    current_number += 1
                    attempt_count = 0  # 重置尝试计数
                elif guard_count >= 2:  # 如果已有两个守卫节点连接到此位置，则认为是连接器节点
                    # 检查是否满足连接器节点条件
                    self.connectors.append((x, y))
                    current_number += 1
                    attempt_count = 0  # 重置尝试计数
            attempt_count += 1
            if attempt_count > self.num_nodes*1.5:  # 避免无限循环
                print("尝试次数过多，停止生成")
                break
            if current_number >= self.num_nodes:
                break

    def connect_edges(self):
        """连接守卫节点和连接器节点"""
        self.edges = []
        all_nodes = self.guards + self.connectors

        if not all_nodes:
            raise ValueError("请先生成节点！")
        
        # 使用简单的距离计算查找邻居
        for i, node in enumerate(all_nodes):
            for j, neighbor in enumerate(all_nodes):
                if i != j:  # 排除自身
                    distance = np.linalg.norm(np.array(node) - np.array(neighbor))
                    if distance <= self.connection_radius*0.8 \
                        and self._is_valid_edge(node, neighbor)\
                            and distance > self.min_connection_radius*1.2:
                        self.edges.append((node, neighbor))

    def _is_valid_edge(self, node1, node2):
        """检查边是否有效（不穿过障碍物）"""
        x1, y1 = node1
        x2, y2 = node2
        num_points = max(1, int(np.linalg.norm([x2 - x1, y2 - y1]) * 10))  # 插值点数
        for i in range(num_points + 1):
            t = i / num_points
            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)
            if not self._is_valid_position(x, y):
                return False
        return True

    def generate_prm(self):
        """生成概率路图"""
        self.generate_nodes()
        self.connect_edges()
        self.nodes = self.guards + self.connectors
        return self.nodes, self.edges


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
    prm_generator = PRMGenerator(grid_width, grid_height, obstacles, num_nodes=200, connection_radius=0.8)
    nodes, edges = prm_generator.generate_prm()
    print(len(nodes), "nodes generated")
    print(len(edges), "edges generated")
    renderer = PRMRenderer(grid_width, grid_height)
    renderer.run(nodes, edges, obstacles)