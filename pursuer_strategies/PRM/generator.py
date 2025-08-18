import pygame
import numpy as np
from scipy.spatial import KDTree
import sys
sys.path.append('.')
from config import ENV_CONFIG


# 定义一个节点类，储存更多节点信息方便计算
class SunNode:
    def __init__(self, point):
        self.connections = [] #储存(角度，半径)数据对，角度取水平向右为 0 度，逆时针为正方向
        self.position = point  # (x, y) 坐标
     
    def add_connection(self, angle, radius):
        self.connections.append((angle, radius))            

    def get_angle_iter(self):
        iter_range = np.arange(0, 2 * np.pi, ENV_CONFIG["angle_iter_step"])
        is_forbidden = np.zeros_like(iter_range, dtype=bool)
        boundary_high = ENV_CONFIG["angle_of_repulsion_high"]
        boundary_low = ENV_CONFIG["angle_of_repulsion_low"]

        if not self.connections:
            return iter_range  # 返回一个默认的角度迭代器
        else:
            for edge in self.connections:
                angle = edge[0]

                # 大于斥力引入长度边界，不做处理
                if edge[1] > ENV_CONFIG["radius_boundary_of_repulsion"]:
                    continue
                
                # 在斥力两级边界之间，做遍历范围限制
                elif edge[1] > ENV_CONFIG["radius_boundary_of_repulsion_low"] and edge[1] <= ENV_CONFIG["radius_boundary_of_repulsion_high"]:
                    start = angle - boundary_high
                    end = angle + boundary_high
                
                elif edge[1] > 0 and edge[1] <= ENV_CONFIG["radius_boundary_of_repulsion_low"]:
                    start = angle - boundary_low
                    end = angle + boundary_low

                if start < 0:
                    is_forbidden |= (iter_range >= start + 2 * np.pi) | (iter_range < end)
                elif end > 2 * np.pi:
                    is_forbidden |= (iter_range >= start) | (iter_range < end - 2 * np.pi)
                else:
                    is_forbidden |= (iter_range < end) & (iter_range >= start)
                
            valid_iter_range = iter_range[~is_forbidden]
            return valid_iter_range

                    

class PRMGenerator:
    """概率路图生成器，支持守卫节点和连接器节点"""
    def __init__(self, grid_width, grid_height, obstacles, cell_size, num_nodes=100, connection_radius=1.0):
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
        self.cell_size = cell_size

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
        self.nodes = self.guards + self.connectors

        if not self.nodes:
            raise ValueError("请先生成节点！")
        
        # 使用简单的距离计算查找邻居
        for i, node in enumerate(self.nodes):
            for j, neighbor in enumerate(self.nodes):
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
    
    def _generate_strategic_seeds(self):
        seeds = []
        # 沿每个边界取若干点作为初始点
        boundaries = [
            [(i * self.cell_size, self.cell_size) for i in range(1, self.grid_width - 1, 5)],
            [(i * self.cell_size, (self.grid_height - 2) * cell_size) for i in range(1, self.grid_width - 1, 5)],
            [(self.cell_size, i * self.cell_size) for i in range(1,self.grid_height - 1, 5)],
            [((self.grid_width - 2) * self.cell_size, i * self.cell_size) for i in range(1, self.grid_height - 1, 5)]
        ]

        for boundary in boundaries:
            for point in boundary:
                if self._is_valid_position(point[0], point[1]):
                    seeds.append(SunNode(point))

        return seeds
    
    def generate_sun_ray_graph(self):
        # 从边界选一些点作为起始点
        seed_points = self._generate_strategic_seeds()
        open_list = seed_points
        closed_list = []
        while open_list:
            current_node: SunNode = open_list.pop(0)
            closed_list.append(current_node)

            # 计算当前节点的角度迭代器
            angle_iter = current_node.get_angle_iter()
            
            

    def generate_prm(self, node_generate_mode):
        """生成概率路图"""
        if node_generate_mode == "sampling":
            self.generate_nodes()
            self.connect_edges()
        elif node_generate_mode == "sun_ray":
            self.generate_sun_ray_graph()
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
    node_generate_mode = ENV_CONFIG['node_generate_mode']
    total_cells = grid_width * grid_height
    num_obstacles = int(total_cells * 0.2)  # 20% 障碍物
    obstacles = []
    while len(obstacles) < num_obstacles:
        x = np.random.randint(0, grid_width)
        y = np.random.randint(0, grid_height)
        obstacles.append((x, y))
    prm_generator = PRMGenerator(grid_width, grid_height, obstacles, cell_size, num_nodes=200, connection_radius=0.8)
    nodes, edges = prm_generator.generate_prm(node_generate_mode=node_generate_mode)
    print(len(nodes), "nodes generated")
    print(len(edges), "edges generated")
    renderer = PRMRenderer(grid_width, grid_height)
    renderer.run(nodes, edges, obstacles)