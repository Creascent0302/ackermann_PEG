import pygame
import numpy as np
from scipy.spatial import KDTree
import sys
sys.path.append('.')
from config import ENV_CONFIG
from collections import deque
import math

# 定义一个节点类，储存更多节点信息方便计算
class SunNode:
    def __init__(self, point):
        self.connections = [] #储存(角度，半径，邻居节点)数据对，角度取水平向右为 0 度，逆时针为正方向，用弧度表示
        self.position = point  # (x, y) 坐标
     
    def __eq__(self, value):
        if not isinstance(value, SunNode):
            return NotImplemented
        return self.position == value.position

    def add_connection(self, angle, radius, neighbor):
        self.connections.append((angle, radius, neighbor))            

    # 对每个已有光线都引入斥力，避免过于密集的光线生成，思想类似两个相同点电荷之间的电场线
    def get_valid_angles(self):
        iter_range = np.arange(0, 2 * np.pi, ENV_CONFIG["angle_iter_step"])
        # 用来获取无效的角度范围，最后取反就是有效的角度范围
        is_forbidden = np.zeros_like(iter_range, dtype=bool)
        boundary_default = ENV_CONFIG["angle_of_repulsion_default"]
        boundary_high = ENV_CONFIG["angle_of_repulsion_high"]
        boundary_low = ENV_CONFIG["angle_of_repulsion_low"]

        if not self.connections:
            return iter_range  # 返回一个默认的角度迭代器
        else:
            for edge in self.connections:
                angle = edge[0]

                start = None
                end = None
                # 大于斥力引入长度边界
                if edge[1] > ENV_CONFIG["radius_boundary_of_repulsion_high"]:
                    start = angle - boundary_default
                    end = angle + boundary_default
                
                # 在斥力两级边界之间，做遍历范围限制
                elif edge[1] > ENV_CONFIG["radius_boundary_of_repulsion_low"] and edge[1] <= ENV_CONFIG["radius_boundary_of_repulsion_high"]:
                    start = angle - boundary_high
                    end = angle + boundary_high
                
                elif edge[1] > 0 and edge[1] <= ENV_CONFIG["radius_boundary_of_repulsion_low"]:
                    start = angle - boundary_low
                    end = angle + boundary_low

                # print("start:", start, "end:", end, "angle:", angle, "radius:", edge[1])
                if start is not None and end is not None:
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
        self.min_tangent_radius = cell_size * 2
        self.guards = []  # 守卫节点
        self.connectors = []  # 连接器节点
        self.edges = []
        self.collision_radius = 2 * ENV_CONFIG['agent_collision_radius'] / 3
        self.cell_size = cell_size
        self.ray_step = cell_size * 0.1 # 光线发射时前进的步长
        self.ray_forward = cell_size * 1.2 # tangent 节点前进的固定步长
        self.tangent_limit = cell_size # tangent 节点采用固定步长拓展的最小光线长度
        self.ray_forward_low = cell_size * 0.8 # tangent 节点前进的比率

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
    
    # 阳光算法的初始点获取策略
    def _generate_strategic_seeds(self):
        """从边界选取若干点作为初始点"""
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
    
    def cast_ray(self, position, angle):
        """从指定位置沿指定角度发射射线，返回射线长度"""
        x, y = position
        dx = math.cos(angle)
        dy = - math.sin(angle) # 注意坐标系不是标准平面坐标系，纵轴是反过来的
        current_length = 0
        max_length = min(self.grid_width, self.grid_height) * ENV_CONFIG['cell_size'] * 0.8
        
        while current_length < max_length:
            current_length += self.ray_step
            x += dx * self.ray_step
            y += dy * self.ray_step
            if not self._is_valid_position(x, y):
                return max(0, current_length - self.ray_step)
        
        return max_length

    def generate_tangent_point(self, node: SunNode, angle, ray_length, open_list):
        """生成切线点并将其添加到图中"""
        x, y = node.position
        dx = math.cos(angle)
        dy = - math.sin(angle)

        # 根据光线长度决定切线点的前进距离，但其实感觉没啥效果
        if ray_length < self.tangent_limit:
            tangent_x = x + dx * (ray_length + self.ray_forward_low)
            tangent_y = y + dy * (ray_length + self.ray_forward_low)
        else:
            tangent_x = x + dx * (ray_length + self.ray_forward)
            tangent_y = y + dy * (ray_length + self.ray_forward)

        if self._is_valid_position(tangent_x, tangent_y):

            tangent_node = SunNode((tangent_x, tangent_y))
            connection_record = []
            # 先记录 tangent node 与其他节点的连接信息，如果有临近节点则考虑替换
            i= 0
            delete_list = []
            while i < len(open_list):
                if i in delete_list:
                    i += 1
                    continue

                other_node = open_list[i]

                if self._is_valid_edge(tangent_node.position, other_node.position):
                    to_other_angle = math.atan2(tangent_node.position[1] - other_node.position[1], other_node.position[0] - tangent_node.position[0]) # 同理，纵坐标是负值，因为坐标系是反的
                    to_tangent_angle = math.atan2(other_node.position[1] - tangent_node.position[1], tangent_node.position[0] - other_node.position[0])
                    loop_ray_length = np.linalg.norm(np.array(tangent_node.position) - np.array(other_node.position))
                    connection_record.append((i, to_other_angle, to_tangent_angle, loop_ray_length))

                    # 如果切线点和其他节点之间的距离小于最小连接半径，则尝试替换 tangent 节点
                    if loop_ray_length < self.min_tangent_radius:
                        if self._is_valid_edge(node.position, other_node.position):
                            tangent_node = other_node
                            delete_list.append(i) # 删除当前节点，避免重复连接
                            connection_record = [] # 清空连接记录
                            i = 0 # 从头重新遍历
                i += 1

            # 连接当前节点和tangent 节点
            node_to_tangent_angle = math.atan2(node.position[1] - tangent_node.position[1], tangent_node.position[0] - node.position[0])
            tangent_to_node_angle = math.atan2(tangent_node.position[1] - node.position[1], node.position[0] - tangent_node.position[0])
            if node_to_tangent_angle < 0:
                node_to_tangent_angle += 2 * np.pi
            if tangent_to_node_angle < 0:
                tangent_to_node_angle += 2 * np.pi
            ray_length_to_tangent = np.linalg.norm(np.array(tangent_node.position) - np.array(node.position))
            node.add_connection(node_to_tangent_angle, ray_length_to_tangent, tangent_node)
            tangent_node.add_connection(tangent_to_node_angle, ray_length_to_tangent, node)
            # 连接所有记录的连接
            for record in connection_record:
                idx, to_other_angle, to_tangent_angle, loop_ray_length = record
                other_node = open_list[idx]
                if to_other_angle < 0:
                    to_other_angle += 2 * np.pi
                if to_tangent_angle < 0:
                    to_tangent_angle += 2 * np.pi
                tangent_node.add_connection(to_other_angle, loop_ray_length, other_node)
                other_node.add_connection(to_tangent_angle, loop_ray_length, tangent_node)
            if tangent_node not in open_list:
                # 如果切线点不在 open_list 中，则添加到 open_list 中
                open_list.append(tangent_node)
        return


    def generate_sun_ray_graph(self):
        # 从边界选一些点作为起始点
        seed_points = self._generate_strategic_seeds()
        open_list = deque(seed_points)
        closed_list = deque()
        num_try = 0
        # for node in open_list:
        #     print(node.position)
        while open_list and num_try < self.num_nodes:
            num_try += 1
            # print(num_try)
            if num_try % 10 == 0:
                print(f"探索次数: {num_try}/{self.num_nodes}")

            current_node: SunNode = open_list.popleft()
            # print(num_try, current_node.position)
            # 计算当前节点的角度迭代范围
            valid_angles = current_node.get_valid_angles()
            # print(valid_angles)
            ray_queue = deque() # 记录射线长度
            first_ray_len = -1
            second_ray_len = -1
            angle_iter = np.arange(0, 2 * np.pi, ENV_CONFIG['angle_iter_step'])
            finished_angle = np.zeros_like(angle_iter) # 用来记录已经选择进行节点拓展的光线角度，防止重复拓展

            for i, angle in enumerate(angle_iter):
                ray_length = self.cast_ray(current_node.position, angle) # 根据太阳位置和角度发射射线并计算长度
                # 通过判断前后两次光线长度的变化量的比率来判断是否需要生成切线点
                if not ray_queue:
                    ray_queue.append(ray_length)
                    first_ray_len = ray_length
                    continue
                elif len(ray_queue) == 1:
                    ray_queue.append(ray_length)
                    second_ray_len = ray_length
                    continue
                
                # 此时已经有至少两条光线
                elif len(ray_queue) == 2:
                    # 通过比较相邻三条光线两两之间的长度差来判断是否为边界
                    pre_delta = ray_queue[1] - ray_queue[0]
                    pre_sec_lenth = ray_queue.popleft()
                    ray_queue.append(ray_length)
                    delta = ray_queue[1] - ray_queue[0]
                    rate = max(abs(delta), 0.1 * self.cell_size) / max(abs(pre_delta), 0.1 * self.cell_size)

                    if rate >= 5:
                        if delta > 0 and angle in valid_angles:
                            self.generate_tangent_point(current_node, angle, ray_queue[0], open_list) # 当前光线突变，以前一条光线长度为基础，长短短
                            finished_angle[i] = 1
                        elif delta < 0 and finished_angle[i - 1] == 0 and (angle_iter[i - 1] in valid_angles):
                            self.generate_tangent_point(current_node, angle_iter[i - 1], ray_length, open_list) # 前一条光线突变，以当前光线长度为基础拓展前一条光线，短长长
                            finished_angle[i - 1] = 1
                    elif rate <= 0.2:
                        if pre_delta > 0 and finished_angle[i - 1] == 0 and angle_iter[i - 1] in valid_angles:
                            self.generate_tangent_point(current_node, angle_iter[i - 1], pre_sec_lenth, open_list) # 从前一条光线开始突变，以前面第二条光线长度为基础拓展前一条光线，长长短
                            finished_angle[i - 1] = 1
                        elif pre_delta < 0 and finished_angle[i - 2] == 0 and angle_iter[i - 2] in valid_angles:
                            self.generate_tangent_point(current_node, angle_iter[i - 2], ray_queue[0], open_list) # 前面第二条光线是突变光线，以前一条光线长度为基础拓展前面第二条光线，短短长
                            finished_angle[i - 2] = 1

                    elif finished_angle[i - 1] == 0 and delta < 0 and pre_delta > 0 and abs(delta) > self.cell_size and pre_delta > self.cell_size and angle_iter[i - 1] in valid_angles:
                        # 第三种情况，短长短，这种情况应该比较少见但是不能忽略
                        self.generate_tangent_point(current_node, angle_iter[i - 1], max(ray_length, pre_sec_lenth), open_list)
                        finished_angle[i - 1] = 1

            closed_list.append(current_node)
        self.nodes = []
        self.edges = []
        for node in (open_list + closed_list):
            self.nodes.append(node.position)
            for connection in node.connections:
                angle, radius, neighbor = connection
                self.edges.append((node.position, neighbor.position))


    def generate_prm(self, node_generate_mode="sampling"):
        """生成概率路图"""
        if node_generate_mode == "sampling":
            self.generate_nodes()
            self.connect_edges()
        elif node_generate_mode == "sun_ray":
            self.generate_sun_ray_graph()
            # print(self.nodes, self.edges)
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

    SEED = 42
    np.random.seed(SEED)

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