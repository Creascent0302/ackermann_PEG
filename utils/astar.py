import heapq
import numpy as np
import sys
sys.path.append('..')  # 添加上级目录到路径中
from config import ENV_CONFIG, BICYCLE_MODEL_CONFIG, PYGAME_CONFIG

# 从配置中获取网格参数
grid_width = ENV_CONFIG['gridnum_width']
grid_height = ENV_CONFIG['gridnum_height']
cell_size = ENV_CONFIG['cell_size']

def obs_trap(cur_x,cur_y,next_x,next_y,obstacles):
    """
    situations might cause trap
    """
    dx = next_x - cur_x
    dy = next_y - cur_y
    if (next_x + dx,next_y) in obstacles and (next_x,next_y + dy) in obstacles\
        or ((next_x + dx > grid_width or next_x + dx < 0) and (next_y + dy > grid_height or next_y + dy < 0)):
        return True
    return False

def second_step_constraint(node, veh_pos):
    """检查第二个移动是否符合约束条件"""
    # angle between hearning and target
    center_nodex = node[0] * cell_size + cell_size / 2
    center_nodey = node[1] * cell_size + cell_size / 2
    dx = center_nodex - veh_pos[0]
    dy = center_nodey - veh_pos[1]
    if dx**2 + dy**2 < 0.02:  # 如果距离过近，直接返回 False
        return False
    return True

def first_step_constraint(node, veh_pos):
    """检查第一个移动是否符合约束条件"""
    if len(veh_pos) == 2:
        return True
    # angle between hearning and target
    center_nodex = node[0] * cell_size + cell_size / 2
    center_nodey = node[1] * cell_size + cell_size / 2
    dx = center_nodex - veh_pos[0]
    dy = center_nodey - veh_pos[1]
    if dx == 0 and dy == 0:
        return True
    angle = np.arctan2(dy, dx)
    # 计算与车辆当前方向的角度差
    current_angle = veh_pos[2] if len(veh_pos) > 2 else 0
    angle_diff = (angle - current_angle + np.pi) % (2 * np.pi) - np.pi
    return -2*np.pi/3 <= angle_diff <= 2*np.pi/3

def astar_search(start, goal, obstacles):
    """
    A* 搜索算法
    :param start: 起点 (x, y)
    :param goal: 目标点 (x, y)
    :param obstacles: 障碍物集合，包含所有障碍物的 (x, y) 坐标
    :return: 从起点到目标点的路径 (list of (x, y))，如果无法到达目标点，则返回空列表
    """
    veh_pos = start
    # 将实际坐标转换为网格坐标
    start = (int(start[0] / cell_size), int(start[1] / cell_size))
    goal = (int(goal[0] / cell_size), int(goal[1] / cell_size))
    unexplored = 0 #mark
    def heuristic(a, b):
        """计算启发式函数值（曼哈顿距离）"""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def get_neighbors(node):
        """获取当前节点的邻居节点"""
        neighbors = [
            (node[0] + 1, node[1]),  # 右
            (node[0] - 1, node[1]),  # 左
            (node[0], node[1] + 1),  # 上
            (node[0], node[1] - 1),  # 下
            (node[0] + 1, node[1] + 1),  # 右上
            (node[0] - 1, node[1] + 1),  # 左上
            (node[0] + 1, node[1] - 1),  # 右下
            (node[0] - 1, node[1] - 1)   # 左下
        ]
        valid_neighbors = []
        for x, y in neighbors:
            # 检查是否越界或在障碍物中
            if 0 <= x < grid_width and 0 <= y < grid_height and (x, y) not in obstacles:
                # 对角线方向需要额外检查
                if (x != node[0] and y != node[1]):  # 对角线方向
                    if (x, node[1]) in obstacles or (node[0], y) in obstacles:
                        continue  # 如果水平或垂直方向被阻挡，则跳过
                valid_neighbors.append((x, y))
        return valid_neighbors

    # 优先队列（最小堆）
    open_set = []
    heapq.heappush(open_set, (0, start))  # (优先级, 节点)

    # 路径记录
    came_from = {}

    # g_score: 从起点到当前节点的实际代价
    g_score = {start: 0}

    # f_score: 从起点经过当前节点到目标点的估计代价
    f_score = {start: heuristic(start, goal)}

    while open_set:

        # 获取优先级最低的节点
        _, current = heapq.heappop(open_set)

        # 如果到达目标点，重建路径
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]  # 返回从起点到目标点的路径

        # 遍历当前节点的所有邻居
        for neighbor in get_neighbors(current):
            if obs_trap(current[0], current[1], neighbor[0], neighbor[1], obstacles):
                continue
            tentative_g_score = g_score[current] + 1  # 假设每一步的代价为1
            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                if unexplored == 0:
                    if not first_step_constraint(neighbor, veh_pos):#add a contraint at the first step
                        continue    
                if unexplored == 1:      
                    if not second_step_constraint(neighbor, veh_pos):#add a contraint at the second step
                        continue
                # 更新路径记录
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)

                # 如果邻居不在 open_set 中，将其加入
                if neighbor not in [item[1] for item in open_set]:
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        unexplored = 1

    # 如果无法到达目标点，返回空列表
    return []