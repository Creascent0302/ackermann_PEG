import numpy as np
import sys
sys.path.append('..')  # 添加上级目录到路径中
from config import ENV_CONFIG
import heapq
import time 
from utils.astar import astar_search
from utils.astar_pp_dwa import evader_track_path, num_obs_neighbor
from utils.ttr import (
    compute_ttr,
    env_to_image
)
CURRENT_PATH = None  # 全局变量，用于存储当前路径
LAST_DECISION_TIME = 0
# 从配置中获取网格参数
grid_width = ENV_CONFIG['gridnum_width']
grid_height = ENV_CONFIG['gridnum_height']
cell_size = ENV_CONFIG['cell_size']


def path_evaluate(path,ttr):
    """
    check whether the path is still qualified
    """
    if path == None:
        return False
    if len(path) == 0:
        return False
    advantage = ttr['advantage']
    for i in range(len(path)):
        x, y = path[i]
        if advantage[x, y] < 1:
            return False
    if advantage[path[-1][0], path[-1][1]] < 12:
        return False
    return True

def final_pos_select(ttr_result,observation):
    pursuer_field = ttr_result['pursuer_field']
    pursuer_field_copy = np.copy(pursuer_field)
    obstacles = set(observation['obstacles'])
    while True:
        x, y = np.unravel_index(np.argmax(pursuer_field_copy), pursuer_field_copy.shape)
        if (x, y) not in obstacles and num_obs_neighbor(x, y, obstacles) <= 2\
            and pursuer_field_copy[x, y] < np.inf:
            break
        pursuer_field_copy[x, y] = float('-inf')  # Exclude this position and search again
    return (x, y)

def compute_path(final_pos, observation,ttr_result):
    """
    计算从逃避者当前位置到最终位置的路径
    """
    obstacles = observation['obstacles']
    evader_state = observation['evader_state']
    final_gridx = (final_pos[0]+0.5) * ENV_CONFIG['cell_size']
    final_gridy = (final_pos[1]+0.5) * ENV_CONFIG['cell_size']

    #use astar to search the path toward (final_gridx, final_gridy)
    path = evader_astar_search(evader_state,
                            (final_gridx, final_gridy),
                            obstacles,ttr_result)
    return path

def break_encircle(observation):
    """
    找到一个路径，使得逃避者能够打破包围圈
    """
    env_image = env_to_image(observation)
    ttr_result = compute_ttr(env_image, observation['pursuers_states'])
    final_pos = final_pos_select(ttr_result,observation)
    path = compute_path( final_pos, observation,ttr_result)
    return path

def update_path(path,observation):
    evader_state = observation['evader_state']
    #find the closest point on the path to the evader
    min_distance = float('inf')
    closest_point = None
    for point in path:
        point_world = ((point[0] + 0.5) * ENV_CONFIG['cell_size'], (point[1] + 0.5)* ENV_CONFIG['cell_size'])
        distance = np.linalg.norm(np.array(point_world) - np.array(evader_state[:2]))
        if distance < min_distance:
            min_distance = distance
            closest_point = point
    #update the path to start from the closest point
    if closest_point is not None:
        path = path[path.index(closest_point):]
    return path

def break_encircle_policy(observation):
    """
    使用 A* 路径规划和 Pure Pursuit 策略的逃逸者策略
    """
    global CURRENT_PATH, LAST_DECISION_TIME
    obstacles = observation['obstacles']
    evader_x, evader_y, evader_theta = observation['evader_state']
    time_since_last_decision = time.time() - LAST_DECISION_TIME
    if time_since_last_decision < 0.25:
        CURRENT_PATH = CURRENT_PATH
    elif path_evaluate(CURRENT_PATH, compute_ttr(env_to_image(observation), observation['pursuers_states'])):
        path = update_path(CURRENT_PATH, observation)
        CURRENT_PATH = path
    else:
        path = break_encircle(observation)
        CURRENT_PATH = path  # 更新全局路径变量
        LAST_DECISION_TIME = time.time()
    (steering_angle, speed), path, preview_point = evader_track_path(CURRENT_PATH, (evader_x, evader_y, evader_theta), obstacles)

    return (steering_angle, speed), path, preview_point

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

def evader_astar_search(start, goal, obstacles,ttr_result):
    """
    A* 搜索算法
    """
    veh_pos = start
    # 将实际坐标转换为网格坐标
    start = (int(start[0] / cell_size), int(start[1] / cell_size))
    goal = (int(goal[0] / cell_size), int(goal[1] / cell_size))
    unexplored = 0 #mark

    # 新增: 引入 pursuer_field 影响启发式
    pursuer_field = ttr_result['pursuer_field']
    pf_max = np.max(pursuer_field[np.isfinite(pursuer_field)]) if pursuer_field.size > 0 else 1.0
    risk_weight = 100  # 可调参数

    def heuristic(a, b):
        # 原始曼哈顿距离
        dist = abs(a[0] - b[0]) + abs(a[1] - b[1])
        # 安全项：pursuer_field 越大越安全
        safety = 1 - pursuer_field[a[0], a[1]] / (pf_max + 1e-6)
        h = dist + risk_weight * safety
        return h if h > 0 else 0.0

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
                    if not second_step_constraint(neighbor, veh_pos):  # add a contraint at the second step
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