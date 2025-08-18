import numpy as np
import sys
sys.path.append('./pursuer_strategies/PRM')
sys.path.append('.')
sys.path.append('./utils')
from config import BICYCLE_MODEL_CONFIG, ENV_CONFIG, PYGAME_CONFIG
from utils.pure_pursuit import pure_pursuit_control
from utils.dwa import dwa_control
import heapq
import numpy as np
from generator import PRMGenerator

PRM = {
    'nodes': [],
    'edges': []
}

def heuristic(node, target):
    """计算启发式函数（欧几里得距离）"""
    return np.linalg.norm(np.array(node) - np.array(target))

def a_star_search(prm_nodes, prm_edges, start, target):
    """
    使用 A* 算法在 PRM 上寻找路径
    Args:
        prm_nodes: PRM 的节点列表
        prm_edges: PRM 的边列表
        start: 起点 (x, y)
        target: 目标点 (x, y)
    Returns:
        path: 从起点到目标点的路径
    """
    # 构建邻接表
    adjacency_list = {node: [] for node in prm_nodes}
    for edge in prm_edges:
        node1, node2 = edge
        adjacency_list[node1].append(node2)
        adjacency_list[node2].append(node1)
    #find nearest nodes in prm
    start_node = min(prm_nodes, key=lambda node: np.linalg.norm(np.array(node) - np.array(start)))
    target_node = min(prm_nodes, key=lambda node: np.linalg.norm(np.array(node) - np.array(target)))

    # 优先队列 (open list)
    open_list = []
    heapq.heappush(open_list, (0 + heuristic(start_node, target_node), 0, start_node, [start_node]))  # (f, g, current_node, path)

    # 已访问节点 (closed list)
    closed_list = set()

    while open_list:
        # 从优先队列中取出 f 值最小的节点
        _, g, current_node, path = heapq.heappop(open_list)

        # 如果当前节点是目标点，返回路径
        if np.linalg.norm(np.array(current_node) - np.array(target_node)) < 0.2:  # 允许一定误差
            return path

        # 将当前节点加入已访问列表
        closed_list.add(current_node)

        # 遍历邻居节点
        for neighbor in adjacency_list[current_node]:
            if neighbor in closed_list:
                continue

            # 计算 g 和 f 值
            g_new = g + np.linalg.norm(np.array(current_node) - np.array(neighbor))
            f_new = g_new + heuristic(neighbor, target_node)

            # 将邻居节点加入优先队列
            heapq.heappush(open_list, (f_new, g_new, neighbor, path + [neighbor]))

    # 如果没有找到路径，返回空列表
    return []

def prm_pursuer_policy(observation, self_index):
    """
    追捕者策略：使用 A* 在 PRM 上寻找路径
    Args:
        observation: 环境观测
        self_index: 当前追捕者的索引
    Returns:
        action: (steering_angle, speed)
        path: A* 计算的路径
        preview_point: 路径上的预览点
    """
    global PRM
    lookahead_distance = 0
    # 获取追捕者和目标点位置
    pursuer_state = observation['pursuers_states'][self_index]
    evader_state = observation['evader_state']
    start = (pursuer_state[0], pursuer_state[1])  # 追捕者位置
    target = (evader_state[0], evader_state[1])  # 目标点位置

    if not PRM['nodes'] or not PRM['edges']:
        prm_generator = PRMGenerator(ENV_CONFIG['gridnum_width'], ENV_CONFIG['gridnum_height'],\
                                      observation['obstacles'], 
                                      ENV_CONFIG['cell_size'], num_nodes=200, connection_radius=0.8)
        nodes, edges = prm_generator.generate_prm()
        PRM['nodes'] = nodes
        PRM['edges'] = edges
    else:
        nodes = PRM['nodes']
        edges = PRM['edges']
    path = a_star_search(nodes, edges, start, target)
    # 如果找到路径，选择路径上第一个超过 lookahead_distance 的点作为预览点
    if path:
        cumulative_distance = 0
        preview_point = path[0]
        for i in range(1, len(path)):
            cumulative_distance += np.linalg.norm(np.array(path[i]) - np.array(path[i - 1]))
            if cumulative_distance > lookahead_distance:
                preview_point = path[i]
                break
        preview_point = (preview_point[0], preview_point[1] )
    else:
        preview_point = start  # 如果没有路径，保持当前位置

    # 使用纯追踪控制计算动作
    steering_angle, speed = pure_pursuit_control(pursuer_state, preview_point)
    steering_angle = dwa_control(pursuer_state, (steering_angle, speed), observation['obstacles'], preview_point)
    return (steering_angle, speed)#, path, preview_point

if __name__ == "__main__":
    # 测试 PRM 路径规划
    observation = {
        'pursuers_states': [(0, 0, 0)],
        'evader_state': (5, 5, 0),
        'obstacles': [(1, 1), (2, 2), (3, 3)],
        'gridnum_width': ENV_CONFIG['gridnum_width'],
        'gridnum_height': ENV_CONFIG['gridnum_height']
    }
    action, path, preview_point = prm_pursuer_policy(observation, 0)
    print("Action:", action)
    print("Path:", path)
    print("Preview Point:", preview_point)