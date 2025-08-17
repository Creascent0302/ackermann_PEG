import numpy as np
import sys
sys.path.append('..')  # 添加上级目录到路径中
from config import ENV_CONFIG
from utils.astar import astar_search
import time
from utils.astar_pp_dwa import evader_track_target


CURRENT_PATH = None  # 全局变量，用于存储当前路径
PATH_EVALUATE_THRESHOLD = 2 # 路径评估阈值
LAST_DECISION_TIME = 0
def path_evaluate(observation, path):
    """
    Evaluate the path based on distance to pursuers and obstacles
    """
    evader_x, evader_y, _ = observation['evader_state']
    pursuers_states = observation['pursuers_states']
    obstacles = observation['obstacles']
    
    # 计算路径上每个点到最近追捕者的距离
    min_distance_to_pursuer = float('inf')
    for i, point in enumerate(path):
        px_path, py_path = point
        px_path = px_path * ENV_CONFIG['cell_size'] + ENV_CONFIG['cell_size'] / 2
        py_path = py_path * ENV_CONFIG['cell_size'] + ENV_CONFIG['cell_size'] / 2
        for pursuer_state in pursuers_states:
            px, py, _ = pursuer_state
            distance = np.hypot(px_path - px, py_path - py)
            if distance < min_distance_to_pursuer and len(path) > 10:
                if i >= 10:
                    min_distance_to_pursuer = distance
            elif distance < min_distance_to_pursuer and len(path) <= 10:
                min_distance_to_pursuer = distance 
    return min_distance_to_pursuer

def target_selection(observation):
    """
    Randomly select target spots and choose the one with greatest distance to pursuers
    """
    global CURRENT_PATH, PATH_EVALUATE_THRESHOLD
    threshold = PATH_EVALUATE_THRESHOLD
    path = None
    evader_x, evader_y, evader_theta = observation['evader_state']
    pursuers_states = observation['pursuers_states']
    obstacles = observation['obstacles']
    
    # Environment boundaries
    grid_width = ENV_CONFIG['gridnum_width']
    grid_height = ENV_CONFIG['gridnum_height']

    attempts = 0
    while attempts < 50:
        attempts += 1
        if attempts > 20:
            threshold = 1.2
        if attempts > 30:
            threshold = 0.5
        # Generate random grid coordinates
        tx = np.random.randint(0, grid_width)
        ty = np.random.randint(0, grid_height)
        
        # Skip if the target is in an obstacle
        if (tx, ty) in obstacles:
            continue
            
        target_x = tx * ENV_CONFIG['cell_size'] + ENV_CONFIG['cell_size'] / 2
        target_y = ty * ENV_CONFIG['cell_size'] + ENV_CONFIG['cell_size'] / 2
        path = astar_search( (evader_x, evader_y, evader_theta), (target_x, target_y), obstacles)
        if path_evaluate(observation, path) > threshold and len(path) > 1:
            return path
    path = [(evader_x, evader_y)]
    return path


def astar_evader_policy(observation):
    """
    使用 A* 路径规划和 Pure Pursuit 策略的逃逸者策略
    """
    global CURRENT_PATH, PATH_EVALUATE_THRESHOLD, LAST_DECISION_TIME
    obstacles = observation['obstacles']
    evader_x, evader_y, evader_theta = observation['evader_state']
    time_since_last_decision = time.time() - LAST_DECISION_TIME
    if (CURRENT_PATH and path_evaluate(observation, CURRENT_PATH) > PATH_EVALUATE_THRESHOLD) \
        or time_since_last_decision < 1.5:
        target_spot = CURRENT_PATH[-1][0] * ENV_CONFIG['cell_size'] + ENV_CONFIG['cell_size'] / 2, \
                      CURRENT_PATH[-1][1] * ENV_CONFIG['cell_size'] + ENV_CONFIG['cell_size'] / 2
    else:
        path = target_selection(observation)
        target_spot = path[-1][0] * ENV_CONFIG['cell_size'] + ENV_CONFIG['cell_size'] / 2, \
                      path[-1][1] * ENV_CONFIG['cell_size'] + ENV_CONFIG['cell_size'] / 2
        CURRENT_PATH = path  # 更新全局路径变量
        LAST_DECISION_TIME = time.time()
    (steering_angle, speed), path, preview_point = evader_track_target(target_spot, (evader_x, evader_y, evader_theta), obstacles)

    return (steering_angle, speed), path, preview_point