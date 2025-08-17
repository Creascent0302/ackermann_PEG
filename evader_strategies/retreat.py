import numpy as np
import sys
sys.path.append('..')  # 添加上级目录到路径中
from config import ENV_CONFIG
from utils.astar import astar_search, first_step_constraint,second_step_constraint, obs_trap
import time
from utils.astar_pp_dwa import  evader_track_path, num_obs_neighbor
from utils.ttr import (
    compute_ttr,
    env_to_image
)

CURRENT_PATH = None  # 全局变量，用于存储当前路径
LAST_DECISION_TIME = 0

def in_corner(position):
    """Check if the position (nx, ny) is in a corner of the grid."""
    if position is None:
        return False
    nx, ny = position
    grid_width = ENV_CONFIG['gridnum_width']
    grid_height = ENV_CONFIG['gridnum_height']
    return (nx == 0 and ny == 0) or (nx == 0 and ny == grid_height - 1) or \
           (nx == grid_width - 1 and ny == 0) or (nx == grid_width - 1 and ny == grid_height - 1)

def retreat_path(observation):
    """
    find the path that has maximum distance to pursuers
    """
    length_upper_bound = 6#max length
    env_image = env_to_image(observation)
    obstacles = set(observation['obstacles']) 
    ttr_result = compute_ttr(env_image, observation['pursuers_states'])
    advantage = ttr_result['advantage']
    evader_gridx = int(observation['evader_state'][0] / ENV_CONFIG['cell_size'])
    evader_gridy = int(observation['evader_state'][1] / ENV_CONFIG['cell_size'])
    path = []
    length = 0
    while length < length_upper_bound:
        max_advantage = float('-inf')
        best_move = None
        obs_num = 10
        # Search for the best move among 8 directions
        for dx, dy in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
            nx, ny = evader_gridx + dx, evader_gridy + dy
            if not (0 <= nx < ENV_CONFIG['gridnum_width'] and 0 <= ny < ENV_CONFIG['gridnum_height']):
                continue
            # 目标格不能是障碍
            if (nx, ny) in obstacles or num_obs_neighbor(nx, ny, obstacles) > 2 \
                or obs_trap(evader_gridx,evader_gridy,nx,ny,obstacles)\
                or (best_move != None and in_corner((nx, ny))):
                continue
            # 对角移动时，若两条相邻正交边都被障碍阻挡，则禁止“穿角”
            if dx != 0 and dy != 0:
                if (evader_gridx + dx, evader_gridy) in obstacles or (evader_gridx, evader_gridy + dy) in obstacles:
                    continue
            if (advantage[nx, ny] > max_advantage or\
                 (num_obs_neighbor(nx, ny, obstacles) < obs_num and advantage[nx, ny] == max_advantage) or in_corner(best_move)) \
                 and (nx,ny) not in path:
                if length == 0 and first_step_constraint((nx, ny), observation['evader_state']):
                    # 如果是第一步，且符合约束条件
                    max_advantage = advantage[nx][ny]
                    obs_num = num_obs_neighbor(nx, ny, obstacles)
                    best_move = (nx, ny)
                elif length >= 1 and second_step_constraint((nx, ny), observation['evader_state']):
                    # 如果是第二步，且符合约束条件
                    max_advantage = advantage[nx][ny]
                    obs_num = num_obs_neighbor(nx, ny, obstacles)
                    best_move = (nx, ny)           
        # If no better move is found, stop (local maximum)
        if max_advantage == float('-inf'):
            break
        # Update path and move to the best position
        evader_gridx, evader_gridy = best_move
        path.append(best_move)
        length += 1
    return path

def evader_retreat_policy(observation):
    """
    使用 A* 路径规划和 Pure Pursuit 策略的逃逸者策略
    """
    global CURRENT_PATH, LAST_DECISION_TIME
    obstacles = observation['obstacles']
    evader_x, evader_y, evader_theta = observation['evader_state']
    time_since_last_decision = time.time() - LAST_DECISION_TIME
    if time_since_last_decision < 0.3:
        CURRENT_PATH = CURRENT_PATH
    else:
        path = retreat_path(observation)
        CURRENT_PATH = path  # 更新全局路径变量
        LAST_DECISION_TIME = time.time()
    (steering_angle, speed), path, preview_point = evader_track_path(CURRENT_PATH, (evader_x, evader_y, evader_theta), obstacles)

    return (steering_angle, speed), path, preview_point