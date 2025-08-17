from utils.astar import astar_search
from utils.pure_pursuit import pure_pursuit_control
from utils.dwa import dwa_control
import sys
sys.path.append('..')
from config import BICYCLE_MODEL_CONFIG, ENV_CONFIG
import numpy as np
import time
PURSUER_TIME_DICT = {}
PURSUER_PATH_DICT = {}
EVADER_TIME = 0
EVADER_PATH = []


def num_obs_neighbor(nx,ny,obstacles):
    """
    calculate the number of obstacles around nx,ny
    """
    count = 0
    for (dx,dy) in [ (-1, 0), (0, -1), (0, 1), (1, 0)]:
            if (nx + dx, ny + dy) in obstacles:
                count += 1
    return count

def path_refinement(path,obstacles):

    refined_path = []
    if len(path) == 0:
        return None
    if len(path) < 3:
        refined_path = path
    else:
        i = 0
        while i < len(path) - 2:
            start = path[i]
            end = path[i + 2]
            refined_path.append(start)
            # Check if the direct line between start and end is clear
            mark, position = is_path_clear(start, end, obstacles, path)
            if mark: 
                refined_path.append(position)
                i += 2
            else:
                i += 1
        refined_path.append(path[-2])
        refined_path.append(path[-1])
    if num_obs_neighbor(refined_path[-1][0], refined_path[-1][1], obstacles) > 2:
        cur_pos = refined_path[-1]
        while num_obs_neighbor(cur_pos[0], cur_pos[1], obstacles) > 1:
            refined_path.pop()
            cur_pos = refined_path[-1]
    elif num_obs_neighbor(refined_path[-1][0], refined_path[-1][1], obstacles) == 2:
        sum_dx = 0
        sum_dy = 0
        for (dx,dy) in [ (-1, 0), (0, -1), (0, 1), (1, 0)]:
            if (refined_path[-1][0] + dx, refined_path[-1][1] + dy) in obstacles:
                sum_dx += dx
                sum_dy += dy
        if abs(sum_dx) == 1 and abs(sum_dy) == 1:
            refined_path.pop()
    return refined_path

def is_path_clear(start, end, obstacles,path):
    """
    Check if the direct path between start and end is clear of obstacles
    """
    if start[0] == end[0]:
        for y in range(min(start[1], end[1])+ 1, max(start[1], end[1])):
            if (start[0], y) not in obstacles and (start[0], y) not in path:
                return True,(start[0], y)
    elif start[1] == end[1]:
        for x in range(min(start[0], end[0])+ 1, max(start[0], end[0])):
            if (x, start[1]) not in obstacles and (x, start[1]) not in path:
                return True,(x, start[1])
    return False,None

def pursuer_track_target(target,pursuer,obstacles,pursuer_idx):  
    """
    Track the target using A* path planning and Pure Pursuit control
    """
    global PURSUER_TIME_DICT, PURSUER_PATH_DICT
    if pursuer_idx not in PURSUER_TIME_DICT:
        PURSUER_TIME_DICT[pursuer_idx] = time.time()
        path = astar_search(
                (pursuer[0], pursuer[1], pursuer[2]),
                (target[0], target[1]),
                obstacles
            )
        PURSUER_PATH_DICT[pursuer_idx] = path
    if time.time() - PURSUER_TIME_DICT[pursuer_idx] <  0.3:
        path = PURSUER_PATH_DICT.get(pursuer_idx, None)
    else:
        path = astar_search(
                (pursuer[0], pursuer[1], pursuer[2]),
                (target[0], target[1]),
                obstacles
            )
        path = path_refinement(path, obstacles)
        PURSUER_PATH_DICT[pursuer_idx] = path
        PURSUER_TIME_DICT[pursuer_idx] = time.time()
    preview_point = None
    if not path or len(path) < 2:
        return (0,0),path, preview_point
    lookahead_distance = 0.18
    for i in range(1, len(path)):        
        point_x, point_y = (path[i][0] + 0.5) * ENV_CONFIG['cell_size'], (path[i][1] + 0.5) * ENV_CONFIG['cell_size']
        distance = np.sqrt((point_x - pursuer[0])**2 + (point_y - pursuer[1])**2)
        if distance >= lookahead_distance :
            preview_point = (point_x, point_y)            
            break

    # 如果没有找到合适的预瞄点，目标点为路径的最后一个点
    if preview_point is None:
        print("preview None")
        return (0, 0), path, preview_point

    steering_angle,_ = pure_pursuit_control(pursuer, preview_point)
    steering_angle, speed = dwa_control(
        pursuer,
        (steering_angle, BICYCLE_MODEL_CONFIG['max_speed']),
        obstacles,
        preview_point if preview_point else (pursuer[0], pursuer[1])
    )
    return (steering_angle, speed), path, preview_point

def evader_track_path(path, evader, obstacles):
    """
    Track the path using Pure Pursuit control
    """
    global EVADER_TIME, EVADER_PATH
    path = path_refinement(path, obstacles)
    EVADER_PATH = path
    EVADER_TIME = time.time()
    preview_point = None
    if not path or len(path) < 2:
        return (0,0),path, preview_point
    lookahead_distance = 0.18
    for i in range(1, len(path)):
        point_x, point_y = (path[i][0] + 0.5) * ENV_CONFIG['cell_size'], (path[i][1] + 0.5) * ENV_CONFIG['cell_size']
        distance = np.sqrt((point_x - evader[0])**2 + (point_y - evader[1])**2)
        if distance > lookahead_distance:
            preview_point = (point_x, point_y)
            break
    # 如果没有找到合适的预瞄点，目标点为路径的最后一个点
    if preview_point is None:
        return (0, 0), path, preview_point
    steering_angle,_ = pure_pursuit_control(evader, preview_point)
    steering_angle, speed = dwa_control(
        evader,
        (steering_angle, BICYCLE_MODEL_CONFIG['max_speed']),
        obstacles,
        preview_point if preview_point else (evader[0], evader[1])
    )
    return (steering_angle, speed), path, preview_point

def evader_track_target(target, evader, obstacles):
    """
    Track the target using A* path planning and Pure Pursuit control
    """
    global EVADER_TIME, EVADER_PATH
    if EVADER_TIME == 0:
        EVADER_TIME = time.time()
        path = astar_search(
            (evader[0], evader[1], evader[2]),
            (target[0], target[1]),
            obstacles
        )
        EVADER_PATH = path
    if time.time() - EVADER_TIME < 0.2:
        path = EVADER_PATH
    else:
        path = astar_search(
            (evader[0], evader[1], evader[2]),
            (target[0], target[1]),
            obstacles
        )
        EVADER_PATH = path
    preview_point = None
    if not path or len(path) < 2:
        return (0,0),path, preview_point
    lookahead_distance = 0.18
    for i in range(0, len(path)):
        point_x, point_y = (path[i][0] + 0.5) * ENV_CONFIG['cell_size'], (path[i][1] + 0.5) * ENV_CONFIG['cell_size']
        distance = np.sqrt((point_x - evader[0])**2 + (point_y - evader[1])**2)
        if distance >= lookahead_distance:
            preview_point = (point_x, point_y)
            break
    # 如果没有找到合适的预瞄点，目标点为路径的最后一个点
    if preview_point is None:
        return (0, 0), path, preview_point
    steering_angle,_ = pure_pursuit_control(evader, preview_point)
    steering_angle, speed = dwa_control(
        evader,
        (steering_angle, BICYCLE_MODEL_CONFIG['max_speed']),
        obstacles,
        preview_point if preview_point else (evader[0], evader[1])
    )
    return (steering_angle, speed), path, preview_point