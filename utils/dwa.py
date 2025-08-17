import numpy as np
import sys
sys.path.append('..')  # 添加上级目录到路径中
from config import BICYCLE_MODEL_CONFIG, ENV_CONFIG, PYGAME_CONFIG,SIMULATION_CONFIG 
from utils.astar import astar_search
from utils.pure_pursuit import pure_pursuit_control
from agent_model import bicycle_model

def dwa_control(pursuer_state, current_control, obstacles, preview_point):
    """
    仅控制转向角的DWA算法，速度保持恒定
    """
    x, y, theta = pursuer_state
    
    current_steer, speed = current_control
    dt = SIMULATION_CONFIG["time_step"]
    # 1. 计算转向角的动态窗口 - 受限于车辆物理约束
    max_steer = BICYCLE_MODEL_CONFIG['max_steering_angle']
    speed = BICYCLE_MODEL_CONFIG['max_speed']

    w_low = -max_steer
    w_high = max_steer
    w_res = 0.05

    # 2. 在动态窗口中采样转向角
    steering_samples = np.arange(w_low, w_high, w_res)
    
    # 如果采样为空，使用当前转向角
    if len(steering_samples) == 0:
        steering_samples = np.array([max(-max_steer, min(max_steer, current_steer))])
    
    best_score = -np.inf
    best_steer = current_steer
    collision_avoidable = False
    
    # 预测时间范围
    predict_time = 0.4 # seconds
    obstacle_radius = ENV_CONFIG['agent_collision_radius']
    wheelbase = BICYCLE_MODEL_CONFIG['wheelbase']
    
    # 3. 评价每个采样的转向角
    for steer in steering_samples:
        # 4. 预测轨迹 (使用自行车模型)
        traj = predict_trajectory(x, y, theta, speed, steer, wheelbase, dt, predict_time)
        
        # 5. 计算评价分数
        # 5.1 目标朝向评分
        if preview_point:
            goal_score = calculate_goal_score(traj, preview_point)
        else:
            goal_score = 0.0

        # 5.2 路径平滑性评分 (与当前转向角差异小的评分高)
        smooth_score = calculate_smooth_score(steer, current_steer, max_steer)
        
        # 5.3 安全性评分 (远离障碍物)
        dist_to_obstacle = calculate_distance_to_obstacle(traj, obstacles)
        if dist_to_obstacle < 0:  # 碰撞
            continue
        collision_avoidable = True
        obstacle_score = calculate_obstacle_score(dist_to_obstacle)
        
        # 5.4 综合评分
        weight_goal = 2.0
        weight_smooth = 0.3
        weight_obstacle = 0.8
        
        total_score = (weight_goal * goal_score +
                      weight_smooth * smooth_score +
                      weight_obstacle * obstacle_score)
        
        # 6. 选择最优转向角
        if total_score > best_score:
            best_score = total_score
            best_steer = steer
    
    if not collision_avoidable:
        print("Collision cannot be avoided!")
        return 0, 0

    return best_steer, speed

def predict_trajectory(x, y, theta, v, steer, L, dt, predict_time):
    """使用自行车模型预测未来轨迹"""
    traj = []
    time = 0
    state = (x, y, theta)
    action = (steer, v)
    
    while time <= predict_time:
        traj.append([state[0], state[1]])
        state = bicycle_model(state, action)
        time += dt
    
    return np.array(traj)

def calculate_goal_score(traj, goal):
    """计算轨迹终点到目标点的距离评分"""
    if len(traj) == 0:
        return 0.0
        
    end_x, end_y = traj[-1]
    dx = goal[0] - end_x
    dy = goal[1] - end_y
    distance = np.sqrt(dx**2 + dy**2)
    
    # 距离越近评分越高，最大值为1
    return 1.0 / (1.0 + distance)

def calculate_smooth_score(steer, current_steer, max_steer):
    """计算转向平滑性评分，与当前转向角差异越小评分越高"""
    diff = abs(steer - current_steer)
    return 1.0 - (diff / (2 * max_steer))  # 归一化到0-1范围

def calculate_distance_to_obstacle(traj, obstacles):
    """计算轨迹到最近障碍物的距离，小于安全半径则返回负值"""
    width_bounds = ENV_CONFIG['gridnum_width'] * ENV_CONFIG['cell_size']
    height_bounds = ENV_CONFIG['gridnum_height'] * ENV_CONFIG['cell_size']
    min_dist = np.inf
    weighted_dist = 0.0
    for i, (x,y) in enumerate(traj):
        #check collision with neighbor cells
        cell_x = int(x / ENV_CONFIG['cell_size'])
        cell_y = int(y / ENV_CONFIG['cell_size'])

            
        # Check collision with neighboring cells
        collision_radius = ENV_CONFIG['agent_collision_radius']

        # Check if out of bounds
        if x < collision_radius or x > width_bounds - collision_radius or y < collision_radius or y > height_bounds - collision_radius:
            return -1.0
            
        # Check for collision with obstacles
        # First check if we're in an obstacle cell
        if (cell_x, cell_y) in obstacles:
            return -1.0
        for dx, dy in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
            neighbor_x = cell_x + dx
            neighbor_y = cell_y + dy
            
            if (neighbor_x, neighbor_y) in obstacles:
                neighbor_center_x = (neighbor_x + 0.5) * ENV_CONFIG['cell_size']
                neighbor_center_y = (neighbor_y + 0.5) * ENV_CONFIG['cell_size']
                
                # Check multiple points on the obstacle cell
                for x_check, y_check in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
                    check_x = neighbor_center_x + x_check * ENV_CONFIG['cell_size'] / 2
                    check_y = neighbor_center_y + y_check * ENV_CONFIG['cell_size'] / 2
                    if np.sqrt((x - check_x) ** 2 + (y - check_y) ** 2) < collision_radius:
                        return -1.0
                    if np.sqrt((x - check_x) ** 2 + (y - check_y) ** 2) < min_dist:
                        min_dist = np.sqrt((x - check_x) ** 2 + (y - check_y) ** 2)
                        weighted_dist = min_dist * (len(traj) - i) / len(traj)
    return weighted_dist

def check_bounds(x, y, width_bounds, height_bounds):
    """检查点(x, y)是否在环境边界内"""
    return x < 0 or x > width_bounds or y < 0 or y > height_bounds

def check_cell(x, y, cell_x, cell_y):
    """检查点(x, y)是否在网格(cell_x, cell_y)内"""
    cell_size = ENV_CONFIG['cell_size']
    cell_center_x = (cell_x + 0.5) * cell_size
    cell_center_y = (cell_y + 0.5) * cell_size
    for dx in np.linspace(-cell_size/2, cell_size/2, 5):
        for dy in np.linspace(-cell_size/2, cell_size/2, 5):
            if np.hypot(x - (cell_center_x + dx), y - (cell_center_y + dy)) < ENV_CONFIG['agent_collision_radius']:
                return True
    return False

def calculate_obstacle_score(distance):
    """计算障碍物评分，距离越远评分越高"""
    # 使用sigmoid函数映射，距离越近评分下降越快
    return 1.0 - np.exp(-distance)
