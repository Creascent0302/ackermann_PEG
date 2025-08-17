import numpy as np
import sys
sys.path.append('..')
from config import ENV_CONFIG, BICYCLE_MODEL_CONFIG

def pure_pursuit_control(agent_state, target_point):
    """
    Pure Pursuit control for a pursuer in the environment.

    Args:
        pursuer_state: Current state of the pursuer (x, y, theta).
        target_point: The target point (x, y) to pursue.

    Returns:
        tuple: (steering_angle, speed) - 控制指令
    """
    # 提取当前状态
    x, y, theta = agent_state
    tx, ty = target_point

    # 计算到目标点的距离
    dx = tx - x
    dy = ty - y
    distance_to_target = np.hypot(dx, dy)
 
    # 将目标点从全局坐标系转换到车辆坐标系
    x_look = dx * np.cos(theta) + dy * np.sin(theta)
    y_look = -dx * np.sin(theta) + dy * np.cos(theta)

    # 纯跟踪算法计算转向角
    L = BICYCLE_MODEL_CONFIG['wheelbase']
    
    # 安全处理边界情况
    if abs(x_look) < 1e-6 and abs(y_look) < 1e-6:
        # 目标点与车辆重合
        steering_angle = 0.0
    elif abs(x_look) < 1e-6:
        # 目标点在车辆正左/右方向
        steering_angle = np.sign(y_look) * BICYCLE_MODEL_CONFIG['max_steering_angle']
    else:
        # 标准纯跟踪计算
        curvature = 2 * y_look / (x_look**2 + y_look**2)
        steering_angle = np.arctan(curvature * L)
        
        # 限制最大转向角
        max_steer = BICYCLE_MODEL_CONFIG.get('max_steering_angle', np.pi/4)
        steering_angle = np.clip(steering_angle, -max_steer, max_steer)

    # 计算速度
    speed = BICYCLE_MODEL_CONFIG['max_speed']
    # 返回控制指令
    return (steering_angle, speed)