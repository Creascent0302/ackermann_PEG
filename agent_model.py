import numpy as np
from config import BICYCLE_MODEL_CONFIG, SIMULATION_CONFIG
# A bicycle model for agents
def bicycle_model(state, action):

    if len(state) != 3 or len(action) != 2:
        raise ValueError("State must be a tuple of (x, y, theta) and action must be a tuple of (steering_angle, speed).")
    
    wheelbase = BICYCLE_MODEL_CONFIG['wheelbase']
    dt = SIMULATION_CONFIG['time_step']
    #clip
    steering_angle = np.clip(action[0], -BICYCLE_MODEL_CONFIG['max_steering_angle'], BICYCLE_MODEL_CONFIG['max_steering_angle'])
    #print(BICYCLE_MODEL_CONFIG['max_steering_angle'],steering_angle)
    speed = np.clip(action[1], 0, BICYCLE_MODEL_CONFIG['max_speed'])
    x, y, theta = state
    steering_angle, speed = action

    # 1. 先用当前时刻的航向角计算位置更新
    x += speed * np.cos(theta) * dt
    y += speed * np.sin(theta) * dt

    # 2. 再用当前控制量更新航向角
    theta += (2*speed*np.sin(np.arctan(0.5*np.tan(steering_angle)))) * dt/ wheelbase

    # 3. 归一化航向角到[-π, π]范围，避免角度累积溢出
    theta = (theta + np.pi) % (2 * np.pi) - np.pi
    return (x, y, theta)
