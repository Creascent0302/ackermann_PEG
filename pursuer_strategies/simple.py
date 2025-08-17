import numpy as np
import sys
sys.path.append('..')  # 添加上级目录到路径中                                                                                   
from config import BICYCLE_MODEL_CONFIG

def simple_pursuer_policy(observation):
    pursuers_actions = []
    for i in range(len(observation['pursuers_states'])):
        pursuer_action = single_simple_pursuer_policy(observation, i)
        pursuers_actions.append(pursuer_action)
    return pursuers_actions

def single_simple_pursuer_policy(observation,self_index):
    """简单的追捕者策略：朝向逃避者"""
    pursuer_x, pursuer_y, pursuer_theta = observation['pursuers_states'][self_index]
    evader_x, evader_y, _ = observation['evader_state']
    
    # 计算朝向逃避者的方向
    dx = evader_x - pursuer_x
    dy = evader_y - pursuer_y
    target_angle = np.arctan2(dy, dx)
    
    # 计算转向角度差
    angle_diff = target_angle - pursuer_theta
    angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi
    
    # 根据角度差计算转向
    steering_angle = np.clip(angle_diff * 0.5,
                           -BICYCLE_MODEL_CONFIG['max_steering_angle'],
                           BICYCLE_MODEL_CONFIG['max_steering_angle'])
    
    distance = np.sqrt(dx**2 + dy**2)
    speed = 4.0 if distance > 1.0 else 2.0
    
    return (steering_angle, speed)