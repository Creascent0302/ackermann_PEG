import numpy as np
import sys
sys.path.append('..')  # 添加上级目录到路径中                                                                                   
from config import BICYCLE_MODEL_CONFIG

def simple_evader_policy(observe):
    """简单的逃避者策略：远离最近的追捕者"""
    evader_state = observe['evader_state']
    pursuers_states = observe['pursuers_states']

    evader_x, evader_y, evader_theta = evader_state
    
    if not pursuers_states:
        return (0.0, 2.0)
    
    # 找到最近的追捕者
    min_distance = float('inf')
    closest_pursuer = None
    
    for pursuer_state in pursuers_states:
        pursuer_x, pursuer_y, _ = pursuer_state
        distance = np.sqrt((evader_x - pursuer_x)**2 + (evader_y - pursuer_y)**2)
        if distance < min_distance:
            min_distance = distance
            closest_pursuer = (pursuer_x, pursuer_y)
    
    if closest_pursuer:
        # 计算远离追捕者的方向
        dx = evader_x - closest_pursuer[0]
        dy = evader_y - closest_pursuer[1]
        escape_angle = np.arctan2(dy, dx)
        
        # 计算转向角度差
        angle_diff = escape_angle - evader_theta
        angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi
        
        # 根据角度差计算转向
        steering_angle = np.clip(angle_diff * 0.5, 
                               -BICYCLE_MODEL_CONFIG['max_steering_angle'],
                               BICYCLE_MODEL_CONFIG['max_steering_angle'])
        
        speed = 3.0 if min_distance < 2.0 else 2.0
    else:
        steering_angle = 0.0
        speed = 2.0
    
    return (steering_angle, speed)
