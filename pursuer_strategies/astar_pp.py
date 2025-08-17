import sys
sys.path.append('..')  # 添加上级目录到路径中
from utils.astar_pp_dwa import pursuer_track_target

def astar_pursuer_policy(observation):
    pursuers_actions = []
    for i in range(len(observation['pursuers_states'])):
        pursuer_action, path, preview_point = single_astar_pursuer_policy(observation, i)
        pursuers_actions.append(pursuer_action)
    return pursuers_actions, path, preview_point


def single_astar_pursuer_policy(observation, self_index):
    """
    使用 A* 路径规划和 Pure Pursuit 策略的追捕者策略
    """
    
    pursuer_x, pursuer_y, pursuer_theta = observation['pursuers_states'][self_index]
    evader_x, evader_y, _ = observation['evader_state']
    obstacles = observation['obstacles']
    (steering_angle, speed), path, preview_point = pursuer_track_target((evader_x, evader_y), (pursuer_x, pursuer_y, pursuer_theta), obstacles,self_index)
    return (steering_angle, speed), path, preview_point