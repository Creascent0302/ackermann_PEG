import numpy as np
from config import ENV_CONFIG, BICYCLE_MODEL_CONFIG, SIMULATION_CONFIG
from agent_model import bicycle_model
from gymnasium import  spaces

class GameENV:
    def __init__(self,num_pursuers,seed):
        self.gridnum_width = ENV_CONFIG['gridnum_width']
        self.gridnum_height = ENV_CONFIG['gridnum_height']
        self.obstacle_probability = ENV_CONFIG['obstacle_probability']
        self.gridsize = ENV_CONFIG['cell_size']
        self.seed = seed
        np.random.seed(seed)  # 设置随机种子以确保可重复性
        # 初始化环境状态
        self.obstacles = self._generate_grid_obstacles()
        self.evader_state = None
        self.pursuers_states = []
        self.num_pursuers = num_pursuers
        self.time_step = 0
        self.max_steps = SIMULATION_CONFIG['max_steps']
        self.dt = SIMULATION_CONFIG['time_step']
        
        # 游戏状态
        self.done = False
        self.evader_collision = False
        self.pursuer_collision = np.zeros(self.num_pursuers, dtype=int)
        self.capture = False
        self.pursuers_collision = False
        self.capture_distance = ENV_CONFIG['agent_capture_distance']

        # 设置观测空间
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(3 + num_pursuers * 3 + len(self.obstacles) * 2 + 1,),  # evader_state + pursuers_states + obstacles + time_step
            dtype=np.float32
        )
        # 设置动作空间
        self.action_space = spaces.Dict({
            "evader": spaces.Box(low=np.array([-BICYCLE_MODEL_CONFIG['max_steering_angle'], 0]),\
                                  high=np.array([BICYCLE_MODEL_CONFIG['max_steering_angle'], BICYCLE_MODEL_CONFIG['max_speed']]), dtype=np.float32),\
            "pursuers": spaces.Box(low=np.array([-BICYCLE_MODEL_CONFIG['max_steering_angle'], 0.0] )* num_pursuers,\
                                   high=np.array([BICYCLE_MODEL_CONFIG['max_steering_angle'], BICYCLE_MODEL_CONFIG['max_speed']]) * num_pursuers,\
                                   dtype=np.float32)
        })

    def _generate_grid_obstacles(self):
        """生成网格障碍物"""
        obstacles = []
        for x in range(self.gridnum_width):
            for y in range(self.gridnum_height):
                # 边界不放障碍物
                if x == 0 or x == self.gridnum_width-1 or y == 0 or y == self.gridnum_height-1:
                    continue
                
                if np.random.random() < self.obstacle_probability:
                    obstacles.append((x, y))
        return obstacles

    def _is_valid_position(self, x, y):
        """检查位置是否有效（不在障碍物内且在边界内）"""
        collision_radius = ENV_CONFIG['agent_collision_radius']
        # 检查边界
        if x < collision_radius or x >= self.gridnum_width*self.gridsize-collision_radius or y < collision_radius or y >= self.gridnum_height*self.gridsize-collision_radius:
            return False
        
        # 检查网格障碍物
        grid_x = int(x/ self.gridsize)
        grid_y = int(y/ self.gridsize)
        if (grid_x, grid_y) in self.obstacles:
            return False
        #check the collision with neighbor grids
        for (dx, dy) in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
                neighbor_x = grid_x + dx
                neighbor_y = grid_y + dy
                #check the 8 points on the neighbor grids
                if (neighbor_x, neighbor_y) in self.obstacles:
                    neighbor_center_x = (neighbor_x + 0.5) * self.gridsize
                    neighbor_center_y = (neighbor_y + 0.5) * self.gridsize
                    for (x_check,y_check) in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
                        check_x = neighbor_center_x + x_check * self.gridsize / 2
                        check_y = neighbor_center_y + y_check * self.gridsize / 2
                        if np.sqrt((x - check_x) ** 2 + (y - check_y) ** 2) < collision_radius:
                            return False  
        return True


    def _generate_random_valid_position(self):
        """
        检查当前位置及其三个方向是否有效，如果有任何位置无效则重新采样，
        直到找到所有位置都有效的解。
        
        Args:
            x, y: 当前位置坐标
            theta: 当前朝向角度
            
        Returns:
            有效的位置和角度 (x, y, theta)
        """
        # 定义采样范围和最大尝试次数，避免无限循环
        max_attempts = 400  # 最大尝试次数
        attempt = 0
        
        x_range = (0, self.gridnum_width * self.gridsize)
        y_range = (0, self.gridnum_height * self.gridsize)
        x = np.random.uniform(*x_range)
        y = np.random.uniform(*y_range)
        theta = np.random.uniform(0, 2 * np.pi)       
        while attempt < max_attempts:
            # 检查三个方向的预览位置
            directions = [theta - np.pi/3,theta - np.pi/6, theta, theta + np.pi/6, theta + np.pi/3]
            all_valid = True
                
            for direction in directions:
                for dist in [-0.15,0, 0.1, 0.3]:
                    preview_pos = (
                        x + dist * np.cos(direction),
                        y + dist * np.sin(direction)
                    )
                    if not self._is_valid_position(*preview_pos):
                        all_valid = False
                        break  # 发现无效方向，无需检查其他方向
                if not all_valid:
                    break
            if all_valid:
                return x, y, theta  # 所有位置都有效，返回当前位置
            
            attempt += 1
            
            # 重新采样位置和角度
            x = np.random.uniform(*x_range)
            y = np.random.uniform(*y_range)
            theta = np.random.uniform(0, 2 * np.pi)
        if attempt >= max_attempts:
            print("Failed to find a valid position after 100 attempts.")
        # 如果达到最大尝试次数仍未找到有效位置，返回最后一次尝试的结果
        return x, y, theta


    def reset(self,obstacle_change = True):
        """重置环境"""
        num_pursuers = self.num_pursuers
        self.time_step = 0
        self.done = False
        self.evader_collision = False
        self.pursuer_collision = np.zeros(self.num_pursuers, dtype=int)
        # 重新生成障碍物
        if obstacle_change:
            self.obstacles = self._generate_grid_obstacles()
 
        # 随机初始化追捕者位置
        self.pursuers_states = []
        for _ in range(num_pursuers):
            pursuer_x, pursuer_y, pursuer_theta = self._generate_random_valid_position()
            self.pursuers_states.append((pursuer_x, pursuer_y, pursuer_theta))

        # 随机初始化逃避者位置
        # Set minimum distance the evader should be from pursuers at initialization
        min_init_distance = 2
        max_attempts = 200
        attempt = 0

        while attempt < max_attempts:

            # Generate a random valid position
            evader_x, evader_y, evader_theta = self._generate_random_valid_position()

            self.evader_state = (evader_x, evader_y, evader_theta)
            # Check distance to all pursuers
            too_close = False
            for pursuer_state in self.pursuers_states:
                pursuer_x, pursuer_y, _ = pursuer_state
                distance = np.sqrt((evader_x - pursuer_x)**2 + (evader_y - pursuer_y)**2)
                if distance < min_init_distance:
                    too_close = True
                    break
            
            # If we found a good position, break out of the loop
            if not too_close:
                break
            
            attempt += 1
            if attempt >= max_attempts:
                print("Failed to find a valid initial position for evader after 100 attempts.")

        return self._get_observation()

    def step(self,actions):
        """执行一步环境更新"""
        if self.done:
            return self._get_observation(), self._get_reward(), self.done, {}
 
        # 更新逃避者状态
        if 'evader' in actions:
            evader_action = actions['evader']
            steering_angle = np.clip(evader_action[0], 
                                   -BICYCLE_MODEL_CONFIG['max_steering_angle'],
                                   BICYCLE_MODEL_CONFIG['max_steering_angle'])
            speed = np.clip(evader_action[1], 0, BICYCLE_MODEL_CONFIG['max_speed'])
            
            new_state = bicycle_model(self.evader_state, (steering_angle, speed))
            
            if self._is_valid_position(new_state[0], new_state[1]):
                
                self.evader_state = new_state
            else:
                self.evader_collision = True  # 如果逃避者碰撞，设置标志位

        # 更新追捕者状态
        if 'pursuers' in actions:
            pursuers_actions = actions['pursuers']
            for i, pursuer_action in enumerate(pursuers_actions):
                if i < len(self.pursuers_states):
                    steering_angle = np.clip(pursuer_action[0],
                                           -BICYCLE_MODEL_CONFIG['max_steering_angle'],
                                           BICYCLE_MODEL_CONFIG['max_steering_angle'])
                    speed = np.clip(pursuer_action[1], 0, BICYCLE_MODEL_CONFIG['max_speed'])
                    new_state = bicycle_model(self.pursuers_states[i], (steering_angle, speed))
                    if self._is_valid_position(new_state[0], new_state[1]):
                        
                        self.pursuers_states[i] = new_state
                    else:
                        self.pursuer_collision[i] = 1

        
        # 检查游戏结束条件
        self._check_done()
        
        # 更新时间步
        self.time_step += 1
        if self.time_step >= self.max_steps:
            self.done = True
        
        return self._get_observation(), self._get_reward(), self.done, self._get_info()

    def _check_done(self):
        """检查游戏是否结束"""
        evader_x, evader_y, _ = self.evader_state
        if self.evader_collision:
            self.done =True
            #print("Evader collided with an obstacle!")
            #print("Evader state:", self.evader_state)
            return 
        elif all(self.pursuer_collision):
            self.capture = False
            self.pursuers_collision = True
            self.done =True
            #print("All pursuers collided with obstacles!")
            return 
        if any(np.sqrt((evader_x - px)**2 + (evader_y - py)**2) < self.capture_distance 
               for px, py, _ in self.pursuers_states):
            self.capture = True
            self.done = True
            #print("Evader has been captured!")
            return

    def _get_observation(self):
        """获取环境观测"""
        return {
            'evader_state': self.evader_state,
            'pursuers_states': self.pursuers_states,
            'obstacles': self.obstacles,
            'time_step': self.time_step
        }

    def _get_reward(self):
        """计算奖励"""
        if self.done and self.capture:
            if self.time_step < self.max_steps:
                return {'evader': -100, 'pursuers': 100}
            else:
                return {'evader': 100, 'pursuers': -100}
        
        if self.done and self.pursuers_collision:
            return {'evader': 0, 'pursuers': -100}
        if self.done and self.evader_collision:
            return {'evader': -100, 'pursuers': 0}
        evader_x, evader_y, _ = self.evader_state
        min_distance = float('inf')
        
        for pursuer_state in self.pursuers_states:
            pursuer_x, pursuer_y, _ = pursuer_state
            distance = np.sqrt((evader_x - pursuer_x)**2 + (evader_y - pursuer_y)**2)
            min_distance = min(min_distance, distance)

        evader_reward = min_distance / (self.gridnum_width + self.gridnum_height)
        pursuer_reward = -evader_reward
        
        return {'evader': evader_reward, 'pursuers': pursuer_reward}

    def _get_info(self):
        """获取额外信息"""
        evader_x, evader_y, _ = self.evader_state
        distances_to_pursuers = []
        
        for pursuer_state in self.pursuers_states:
            pursuer_x, pursuer_y, _ = pursuer_state
            distance = np.sqrt((evader_x - pursuer_x)**2 + (evader_y - pursuer_y)**2)
            distances_to_pursuers.append(distance)
        
        return {
            'distances_to_pursuers': distances_to_pursuers,
            'min_distance': min(distances_to_pursuers) if distances_to_pursuers else float('inf'),
            'capture_distance': self.capture_distance
        }