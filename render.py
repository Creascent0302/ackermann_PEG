import pygame
import numpy as np
from config import ENV_CONFIG, PYGAME_CONFIG
import time

class PygameRenderer:
    def __init__(self, env_width, env_height):
        self.gridnum_width = env_width
        self.gridnum_height = env_height
        self.cell_size = ENV_CONFIG['cell_size']  # 网格单元大小
        self.render_size = PYGAME_CONFIG['render_size'] # 渲染大小
        self.window_width = self.gridnum_width * self.cell_size * self.render_size
        self.window_height = self.gridnum_height * self.cell_size * self.render_size
        
        # 初始化pygame
        pygame.init()
        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption("追逃游戏")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, PYGAME_CONFIG['font_size'])
        
        # 轨迹存储
        self.evader_trail = []
        self.pursuers_trails = []
        self.max_trail_length = 20 # 最大轨迹长度
        # 新增: 复用的轨迹 Surface，避免频繁创建
        self.trail_surface = pygame.Surface((self.window_width, self.window_height), pygame.SRCALPHA)
        # 可调稀疏目标段数（越小越稀疏，越快）
        self.trail_target_segments = 8
    
    def world_to_screen(self, x, y):
        """将世界坐标转换为屏幕坐标"""
        screen_x = int(x * self.render_size)
        screen_y = int((self.gridnum_height * self.cell_size - y) * self.render_size)
        return screen_x, screen_y
    
    def draw_agent(self, pos, color, size):
        """绘制智能体"""
        x, y, theta = pos
        screen_x, screen_y = self.world_to_screen(x, y)
        
        # Define the dimensions of the agent
        length = size * 2  # Length of the agent (wheelbase)
        width = size  # Width is half of the length
        
        # Calculate the rectangle's corners based on the heading (theta)
        half_length = length / 2
        half_width = width / 2
        corners = [
            (-half_length, -half_width),
            (half_length, -half_width),
            (half_length, half_width),
            (-half_length, half_width)
        ]
        
        # Rotate the corners around the center based on theta
        rotated_corners = [
            (
            screen_x + corner[0] * np.cos(theta) - corner[1] * np.sin(theta),
            screen_y - (corner[0] * np.sin(theta) + corner[1] * np.cos(theta))  # Adjust for screen coordinates
            )
            for corner in corners
        ]
        
        # Draw the rotated rectangle
        pygame.draw.polygon(self.screen, color, rotated_corners)
        pygame.draw.circle(self.screen, color, (screen_x, screen_y), int(size), 1)
        # Draw the direction arrow
        arrow_length = length * 0.75
        end_x = screen_x + arrow_length * np.cos(theta)
        end_y = screen_y - arrow_length * np.sin(theta)  # Adjust for screen coordinates
        pygame.draw.line(self.screen, PYGAME_CONFIG['colors']['BLACK'], 
                 (screen_x, screen_y), (end_x, end_y), int(3))
    
    def draw_trail(self, trail, base_color):
        """绘制稀疏&加速的渐变轨迹"""
        if len(trail) < 2:
            return
        r, g, b = base_color
        n = len(trail)

        # 计算步长，使得线段总数不超过 target
        # 例如 n=15, target=8 -> step=2
        step = max(1, n // self.trail_target_segments)
        indices = list(range(0, n - 1, step))
        if indices[-1] != n - 2:  # 确保最后一段包含
            indices.append(n - 2)

        total_segments = len(indices)
        if total_segments <= 0:
            return

        for seg_idx, i in enumerate(indices):
            start_pos = trail[i]
            end_pos = trail[i + 1]
            # 渐变透明度（前淡后浓）
            alpha = int(60 + 195 * (seg_idx / max(1, total_segments - 1)))
            color = (r, g, b, alpha)

            start_x, start_y = self.world_to_screen(start_pos[0], start_pos[1])
            end_x, end_y = self.world_to_screen(end_pos[0], end_pos[1])
            pygame.draw.line(self.trail_surface, color, (start_x, start_y), (end_x, end_y), 2)

    def update_trails(self, env):
        """更新智能体轨迹"""
        # 更新逃避者轨迹
        # 更新逃避者轨迹
        # 更新逃避者轨迹
        if len(self.evader_trail) == 0 or np.linalg.norm(
            np.array(self.evader_trail[-1]) - np.array((env.evader_state[0], env.evader_state[1]))
        ) > ENV_CONFIG['cell_size']:
            self.evader_trail.clear()
        self.evader_trail.append((env.evader_state[0], env.evader_state[1]))
        if len(self.evader_trail) > self.max_trail_length:
            self.evader_trail.pop(0)
        
        # 确保追捕者轨迹列表与追捕者数量匹配
        while len(self.pursuers_trails) < len(env.pursuers_states):
            self.pursuers_trails.append([])
        
        # 更新每个追捕者的轨迹
        for i, state in enumerate(env.pursuers_states):
            if i < len(self.pursuers_trails):
                if len(self.pursuers_trails[i]) == 0 or np.linalg.norm(
                    np.array(self.pursuers_trails[i][-1]) - np.array((state[0], state[1]))
                ) > ENV_CONFIG['cell_size']:
                    self.pursuers_trails[i].clear()
                self.pursuers_trails[i].append((state[0], state[1]))
                if len(self.pursuers_trails[i]) > self.max_trail_length:
                    self.pursuers_trails[i].pop(0)

    def render(self, env, step_count, path, evader_path, preview_point):
        """渲染一帧"""
        # 处理事件
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
        
        # 更新轨迹
        self.update_trails(env)
        
        # 清屏
        self.screen.fill(PYGAME_CONFIG['colors']['WHITE'])

        if evader_path:
            for i in range(len(evader_path)):
                pygame.draw.rect(self.screen, (200,100,200),
                                (evader_path[i][0] * self.cell_size * self.render_size,
                                (self.gridnum_height - evader_path[i][1] - 1) * self.cell_size * self.render_size,
                                self.cell_size * self.render_size, 
                                self.cell_size * self.render_size))
                #write {i}
                font = pygame.font.Font(None, 24)
                text = font.render(f"{i}", True, (255, 255, 255))
                self.screen.blit(text, (evader_path[i][0] * self.cell_size * self.render_size,
                                         (self.gridnum_height - evader_path[i][1] - 1) * self.cell_size * self.render_size))

        if path:
            for i in range(len(path) - 1):
                screen_x, screen_y = self.world_to_screen(path[i][0], path[i][1])
             
                pygame.draw.rect(self.screen, (0,128,128),
                                 (path[i][0] * self.cell_size * self.render_size,
                                  (self.gridnum_height - path[i][1] - 1) * self.cell_size * self.render_size,
                                  self.cell_size * self.render_size, 
                                  self.cell_size * self.render_size))      

        # render preview point
        if preview_point:
            preview_x, preview_y = preview_point
            preview_screen_x, preview_screen_y = self.world_to_screen(preview_x, preview_y)
            pygame.draw.circle(self.screen, (60,60,60),
                               (preview_screen_x, preview_screen_y), 
                               int(ENV_CONFIG['agent_collision_radius'] * self.render_size), 1)                  

        # 绘制网格
        '''
        for i in range(self.gridnum_width + 1):
            x = i * self.cell_size * self.render_size
            pygame.draw.line(self.screen, PYGAME_CONFIG['colors']['GRAY'], 
                           (x, 0), (x, self.window_height), 1)
        for i in range(self.gridnum_height + 1):
            y = i * self.cell_size * self.render_size
            pygame.draw.line(self.screen, PYGAME_CONFIG['colors']['GRAY'], 
                           (0, y), (self.window_width, y), 1)
        '''
        # 绘制障碍物（网格）
        for obs_x, obs_y in env.obstacles:
            rect_x = obs_x * self.cell_size * self.render_size
            rect_y = (self.gridnum_height - obs_y - 1) * self.cell_size * self.render_size
            pygame.draw.rect(self.screen, PYGAME_CONFIG['colors']['OBSTACLE'], 
                        (rect_x, rect_y, self.cell_size * self.render_size, self.cell_size * self.render_size))


        # 清空轨迹 Surface （仅一次）
        self.trail_surface.fill((0, 0, 0, 0))
        # 绘制轨迹到 trail_surface
        self.draw_trail(self.evader_trail, PYGAME_CONFIG['colors']['EVADER'])
        for i, trail in enumerate(self.pursuers_trails):
            self.draw_trail(trail, PYGAME_CONFIG['colors']['PURSUER'])
        # 一次性叠加
        self.screen.blit(self.trail_surface, (0, 0))
        # 绘制智能体
        self.draw_agent(env.evader_state, PYGAME_CONFIG['colors']['EVADER'], 
                       ENV_CONFIG['agent_collision_radius'] * self.render_size)

        for i, pursuer_state in enumerate(env.pursuers_states):
            self.draw_agent(pursuer_state, PYGAME_CONFIG['colors']['PURSUER'], 
                          ENV_CONFIG['agent_collision_radius'] * self.render_size)
       
        # 绘制捕获范围
        for pursuer_state in env.pursuers_states:
            x, y, _ = pursuer_state
            screen_x, screen_y = self.world_to_screen(x, y)
            pygame.draw.circle(self.screen, PYGAME_CONFIG['colors']['PURSUER'], 
                             (screen_x, screen_y), 
                             int(ENV_CONFIG['agent_capture_distance'] * self.render_size), 1)
        
        # 绘制信息文本
        info_text = f"步数: {step_count}"
        text_surface = self.font.render(info_text, True, PYGAME_CONFIG['colors']['BLACK'])
        self.screen.blit(text_surface, (10, 10))

        # 更新显示
        
        pygame.display.flip()
        self.clock.tick(PYGAME_CONFIG['fps'])
        
        return True
    
    def close(self):
        pygame.quit()