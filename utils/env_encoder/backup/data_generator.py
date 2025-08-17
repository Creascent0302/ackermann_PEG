import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import sys
import random
import os
sys.path.append('.')
sys.path.append('..')

from env import GameENV
from utils.astar import astar_search
from config import ENV_CONFIG

class PathDataGenerator:
    """路径规划数据生成器"""

    def __init__(self, grid_width=ENV_CONFIG['gridnum_width'], grid_height=ENV_CONFIG['gridnum_height']):
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.game_env = GameENV()
        self.game_env.obstacle_probability = 0.35
    
    def generate_dataset(self, num_samples):
        """生成路径规划数据集"""
        dataset = []
        env_count = 0
        
        print(f"正在生成 {num_samples} 个样本...")
        
        while len(dataset) < num_samples:
            # 每5个样本重置环境
            if env_count % 5 == 0:
                self.game_env.reset(num_pursuers=0)
                obstacles = self.game_env.obstacles
            
            env_count += 1
            
            # 生成随机起点数量（1到3个）
            num_starts = random.randint(1, 3)
            starts = []
            for _ in range(num_starts):
                while True:
                    start = self._generate_random_position(obstacles)
                    if start not in starts:
                        starts.append(start)
                        break
            
            # 生成终点，确保与所有起点有一定距离
            while True:
                goal = self._generate_random_position(obstacles)
                if all(self._euclidean_distance(start, goal) >= 5 for start in starts):
                    break
            
            path = []
            for start in starts:
                single_path = astar_search(
                    (start[0]*ENV_CONFIG['cell_size'], start[1]*ENV_CONFIG['cell_size']),
                    (goal[0]*ENV_CONFIG['cell_size'], goal[1]*ENV_CONFIG['cell_size']),
                    [(obs[0], obs[1]) for obs in obstacles]
                )
                if single_path:
                    path.extend(single_path)
            # 只保留有效路径
            if path and 2 <= len(path) <= 100:
                grid_path = [(int(p[0]), int(p[1])) for p in path]
                
                # 创建输入和目标张量 - 修改为符合要求的格式
                env_image = self._create_environment_image(obstacles, starts, goal)
                path_image = self._create_binary_path_image(grid_path)
                
                dataset.append({
                    'env_image': env_image,
                    'path_image': path_image,
                    'start': starts,
                    'goal': goal,
                    'path': grid_path
                })
                if len(dataset) % 50 == 0:
                    print(f"已生成 {len(dataset)}/{num_samples} 个样本")
            
            # 避免无限循环
            if env_count > num_samples * 5:
                break
        
        return dataset
    
    def _generate_random_position(self, obstacles):
        """生成随机位置（不在障碍物上）"""
        while True:
            x = np.random.randint(0, self.grid_width)
            y = np.random.randint(0, self.grid_height)
            if (x, y) not in obstacles:
                return (x, y)
    
    def _euclidean_distance(self, pos1, pos2):
        """计算欧几里得距离"""
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    def _create_environment_image(self, obstacles, starts, goal):
        """创建环境图像：[起点, 终点, 障碍物] - 按照要求重新排序通道"""
        env_image = np.zeros((3, self.grid_height, self.grid_width), dtype=np.float32)
        
        # 起点层 - 通道0
        for start in starts:
            env_image[0, start[1], start[0]] = 1.0
        
        # 终点层 - 通道1
        env_image[1, goal[1], goal[0]] = 1.0
        
        # 障碍物层 - 通道2
        for obs_x, obs_y in obstacles:
            if 0 <= obs_x < self.grid_width and 0 <= obs_y < self.grid_height:
                env_image[2, obs_y, obs_x] = 1.0
        
        return env_image
    
    def _create_binary_path_image(self, path):
        """创建二值路径图像 - 路径上为1，其他为0"""
        path_image = np.zeros((1, self.grid_height, self.grid_width), dtype=np.float32)
        
        # 设置路径点为1
        for x, y in path:
            if 0 <= x < self.grid_width and 0 <= y < self.grid_height:
                path_image[0, y, x] = 1.0
        
        return path_image

class PathPlanningDataset(Dataset):
    """PyTorch数据集封装"""
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform  # 数据增强变换
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        env_image = torch.tensor(sample['env_image'], dtype=torch.float32)
        path_image = torch.tensor(sample['path_image'], dtype=torch.float32)
        
        # 应用数据增强
        if self.transform:
            env_image, path_image = self.transform(env_image, path_image)
        
        return {
            'input': env_image,  # 3通道: [起点, 终点, 障碍物]
            'target': path_image, # 1通道: 二值路径图
            'start': sample['start'],
            'goal': sample['goal'],
            'path': sample['path']
        }

if __name__ == "__main__":
    # 配置 - 使用15x30的网格
    grid_width = ENV_CONFIG['gridnum_width']
    grid_height =  ENV_CONFIG['gridnum_height']
    num_samples = 10  # 生成样本数量
    
    # 创建数据生成器
    generator = PathDataGenerator(grid_width, grid_height)
    
    # 生成数据集
    dataset = generator.generate_dataset(num_samples)
    
    # 创建PyTorch数据集
    path_planning_dataset = PathPlanningDataset(dataset)
    
    # 创建数据加载器
    data_loader = DataLoader(path_planning_dataset, batch_size=32, shuffle=True)
    
    print(f"生成的数据集大小: {len(path_planning_dataset)}")
    print(f"输入形状: {path_planning_dataset[0]['input'].shape}")
    print(f"输出形状: {path_planning_dataset[0]['target'].shape}")
    
    # 使用pygame可视化生成的路径
    import pygame
    pygame.init()
    
    # 设置合理的渲染尺寸缩放因子
    render_size = 20  # 缩小渲染尺寸使其适合屏幕
    
    # 计算实际窗口尺寸
    window_width = grid_width * render_size
    window_height = grid_height * render_size
    
    screen = pygame.display.set_mode((window_width, window_height))
    pygame.display.set_caption("Path Planning Visualization")
    
    # 显示生成的路径
    for i, sample in enumerate(dataset):
        screen.fill((255, 255, 255))
        
        # 绘制障碍物
        obstacles_channel = sample['env_image'][2]
        for y in range(grid_height):
            for x in range(grid_width):
                if obstacles_channel[y, x] > 0:
                    pygame.draw.rect(screen, (0, 0, 0), 
                                    (x * render_size, y * render_size,
                                     render_size, render_size))
        
        # 绘制路径
        path_image = sample['path_image']
        for y in range(grid_height):
            for x in range(grid_width):
                if path_image[0, y, x] > 0:
                    pygame.draw.rect(screen, (0, 255, 0), 
                                    (x * render_size, y * render_size,
                                     render_size, render_size))
        
        # 绘制起点和终点
        start = sample['start']
        goal = sample['goal']
        for s in start:
            pygame.draw.rect(screen, (255, 0, 0), 
                            (s[0] * render_size, s[1] * render_size,
                             render_size, render_size))
        pygame.draw.rect(screen, (0, 0, 255), 
                        (goal[0] * render_size, goal[1] * render_size,
                         render_size, render_size))
        
        pygame.display.flip()
        print(f"显示样本 {i+1}/{len(dataset)}")
        
        # 等待用户按键
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    running = False
    
    pygame.quit()
    print("路径可视化完成")