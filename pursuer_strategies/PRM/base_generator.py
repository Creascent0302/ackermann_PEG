
import pygame
import numpy as np
from scipy.spatial import KDTree
import sys
sys.path.append('.')
from config import ENV_CONFIG
import math
import heapq
from abc import ABC, abstractmethod
import time

class BasePathPlanner(ABC):
    """路径规划算法基类"""
    
    def __init__(self, grid_width, grid_height, obstacles, **kwargs):
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.obstacles = set(obstacles)  # 转换为set提高查询效率
        self.nodes = []
        self.edges = []
        
        # 通用参数
        self.collision_radius = kwargs.get('collision_radius', 2 * ENV_CONFIG['agent_collision_radius'] / 3)
        self.step_size = kwargs.get('step_size', 0.1)
        
        # 构建障碍物KDTree用于加速碰撞检测
        self._build_obstacle_kdtree()
        
    def _build_obstacle_kdtree(self):
        """构建障碍物KDTree用于加速碰撞检测"""
        if not self.obstacles:
            self.obstacle_kdtree = None
            return
        cs = ENV_CONFIG['cell_size']
        pts = [((ox + 0.5) * cs, (oy + 0.5) * cs) for (ox, oy) in self.obstacles]
        self.obstacle_kdtree = KDTree(pts)
    
    def _is_valid_position(self, x, y, dr=None):
        """检查位置是否有效（不在障碍物附近且在边界内）"""
        if dr is None:
            dr = 0.015
            
        # 边界检查
        if x < dr or x >= self.grid_width * ENV_CONFIG['cell_size'] - dr:
            return False
        if y < dr or y >= self.grid_height * ENV_CONFIG['cell_size'] - dr:
            return False
        
        # 障碍物检查 - 使用多点采样
        check_points = [
            (x + dr, y), (x - dr, y), (x, y + dr), (x, y - dr),
            (x + dr/np.sqrt(2), y + dr/np.sqrt(2)), 
            (x + dr/np.sqrt(2), y - dr/np.sqrt(2)),
            (x - dr/np.sqrt(2), y + dr/np.sqrt(2)), 
            (x - dr/np.sqrt(2), y - dr/np.sqrt(2)),
            (x, y)
        ]
        
        cs = ENV_CONFIG['cell_size']
        for px, py in check_points:
            gx, gy = int(px / cs), int(py / cs)
            if gx < 0 or gx >= self.grid_width or gy < 0 or gy >= self.grid_height:
                return False
            if (gx, gy) in self.obstacles:
                return False
                
        return True
    
    def _is_valid_edge(self, node1, node2, num_samples=None):
        """检查边是否有效（沿线段采样检测碰撞）"""
        x1, y1 = node1
        x2, y2 = node2
        
        length = math.hypot(x2 - x1, y2 - y1)
        if num_samples is None:
            num_samples = max(1, int(length * 20))
        
        for i in range(num_samples + 1):
            t = i / num_samples
            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)
            if not self._is_valid_position(x, y):
                return False
        return True
    
    def _distance(self, node1, node2):
        """计算两节点间欧几里得距离"""
        return math.hypot(node1[0] - node2[0], node1[1] - node2[1])
    
    def _random_sample(self):
        """在自由空间中随机采样一个点"""
        max_attempts = 1000
        for _ in range(max_attempts):
            x = np.random.uniform(0, self.grid_width * ENV_CONFIG['cell_size'])
            y = np.random.uniform(0, self.grid_height * ENV_CONFIG['cell_size'])
            if self._is_valid_position(x, y):
                return (x, y)
        return None
    
    @abstractmethod
    def generate_prm(self):
        """生成路径图 - 抽象方法，子类必须实现"""
        pass
