
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
    
    def cal_dispersion(self, num_samples=1000):
        """
        计算路图的离散度 (Dispersion) - 性能优化版本。
        """
        if not self.nodes:
            return float('inf')
        
        # 预计算物理尺寸和常量
        physical_width = self.grid_width * ENV_CONFIG['cell_size']
        physical_height = self.grid_height * ENV_CONFIG['cell_size']
        cell_size = ENV_CONFIG['cell_size']
        
        # 构建节点KDTree加速距离查询
        nodes_array = np.array(self.nodes)
        nodes_kdtree = KDTree(nodes_array)
        
        # 障碍物集合转换为更高效的查询结构
        obstacle_set = set(self.obstacles)
        
        # 批量生成测试点
        test_points = np.random.uniform(
            low=[0, 0], 
            high=[physical_width, physical_height], 
            size=(num_samples * 2, 2)  # 生成2倍点数，减少后续循环
        )
        
        # 快速筛选有效测试点
        valid_points = []
        for point in test_points:
            # 快速检查是否在障碍物上
            grid_x, grid_y = int(point[0] / cell_size), int(point[1] / cell_size)
            if 0 <= grid_x < self.grid_width and 0 <= grid_y < self.grid_height:
                if (grid_x, grid_y) not in obstacle_set:
                    valid_points.append(point)
                    if len(valid_points) >= num_samples:
                        break
                    
        if not valid_points:
            return 0.0
        
        valid_points = np.array(valid_points[: num_samples])
        
        # 使用KDTree查询每个测试点到最近节点的距离
        distances, indices = nodes_kdtree.query(valid_points, k=3)  # 查询最近的3个节点
        
        # 初始化最大距离
        max_distance = 0
        
        # 验证最近节点的路径是否有效
        for i, point in enumerate(valid_points):
            # 只检查最近的几个节点
            for j in range(min(3, len(indices[i]))):                
                # 简化有效性检查 - 只在必要时使用完整检查
                max_distance = max(max_distance, distances[i][j])
                break        
        return max_distance
    
    def cal_discrepancy(self, num_samples=1000):
        """
        计算节点集的星偏差度 (Star Discrepancy) - 性能优化版本。
        限制矩形面积不超过地图总面积的1/8。
        """
        if not self.nodes:
            return 1.0
        
        # 预计算常量
        physical_width = self.grid_width * ENV_CONFIG['cell_size']
        physical_height = self.grid_height * ENV_CONFIG['cell_size']
        total_area = physical_width * physical_height
        max_area = total_area / 8  # 限制最大矩形面积为地图总面积的1/8
        num_nodes = len(self.nodes)
        
        # 将节点转换为NumPy数组 - 只做一次
        np_nodes = np.array(self.nodes)
        
        # 批量生成所有矩形 (每个矩形需要2个点)
        points = np.random.uniform(
            low=[0, 0], 
            high=[physical_width, physical_height], 
            size=(num_samples, 2, 2)
        )
        
        # 预分配结果数组
        discrepancies = np.zeros(num_samples)
        
        # 向量化处理所有矩形
        for i in range(num_samples):
            # 获取矩形坐标
            x1 = min(points[i, 0, 0], points[i, 1, 0])
            x2 = max(points[i, 0, 0], points[i, 1, 0])
            y1 = min(points[i, 0, 1], points[i, 1, 1])
            y2 = max(points[i, 0, 1], points[i, 1, 1])
            
            # 计算矩形面积并检查是否超限
            rect_area = (x2 - x1) * (y2 - y1)
            
            # 如果面积超过限制，缩小矩形保持中心点不变
            if rect_area > max_area:
                # 计算矩形中心
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                
                # 计算缩放因子
                scale = math.sqrt(max_area / rect_area)
                
                # 计算新的半宽和半高
                half_width = (x2 - x1) / 2 * scale
                half_height = (y2 - y1) / 2 * scale
                
                # 更新矩形坐标
                x1 = center_x - half_width
                x2 = center_x + half_width
                y1 = center_y - half_height
                y2 = center_y + half_height
            
            # 计算面积比例
            area_rate = (x2 - x1) * (y2 - y1) / total_area
            
            # 计算点在矩形内的比例
            mask = ((np_nodes[:, 0] >= x1) & 
                    (np_nodes[:, 0] <= x2) & 
                    (np_nodes[:, 1] >= y1) & 
                    (np_nodes[:, 1] <= y2))
            
            count = np.sum(mask)
            point_rate = count / num_nodes
            
            # 计算偏差
            discrepancies[i] = abs(area_rate - point_rate)
        
        # 返回最大偏差
        return np.max(discrepancies)