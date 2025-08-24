import random
# def generate_maze_obstacles(grid_width, grid_height):
#     """
#     使用深度优先算法生成迷宫，路径宽度为2像素，墙壁为1像素
#     返回所有墙壁像素点的坐标列表
    
#     参数:
#     grid_width: 迷宫宽度（奇数）
#     grid_height: 迷宫高度（奇数）
    
#     返回:
#     walls: 所有墙壁像素点坐标的列表 [(x, y), ...]
#     """
    
#     # 确保宽度和高度都是奇数
#     if grid_width % 2 == 0 or grid_height % 2 == 0:
#         raise ValueError("Grid width and height must be odd numbers")
    
#     # 计算逻辑迷宫的尺寸（每个逻辑节点占用3x3像素：2x2路径 + 1像素边界）
#     # 但为了适应原有的奇数尺寸约束，我们采用不同的方案
#     logic_width = (grid_width + 1) // 3
#     logic_height = (grid_height + 1) // 3
    
#     # 初始化迷宫，所有位置都是墙壁（True表示墙壁，False表示通路）
#     maze = [[True for _ in range(grid_width)] for _ in range(grid_height)]
    
#     # 标记逻辑节点访问状态
#     visited = [[False for _ in range(logic_width)] for _ in range(logic_height)]
    
#     # 四个方向：上、下、左、右
#     directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
#     def is_valid_logic(lx, ly):
#         """检查逻辑坐标是否在范围内"""
#         return 0 <= lx < logic_height and 0 <= ly < logic_width
    
#     def logic_to_physical(lx, ly):
#         """将逻辑坐标转换为物理坐标（左上角）"""
#         px = lx * 3 + 1  # 逻辑节点在物理坐标系中的起始位置
#         py = ly * 3 + 1
#         return px, py
    
#     def carve_node(lx, ly):
#         """雕刻一个逻辑节点（2x2像素区域）"""
#         px, py = logic_to_physical(lx, ly)
#         # 确保不越界
#         for dx in range(2):
#             for dy in range(2):
#                 if px + dx < grid_height and py + dy < grid_width:
#                     maze[px + dx][py + dy] = False
    
#     def carve_passage(lx1, ly1, lx2, ly2):
#         """在两个逻辑节点之间雕刻2像素宽的通道"""
#         px1, py1 = logic_to_physical(lx1, ly1)
#         px2, py2 = logic_to_physical(lx2, ly2)
        
#         # 确定通道方向和位置
#         if lx1 == lx2:  # 水平通道
#             # 连接两个节点之间的垂直通道
#             if ly1 < ly2:  # 向右
#                 # 通道位置在两个节点之间
#                 passage_x = px1
#                 passage_y = py1 + 2  # 第一个节点右侧
#                 for dx in range(2):
#                     if passage_x + dx < grid_height and passage_y < grid_width:
#                         maze[passage_x + dx][passage_y] = False
#             else:  # 向左
#                 passage_x = px1
#                 passage_y = py1 - 1  # 第一个节点左侧
#                 for dx in range(2):
#                     if passage_x + dx < grid_height and passage_y >= 0:
#                         maze[passage_x + dx][passage_y] = False
#         else:  # 垂直通道
#             if lx1 < lx2:  # 向下
#                 passage_x = px1 + 2  # 第一个节点下方
#                 passage_y = py1
#                 for dy in range(2):
#                     if passage_x < grid_height and passage_y + dy < grid_width:
#                         maze[passage_x][passage_y + dy] = False
#             else:  # 向上
#                 passage_x = px1 - 1  # 第一个节点上方
#                 passage_y = py1
#                 for dy in range(2):
#                     if passage_x >= 0 and passage_y + dy < grid_width:
#                         maze[passage_x][passage_y + dy] = False
    
#     def get_unvisited_neighbors(lx, ly):
#         """获取未访问的相邻逻辑节点"""
#         neighbors = []
#         for dx, dy in directions:
#             nx, ny = lx + dx, ly + dy
#             if is_valid_logic(nx, ny) and not visited[nx][ny]:
#                 neighbors.append((nx, ny))
#         return neighbors
    
#     # 递归回溯算法
#     def dfs(lx, ly):
#         visited[lx][ly] = True
#         carve_node(lx, ly)  # 雕刻当前节点
        
#         # 随机选择相邻的未访问节点
#         neighbors = get_unvisited_neighbors(lx, ly)
#         random.shuffle(neighbors)  # 随机打乱顺序
        
#         for nx, ny in neighbors:
#             if not visited[nx][ny]:
#                 # 雕刻通道连接两个节点
#                 carve_passage(lx, ly, nx, ny)
#                 # 递归访问相邻节点
#                 dfs(nx, ny)
    
#     # 选择起始点（逻辑坐标）
#     start_lx, start_ly = 0, 0
    
#     # 开始生成迷宫
#     dfs(start_lx, start_ly)
    
#     # 收集所有墙壁的坐标
#     walls = []
#     for x in range(grid_height):
#         for y in range(grid_width):
#             if maze[x][y]:  # 如果是墙壁
#                 walls.append((y, x))  # 返回(x,y)坐标，其中x是列，y是行
    
#     return walls

def generate_maze_obstacles(grid_width, grid_height):
    """
    使用深度优先算法生成迷宫，路径宽度为3像素，墙壁为1像素
    返回所有墙壁像素点的坐标列表
    
    参数:
    grid_width: 迷宫宽度（奇数）
    grid_height: 迷宫高度（奇数）
    
    返回:
    walls: 所有墙壁像素点坐标的列表 [(x, y), ...]
    """
    
    # 确保宽度和高度都是奇数
    if grid_width % 2 == 0 or grid_height % 2 == 0:
        raise ValueError("Grid width and height must be odd numbers")
    
    # 计算逻辑迷宫的尺寸（每个逻辑节点占用4x4像素：3x3路径 + 1像素边界）
    logic_width = (grid_width + 1) // 4
    logic_height = (grid_height + 1) // 4
    
    # 初始化迷宫，所有位置都是墙壁（True表示墙壁，False表示通路）
    maze = [[True for _ in range(grid_width)] for _ in range(grid_height)]
    
    # 标记逻辑节点访问状态
    visited = [[False for _ in range(logic_width)] for _ in range(logic_height)]
    
    # 四个方向：上、下、左、右
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    def is_valid_logic(lx, ly):
        """检查逻辑坐标是否在范围内"""
        return 0 <= lx < logic_height and 0 <= ly < logic_width
    
    def logic_to_physical(lx, ly):
        """将逻辑坐标转换为物理坐标（左上角）"""
        px = lx * 4 + 1  # 逻辑节点在物理坐标系中的起始位置
        py = ly * 4 + 1
        return px, py
    
    def carve_node(lx, ly):
        """雕刻一个逻辑节点（3x3像素区域）"""
        px, py = logic_to_physical(lx, ly)
        # 雕刻3x3区域
        for dx in range(3):
            for dy in range(3):
                if px + dx < grid_height and py + dy < grid_width:
                    maze[px + dx][py + dy] = False
    
    def carve_passage(lx1, ly1, lx2, ly2):
        """在两个逻辑节点之间雕刻3像素宽的通道"""
        px1, py1 = logic_to_physical(lx1, ly1)
        px2, py2 = logic_to_physical(lx2, ly2)
        
        # 确定通道方向和位置
        if lx1 == lx2:  # 水平通道
            if ly1 < ly2:  # 向右
                # 通道位置在两个节点之间
                passage_x = px1
                passage_y = py1 + 3  # 第一个节点右侧
                # 雕刻3像素宽的垂直通道
                for dx in range(3):
                    if passage_x + dx < grid_height and passage_y < grid_width:
                        maze[passage_x + dx][passage_y] = False
            else:  # 向左
                passage_x = px1
                passage_y = py1 - 1  # 第一个节点左侧
                for dx in range(3):
                    if passage_x + dx < grid_height and passage_y >= 0:
                        maze[passage_x + dx][passage_y] = False
        else:  # 垂直通道
            if lx1 < lx2:  # 向下
                passage_x = px1 + 3  # 第一个节点下方
                passage_y = py1
                # 雕刻3像素宽的水平通道
                for dy in range(3):
                    if passage_x < grid_height and passage_y + dy < grid_width:
                        maze[passage_x][passage_y + dy] = False
            else:  # 向上
                passage_x = px1 - 1  # 第一个节点上方
                passage_y = py1
                for dy in range(3):
                    if passage_x >= 0 and passage_y + dy < grid_width:
                        maze[passage_x][passage_y + dy] = False
    
    def get_unvisited_neighbors(lx, ly):
        """获取未访问的相邻逻辑节点"""
        neighbors = []
        for dx, dy in directions:
            nx, ny = lx + dx, ly + dy
            if is_valid_logic(nx, ny) and not visited[nx][ny]:
                neighbors.append((nx, ny))
        return neighbors
    
    # 递归回溯算法
    def dfs(lx, ly):
        visited[lx][ly] = True
        carve_node(lx, ly)  # 雕刻当前节点
        
        # 随机选择相邻的未访问节点
        neighbors = get_unvisited_neighbors(lx, ly)
        random.shuffle(neighbors)  # 随机打乱顺序
        
        for nx, ny in neighbors:
            if not visited[nx][ny]:
                # 雕刻通道连接两个节点
                carve_passage(lx, ly, nx, ny)
                # 递归访问相邻节点
                dfs(nx, ny)
    
    # 选择起始点（逻辑坐标）
    start_lx, start_ly = 0, 0
    
    # 开始生成迷宫
    dfs(start_lx, start_ly)
    
    # 收集所有墙壁的坐标
    walls = []
    for x in range(grid_height):
        for y in range(grid_width):
            if maze[x][y]:  # 如果是墙壁
                walls.append((y, x))  # 返回(x,y)坐标，其中x是列，y是行
    
    return walls

def generate_indoor_obstacles(grid_width, grid_height):  
    """  
    程序化地生成一个室内环境（房间+走廊）的障碍物列表。   
    """  
    obstacles_set = set()
    for i in range(1, 5):
        for j in range(0,50):
            obstacles_set.add((i * 10, j))
            obstacles_set.add((j, i * 10))
    for i in range(0, 2):
        obstacles_set.remove((10, 4 + i))
        obstacles_set.remove((30, 4 + i))
        obstacles_set.remove((40, 4 + i))
        obstacles_set.remove((10, 14 + i))
        obstacles_set.remove((20, 14 + i))
        obstacles_set.remove((40, 14 + i))
        obstacles_set.remove((10, 24 + i))
        obstacles_set.remove((30, 24 + i))
        obstacles_set.remove((20, 34 + i))
        obstacles_set.remove((40, 34 + i))
        obstacles_set.remove((10, 44 + i))
        obstacles_set.remove((30, 44 + i))
        obstacles_set.remove((40, 44 + i))

        obstacles_set.remove((4 + i, 20))
        obstacles_set.remove((4 + i, 30))
        obstacles_set.remove((4 + i, 40))
        obstacles_set.remove((14 + i, 10))
        obstacles_set.remove((14 + i, 30))
        obstacles_set.remove((14 + i, 40))
        obstacles_set.remove((24 + i, 10))
        obstacles_set.remove((24 + i, 30))
        obstacles_set.remove((24 + i, 40))
        obstacles_set.remove((34 + i, 10))
        obstacles_set.remove((34 + i, 20))
        obstacles_set.remove((34 + i, 30))
        obstacles_set.remove((44 + i, 10))
        obstacles_set.remove((44 + i, 30))
        obstacles_set.remove((44 + i, 40))

    return obstacles_set

