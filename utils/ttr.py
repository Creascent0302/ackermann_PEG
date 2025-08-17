from __future__ import annotations
import numpy as np
import pygame
import math
import heapq
from typing import List, Tuple, Dict
import sys
sys.path.append('.')
from config import ENV_CONFIG

def dijkstra_time_field(occupancy: np.ndarray, sources: List[Tuple[int, int]]) -> np.ndarray:
    """
    Compute the shortest time field using Dijkstra's algorithm.
    Prevent diagonal movement if adjacent cells are obstacles.
    """
    h, w = occupancy.shape
    dist = np.full((h, w), np.inf, dtype=float)
    pq = []

    for r, c in sources:
        if 0 <= r < h and 0 <= c < w and occupancy[r, c] == 0:
            dist[r, c] = 0.0
            heapq.heappush(pq, (0.0, r, c))

    while pq:
        d, r, c = heapq.heappop(pq)
        if d > dist[r, c]:
            continue
        for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and occupancy[nr, nc] == 0:
                # Prevent diagonal movement if adjacent cells are obstacles
                if abs(dr) == 1 and abs(dc) == 1:  # Diagonal movement
                    if occupancy[r + dr, c] == 1 or occupancy[r, c + dc] == 1:
                        continue
                cost = math.hypot(dr, dc)
                nd = d + cost
                if nd < dist[nr, nc]:
                    dist[nr, nc] = nd
                    heapq.heappush(pq, (nd, nr, nc))
    return dist

def compute_ttr(env_image: np.ndarray,
                pursuers_world: list | None = None,
                include_individual: bool = True) -> Dict[str, np.ndarray]:
    """
    计算 TTR (保持追捕者顺序):
    pursuers_world: 若提供 ([(x,y,theta),...]) 则按该顺序生成 pursuer_fields
    """
    occupancy = env_image[2]

    # 有序追捕者栅格列表
    ordered_pursuers_grid = []
    if pursuers_world is not None:
        for p in pursuers_world:
            gx = int(p[0] / ENV_CONFIG['cell_size'])
            gy = int(p[1] / ENV_CONFIG['cell_size'])
            ordered_pursuers_grid.append((gx, gy))
    else:
        # 回退: 旧行为 (顺序可能与 observation 不一致)
        unordered = np.argwhere(env_image[0] > 0)
        ordered_pursuers_grid = [tuple(p) for p in unordered]

    # 逃逸者
    evader_idx = np.argwhere(env_image[1] > 0)[0]

    # 多源最小 TTR 场
    pursuer_field = dijkstra_time_field(occupancy, ordered_pursuers_grid)
    evader_field = dijkstra_time_field(occupancy, [tuple(evader_idx)])

    result: Dict[str, np.ndarray] = {}

    if include_individual:
        per_fields = []
        for pg in ordered_pursuers_grid:
            per_fields.append(dijkstra_time_field(occupancy, [pg]))
        pursuer_fields = np.stack(per_fields, axis=0) if per_fields else np.zeros((0,) + occupancy.shape, dtype=float)
        result["pursuer_fields"] = pursuer_fields
    else:
        pursuer_fields = np.zeros((0,) + occupancy.shape, dtype=float)

    advantage = pursuer_field - evader_field
    result.update({
        "pursuer_field": pursuer_field,
        "pursuer_fields": pursuer_fields,
        "evader_field": evader_field,
        "advantage": advantage
    })
    return result

def env_to_image(observe) -> np.ndarray:
    """
    Generate a random environment with obstacles, pursuers, and an evader.
    """
    env_image = np.zeros((3, ENV_CONFIG['gridnum_width'], ENV_CONFIG['gridnum_height']), dtype=float)
    obstacles = observe['obstacles']
    for r, c in obstacles:
        env_image[2, r, c] = 1.0
    pursuers = observe['pursuers_states']
    for pursuer in pursuers:
        env_image[0, int(pursuer[0]/ENV_CONFIG['cell_size']), int(pursuer[1]/ENV_CONFIG['cell_size'])] = 1.0
    evader = observe['evader_state']
    env_image[1, int(evader[0]/ENV_CONFIG['cell_size']), int(evader[1]/ENV_CONFIG['cell_size'])] = 1.0

    return env_image

def target_filter(advantage: np.ndarray, env_image: np.ndarray, eight_connected: bool = False) -> np.ndarray:
    """
    计算每个自由单元格到障碍的层距离
    返回:
        dist_map: 与 advantage 同形状的整数/浮点距离图
    """
    occupancy = env_image[2]
    h, w = occupancy.shape
    dist_map = np.full((h, w), -1, dtype=float)  # -1 表示尚未赋值

    # 与障碍判定始终使用 8 邻域 (包含对角), 但对角距离按 4 计
    seed_dirs = [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]

    from collections import deque
    q = deque()

    # 1. 找到与障碍相邻(含对角)的自由格:
    #    正交相邻 -> 初始距离 1W
    for r in range(h):
        for c in range(w):
            if occupancy[r, c] == 0 and np.abs(advantage[r, c]) <= 1:
                best_d = None
                # 检查与障碍的相邻关系
                for dr, dc in seed_dirs:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and occupancy[nr, nc] == 1:
                        if abs(dr) == 1 and abs(dc) == 1:
                            d0 = np.sqrt(2)  # 对角相邻
                        else:
                            d0 = 1  # 正交相邻
                        if best_d is None or d0 < best_d:
                            best_d = d0
                # 若不靠障碍但在边界上, 视为距离 1
                if best_d is None and (r == 0 or r == h - 1 or c == 0 or c == w - 1):
                    best_d = 1
                if best_d is not None:
                    dist_map[r, c] = best_d
                    q.append((r, c))
            else:
                if occupancy[r, c] == 1:
                    dist_map[r, c] = 0  # 障碍格距离 0
 
    # 2. BFS 扩散 (始终使用 8 邻域)
    eight_dirs = [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]
    while q:
        r, c = q.popleft()
        base_d = dist_map[r, c]
        for dr, dc in eight_dirs:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and np.abs(advantage[nr, nc]) <= 1 and occupancy[nr, nc] == 0:        
                if  dist_map[nr, nc] >= base_d + np.sqrt(dr**2 + dc**2) or dist_map[nr, nc] == -1:
                    dist_map[nr, nc] = base_d + np.sqrt(dr**2 + dc**2)
                    q.append((nr, nc))
    return dist_map

def sum_distance(r,c,dist_map):
    """
    Sum the distances in the distance map.
    """
    sum_grids = [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1),(0,0)]
    sum = 0
    for grid in sum_grids:
        if 0 <= r + grid[0] < dist_map.shape[0] and 0 <= c + grid[1] < dist_map.shape[1] \
            and dist_map[r + grid[0], c + grid[1]] > 0:
            sum += dist_map[r + grid[0], c + grid[1]]
    return sum

def select_targets(ttr, env_image: np.ndarray) -> List[Tuple[int, int]]:
    """
    Select targets based on the advantage field.
    """
    advantage = ttr['advantage']
    h, w = advantage.shape
    targets = []
    eight_dirs = [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]
    #calculate  the distance from obstacles, and save local maximum as targets
    dist_map = target_filter(advantage, env_image, eight_connected=False)
    checked_list = []
    for r in range(h):
        for c in range(w):
            if dist_map[r, c] > 0 and (r,c) not in checked_list:
                checked_list.append((r, c))
                is_max = True
                cur_sum = sum_distance(r, c, dist_map)
                count = 0
                for dr, dc in eight_dirs:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w:
                        if dist_map[nr, nc] > 0:
                            count += 1
                        if (dist_map[nr, nc] > 0 and sum_distance(nr, nc, dist_map) > cur_sum):
                            is_max = False
                        if (nr,nc) in targets:
                            count = np.inf
                            is_max = False
                            break
                    else:
                        checked_list.append((nr, nc))
                if count <= 1:
                    targets.append((r, c))
                elif is_max:
                    targets.append((r, c))

    return targets,dist_map


def visualize_ttr(env_image: np.ndarray, ttr_result: Dict[str, np.ndarray], targets: List[Tuple[int, int]], scale: int = 20):
    """
    Visualize the TTR fields using pygame.
    """
    h, w = env_image.shape[1:]
    pygame.init()
    screen = pygame.display.set_mode((w * scale, h * scale))
    pygame.display.set_caption("TTR Visualization")

    pursuer_field = ttr_result["pursuer_field"]
    evader_field = ttr_result["evader_field"]
    advantage = ttr_result["advantage"]

    def map_color(value, vmin, vmax, cmap):
        norm_value = (value - vmin) / (vmax - vmin) if vmax > vmin else 0.5
        r, g, b, _ = cmap(norm_value)
        return int(r * 255), int(g * 255), int(b * 255)

    import matplotlib.cm as cm
    pursuer_cmap = cm.get_cmap('viridis')
    evader_cmap = cm.get_cmap('viridis')
    advantage_cmap = cm.get_cmap('coolwarm')

    running = True
    mode = 2  # 0: Pursuer TTR, 1: Evader TTR, 2: Advantage
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                mode = (mode + 1) % 3

        screen.fill((0, 0, 0))
        for r in range(h):
            for c in range(w):
                if mode == 0:
                    value = pursuer_field[r, c]
                    color = map_color(value, 0, 45, pursuer_cmap) if np.isfinite(value) else (0, 0, 0)
                elif mode == 1:
                    value = evader_field[r, c]
                    color = map_color(value, 0, 45, evader_cmap) if np.isfinite(value) else (0, 0, 0)
                elif mode == 2:
                    value = advantage[r, c]
                    color = map_color(value, -45, 45, advantage_cmap) if np.isfinite(value) else (0, 0, 0)
                pygame.draw.rect(screen, color, (c * scale, r * scale, scale, scale))

        for r, c in zip(*np.where(env_image[0] > 0)):
            pygame.draw.rect(screen, (255, 0, 0), (c * scale, r * scale, scale, scale))
        for r, c in zip(*np.where(env_image[1] > 0)):
            pygame.draw.rect(screen, (0, 0, 255), (c * scale, r * scale, scale, scale))
        for target in targets:
            r, c = target
            pygame.draw.circle(screen, (0, 255, 0), (r * scale + scale // 2, c * scale + scale // 2), scale // 4)
        #draw obstacles
        for r, c in zip(*np.where(env_image[2] > 0)):
            pygame.draw.rect(screen, (128, 128, 128), (c * scale, r * scale, scale, scale))
 
        pygame.display.flip()

    pygame.quit()
