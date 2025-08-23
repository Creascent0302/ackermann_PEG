import sys, pygame, numpy as np
sys.path.append('.')
from config import ENV_CONFIG
import heapq
# ===== 新增 控制相关导入 =====
from utils.pure_pursuit import pure_pursuit_control
from utils.dwa import dwa_control


def _dijkstra(start, adj):
    dist = {start: 0.0}
    h = [(0.0, start)]
    while h:
        d, u = heapq.heappop(h)
        if d > dist[u] + 1e-9:
            continue
        for v, w in adj.get(u, []):
            nd = d + w
            if nd + 1e-9 < dist.get(v, float('inf')):
                dist[v] = nd
                heapq.heappush(h, (nd, v))
    return dist

def _nearest_medial(point, medial_nodes):
    return min(medial_nodes, key=lambda n: np.linalg.norm(np.array(point) - np.array(n)))


# ===== 调整: 仅在中轴子图上构建邻接 =====
def _build_medial_adj_all(all_nodes, medial_edges):
    adj = {n: [] for n in all_nodes}
    for a, b in medial_edges:
        if a in adj and b in adj:
            w = np.linalg.norm(np.array(a) - np.array(b))
            adj[a].append((b, w))
            adj[b].append((a, w))
    return adj

# ===== 组合优化版本目标选择 =====
def select_medial_targets(pursuers_states,
                          evader_state,
                          medial_axis_nodes,
                          medial_axis_all_nodes,
                          medial_axis_edges,
                          neighbor_radius,
                          max_candidates_per_pursuer=6):
    """
    全投影 + 组合优化 (扩展规则):
      - 若追捕者投影节点 == 逃逸者投影节点 -> 直接目标=逃逸者，忽略其在优势计算中作用
      - 新增: 若追捕者与逃逸者原始欧氏距离 < 1.5 -> 同上直接锁定
    """
    if (not medial_axis_all_nodes) or (not medial_axis_edges):
        return [(None, None, 0, [], False) for _ in pursuers_states]

    all_nodes_set = set(medial_axis_all_nodes)
    adj = _build_medial_adj_all(list(all_nodes_set), medial_axis_edges)
    ev_pt = (evader_state[0], evader_state[1])
    e_proj = _nearest_medial(ev_pt, all_nodes_set)
    dist_evader = _dijkstra(e_proj, adj)

    dist_cache = {}
    def get_dist(node):
        if node not in dist_cache:
            dist_cache[node] = _dijkstra(node, adj)
        return dist_cache[node]

    # ---- 投影 & 固定条件（投影同节点 或 欧氏距离 < 1.5）----
    pursuer_proj_nodes = []
    pursuer_proj_dist = []
    fixed_evader_idx = set()
    for idx, p in enumerate(pursuers_states):
        p_pos = (p[0], p[1])
        p_proj = _nearest_medial(p_pos, all_nodes_set)
        pursuer_proj_nodes.append(p_proj)
        dist_p = get_dist(p_proj)
        pursuer_proj_dist.append(dist_p)
        # 原条件: 投影一致
        if p_proj == e_proj:
            fixed_evader_idx.add(idx)
            continue
        # 新增条件: 原始欧氏距离 < 1.5
        if np.linalg.norm(np.array(p_pos) - np.array(ev_pt)) < 1.5:
            fixed_evader_idx.add(idx)

    # 若全部追捕者都固定为逃逸者目标
    if len(fixed_evader_idx) == len(pursuers_states):
        return [(ev_pt[0], ev_pt[1], 0, [], True) for _ in pursuers_states]

    # ---- 为未固定追捕者生成候选 ----
    pursuer_candidates = []  # 顺序仅包含未固定追捕者
    active_original_indices = []  # 映射组合搜索内部索引 -> 原始追捕者索引

    for idx, (p_proj, dist_p) in enumerate(zip(pursuer_proj_nodes, pursuer_proj_dist)):
        if idx in fixed_evader_idx:
            continue
        # 邻域候选
        cand = [n for n in all_nodes_set if dist_p.get(n, float('inf')) < neighbor_radius]
        if not cand:
            cand = [min(all_nodes_set, key=lambda n: dist_p.get(n, float('inf')))]
        entries = []
        for c in cand:
            dc = get_dist(c)
            est = sum(1 for x, de in dist_evader.items() if dc.get(x, float('inf')) < de)
            entries.append((c, dc, est))
        entries.sort(key=lambda x: (-x[2], dist_p.get(x[0], float('inf'))))
        if len(entries) > max_candidates_per_pursuer:
            entries = entries[:max_candidates_per_pursuer]
        pursuer_candidates.append(entries)
        active_original_indices.append(idx)

    # ---- 回溯组合（仅针对未固定追捕者）----
    nodes_list = list(dist_evader.keys())
    inf = float('inf')
    base_best = {n: inf for n in nodes_list}
    best_group_adv = -1
    best_choice = None  # list[(node, dist_map)] 对应 pursuer_candidates 顺序

    if pursuer_candidates:  # 防止全部固定导致空组合
        def backtrack(ci, cur_choice, cur_min):
            nonlocal best_group_adv, best_choice
            if ci == len(pursuer_candidates):
                covered = sum(1 for n, de in dist_evader.items() if cur_min[n] < de)
                if covered > best_group_adv:
                    best_group_adv = covered
                    best_choice = list(cur_choice)
                return
            covered_now = sum(1 for n, de in dist_evader.items() if cur_min[n] < de)
            potential_max = covered_now + sum(1 for n, de in dist_evader.items() if cur_min[n] >= de)
            if potential_max <= best_group_adv:
                return
            for node_c, dist_c, _ in pursuer_candidates[ci]:
                new_min = cur_min if ci == len(pursuer_candidates)-1 else cur_min.copy()
                for n, dv in dist_c.items():
                    if dv < new_min[n]:
                        new_min[n] = dv
                cur_choice.append((node_c, dist_c))
                backtrack(ci+1, cur_choice, new_min)
                cur_choice.pop()

        backtrack(0, [], base_best)

    # 若没有非固定追捕者，best_choice 允许为空
    if best_choice is None:
        best_choice = []

    # ---- 组装 final_min (仅由非固定目标贡献) ----
    final_min = {}
    for node_c, dist_c in best_choice:
        for n, dv in dist_c.items():
            if dv < final_min.get(n, float('inf')):
                final_min[n] = dv

    # ---- 按原顺序构造结果 ----
    results = [None] * len(pursuers_states)
    # 固定者直接指向逃逸者
    for idx in fixed_evader_idx:
        results[idx] = (ev_pt[0], ev_pt[1], 0, [], True)

    # 非固定者填充
    for (orig_idx, (node_c, dist_c)) in zip(active_original_indices, best_choice):
        dist_p_map = pursuer_proj_dist[orig_idx]
        contrib_nodes = [n for n, dv in dist_c.items()
                         if abs(dv - final_min.get(n, inf)) < 1e-9 and dv < dist_evader.get(n, inf)]
        contrib_count = len(contrib_nodes)
        dist_from_proj = dist_p_map.get(node_c, float('inf'))
        euclid_ev = np.linalg.norm(np.array(node_c) - np.array(ev_pt))
        if dist_from_proj < 1.0 or euclid_ev < 0.5:
            results[orig_idx] = (ev_pt[0], ev_pt[1], 0, [], True)
        else:
            results[orig_idx] = (node_c[0], node_c[1], contrib_count, contrib_nodes, False)

    # 保险: 若有未填（极端情况），置空
    for i, r in enumerate(results):
        if r is None:
            results[i] = (ev_pt[0], ev_pt[1], 0, [], True)

    return results

# ========== 新增: 渲染 PRM + Agents ==========
def render_prm_and_agents(pursuers_states,
                          evader_state,
                          obstacles,
                          nodes,
                          edges,
                          medial_axis_nodes=None,
                          medial_axis_edges=None,
                          targets=None,
                          paths=None,              # 新增: 规划路径
                          window_scale=100,
                          bg_color=(255,255,255)):
    """
    渲染 PRM:
      - 障碍物 / 普通边 / 中轴边 / 节点
      - 追捕者 / 逃逸者
      - 目标 (含优势标记)
      - 规划路径 (paths[i] = [node0, node1, ...])
    """
    if not pygame.get_init():
        pygame.init()
    medial_axis_nodes = medial_axis_nodes or set()
    medial_axis_edges = medial_axis_edges or set()
    gw, gh = ENV_CONFIG['gridnum_width'], ENV_CONFIG['gridnum_height']
    cs = ENV_CONFIG['cell_size']
    W = int(gw * cs * window_scale); H = int(gh * cs * window_scale)
    if not pygame.get_init(): pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("PRM + Agents (Advantage Targets)")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 16)
    def to_px(x,y): return int(x*window_scale), int(y*window_scale)
    medial_set = set(medial_axis_nodes)
    running = True
    color_palette = [
        (220,0,255),  # P0 优势节点色
        (255,140,0),  # P1
        (0,180,255),  # P2
        (255,0,120),  # P3
        (0,200,140)   # P4
    ]
    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                running = False
        screen.fill(bg_color)
        # 障碍
        for ox, oy in obstacles:
            pygame.draw.rect(screen,(0,0,0),
                             pygame.Rect(int(ox*cs*window_scale),
                                         int(oy*cs*window_scale),
                                         int(cs*window_scale),
                                         int(cs*window_scale)))
        # 普通边
        for a,b in edges:
            if a in medial_set and b in medial_set: continue
            pygame.draw.line(screen,(170,180,230),to_px(a[0],a[1]),to_px(b[0],b[1]),1)
        # 中轴边
        for a,b in medial_axis_edges:
            pygame.draw.line(screen,(0,150,0),to_px(a[0],a[1]),to_px(b[0],b[1]),3)
        # 节点
        for n in nodes:
            col = (0,170,0) if n in medial_set else (200,0,0)
            pygame.draw.circle(screen,col,to_px(n[0],n[1]),4 if n not in medial_set else 5)
        # === 新增: 绘制路径 (在节点 / 追捕者之前或之后均可，这里放在追捕者前) ===
        if paths:
            for i, path in paths.items():
                if not path or len(path) < 2:
                    continue
                col = color_palette[i % len(color_palette)]
                # 折线
                for a, b in zip(path[:-1], path[1:]):
                    pygame.draw.line(screen,
                                     (*col[:3],),  # 颜色
                                     to_px(a[0], a[1]),
                                     to_px(b[0], b[1]),
                                     2)
                # 节点点缀
                for n in path:
                    pygame.draw.circle(screen, col, to_px(n[0], n[1]), 3)
        # 追捕者
        for i,p in enumerate(pursuers_states):
            pygame.draw.circle(screen,(0,0,255),to_px(p[0],p[1]),6)
            screen.blit(font.render(f"P{i}",True,(0,0,255)),(p[0]*window_scale+6,p[1]*window_scale+4))
        # 逃逸者
        pygame.draw.circle(screen,(255,128,0),to_px(evader_state[0],evader_state[1]),7)
        screen.blit(font.render("E",True,(255,128,0)),(evader_state[0]*window_scale+6, evader_state[1]*window_scale+4))
        # 目标
        if targets:
            for i, tgt in enumerate(targets):
                tx, ty, adv = tgt[0], tgt[1], tgt[2]
                adv_nodes = tgt[3] if len(tgt) >= 4 else []
                is_evader_target = (len(tgt) >= 5 and tgt[4])
                if is_evader_target:
                    # 直接目标为逃逸者：用实心紫色强调
                    pygame.draw.circle(screen, (170,0,200), to_px(tx,ty), 10, 0)
                    pygame.draw.circle(screen, (255,255,255), to_px(tx,ty), 4, 0)
                    screen.blit(font.render(f"A{adv}", True, (170,0,200)),
                                (tx*window_scale+6, ty*window_scale+2))
                    px,py,_ = pursuers_states[i]
                    #line
                    pygame.draw.line(screen, (140,0,200), to_px(px,py), to_px(evader_state[0],evader_state[1]), 2)
                else:
                    tcol = (128,0,255)
                    pygame.draw.circle(screen, tcol, to_px(tx,ty), 8, 2)
                    screen.blit(font.render(f"A{adv}",True,tcol),
                                (tx*window_scale+6,ty*window_scale+2))
                    # 连线
                    px,py,_ = pursuers_states[i]
                    pygame.draw.line(screen,(140,0,200),to_px(px,py),to_px(tx,ty),2)
                # 优势节点
                ac = (200,0,200)
                for n in adv_nodes:
                    pygame.draw.circle(screen, ac, to_px(n[0], n[1]), 5)
        pygame.display.flip()
        clock.tick(30)
    pygame.quit()

# ========== 新增: 中轴最短路径与动作生成 =====
def _build_medial_adj_weighted(nodes_set, medial_axis_edges):
    adj = {n: [] for n in nodes_set}
    for a, b in medial_axis_edges:
        if a in adj and b in adj:
            w = np.linalg.norm(np.array(a) - np.array(b))
            adj[a].append((b, w))
            adj[b].append((a, w))
    return adj

def _shortest_path_on_medial(adj, start, goal):
    if start not in adj or goal not in adj:
        return []
    if start == goal:
        return [start]
    h = []
    heapq.heappush(h, (0.0, start))
    dist = {start: 0.0}
    parent = {}
    while h:
        d,u = heapq.heappop(h)
        if u == goal:
            break
        if d > dist[u] + 1e-9:
            continue
        for v,w in adj[u]:
            nd = d + w
            if nd + 1e-9 < dist.get(v, float('inf')):
                dist[v] = nd
                parent[v] = u
                heapq.heappush(h,(nd,v))
    if goal not in parent and goal != start:
        return []
    # 回溯
    path = [goal]
    cur = goal
    while cur != start:
        cur = parent.get(cur)
        if cur is None:
            return []
        path.append(cur)
    path.reverse()
    return path

def prm_medial_policy(pursuers_states,
                      evader_state,
                      targets,
                      medial_axis_all_nodes,
                      medial_axis_edges,
                      obstacles):
    """
    基于已选 targets 在中轴上生成路径与 (steering, speed):
      - 起点: 追捕者投影节点
      - 终点: 若 is_evader_target -> 逃逸者投影节点，否则目标节点
      - 控制: 取路径第二个节点(或第一个)作为 pure pursuit 预瞄点
    返回: actions, paths
    """
    if not medial_axis_all_nodes or not medial_axis_edges:
        return [(0.0,0.0)]*len(pursuers_states), {}
    nodes_set = set(medial_axis_all_nodes)
    adj = _build_medial_adj_weighted(nodes_set, medial_axis_edges)
    ev_pt = (evader_state[0], evader_state[1])
    e_proj = _nearest_medial(ev_pt, nodes_set)
    actions = []
    paths = {}
    for i, p in enumerate(pursuers_states):
        p_pos = (p[0], p[1])
        p_proj = _nearest_medial(p_pos, nodes_set)
        tx, ty, _, _, is_evader = targets[i]
        goal_node = e_proj if is_evader else _nearest_medial((tx,ty), nodes_set)
        path = _shortest_path_on_medial(adj, p_proj, goal_node)
        paths[i] = path
        # 预瞄点
        if path:
            if len(path) > 1:
                preview = path[1]
            else:
                preview = path[0]
        else:
            preview = (p_pos[0], p_pos[1])
        # 调用控制（状态包含朝向的第三个值, 若缺省已在示例为 0.0）
        steering, speed = pure_pursuit_control(p, preview)
        steering = dwa_control(p, (steering, speed), obstacles, preview)
        actions.append((steering, speed))
    return actions, paths

# ========== 演示 (修改: 计算并显示优势目标) ==========
if __name__ == "__main__":
    from generator import PRMGenerator
    import time 
    # 示例：三名追捕者 + 一个逃逸者
    grid_width = ENV_CONFIG['gridnum_width']
    grid_height = ENV_CONFIG['gridnum_height']
    cell_size = ENV_CONFIG['cell_size']
    total_cells = grid_width * grid_height
    num_obstacles = int(total_cells * 0.2)  # 20% 障碍物
    obstacles = set()
    while len(obstacles) < num_obstacles:
        x = np.random.randint(0, grid_width)
        y = np.random.randint(0, grid_height)
        obstacles.add((x, y))
    obstacles = list(obstacles)



    # 生成 PRM
    prm_generator = PRMGenerator(grid_width, grid_height, obstacles,
                                 num_nodes=320, connection_radius=0.6)
    start_time = time.time()
    (nodes,
     edges,
     medial_axis_nodes,
     medial_axis_all_nodes,
     medial_axis_edges,
     medial_axis_paths) = prm_generator.generate_prm()
    end_time = time.time()
    print(f"PRM generation took {end_time - start_time:.4f} seconds")
    for i in range (5):
        # 追捕者 (x,y 在连续坐标系, 不放置在障碍格中心正中只是示例)
        pursuers = []
        while len(pursuers) < 3:
            x = np.random.uniform(0, grid_width * cell_size)
            y = np.random.uniform(0, grid_height * cell_size)
            if (int(x / cell_size), int(y / cell_size)) not in obstacles:
                pursuers.append((x, y, 0.0))
        evader_state = (grid_width * cell_size * 0.8,
                        grid_height * cell_size * 0.7,
                        0.0)
        neighbor_radius = 2  # 可调

        start_time = time.time()
        targets = select_medial_targets(pursuers, evader_state,
                                        medial_axis_nodes,
                                        medial_axis_all_nodes,
                                        medial_axis_edges,
                                        neighbor_radius)
        end_time = time.time()
        print(f"Target selection took {end_time - start_time:.4f} seconds")
        # 新增: 规划动作
        act_start = time.time()
        actions, paths = prm_medial_policy(pursuers,
                                           evader_state,
                                           targets,
                                           medial_axis_all_nodes,
                                           medial_axis_edges,
                                           obstacles)
        act_end = time.time()
        print("Actions:", actions, f"(planning {act_end - act_start:.4f}s)")
        # 可视化
        render_prm_and_agents(pursuers, evader_state, obstacles,
                              nodes, edges,
                              medial_axis_nodes, medial_axis_edges,
                              targets=targets,
                              paths=paths,          # 传入路径
                              window_scale=100)
