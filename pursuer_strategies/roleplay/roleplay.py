from __future__ import annotations
import sys
import time
from typing import List, Tuple, Dict
import numpy as np
sys.path.append('.')  # ensure relative imports work
from config import ENV_CONFIG
from utils.ttr import compute_ttr, select_targets, env_to_image
from utils.astar_pp_dwa import pursuer_track_target
from pursuer_strategies.roleplay.pursuer_roles import direct_pursuer, encircle_pursuer
last_pursuit_decision_time = 0.0
last_match_list: List[Tuple[float, float]] = []
last_evader_observe: List[float] = []

def commander_roles(observation):
    """
    Commander decides roles and returns a role list (one role per pursuer).
    Roles:
      - 'direct'   : pursuer that chases the evader directly (closest to evader)
      - 'encircle' : pursuers that will try to take/hold encircling targets
    This function is centralized but only assigns roles (no target assignment).
    """
    global last_evader_observe
    pursuers = observation['pursuers_states']
    evader = observation['evader_state']

    N = len(pursuers)
    if N == 0:
        return []

    # choose the direct chaser (closest to evader in Euclidean distance)
    pursuer_positions = np.array([p[:2] for p in pursuers])
    ev_pos = np.array(evader[:2])
    dists = np.linalg.norm(pursuer_positions - ev_pos, axis=1)
    direct_idx = int(np.argmin(dists))

    roles = ['encircle'] * N
    roles[direct_idx] = 'direct'

    last_evader_observe = evader[:2]
    return roles


def assign_targets_decentralized(observation, targets, ttr_result, role_list, use_ttr_cost: bool = True):
        """
        Decentralized assignment that uses:
          - ttr_result (pursuer TTR maps)
          - role_list (from commander_roles)
          - targets (world coords list)
          - observation (for evader motion prediction)
        Returns match_list: a list of (x,y) targets (or evader position) for each pursuer.
        Encircle pursuers pick targets independently (no global uniqueness enforced).
        The direct pursuer is assigned the evader or a short prediction of it.
        """
        global last_evader_observe, last_match_list
        pursuer_ttr = ttr_result.get('pursuer_fields', None)  # expected shape (N, H, W)
        evader = observation['evader_state']
        pursuers = observation['pursuers_states']
        cell_size = ENV_CONFIG['cell_size']

        N = len(pursuers)
        # fallback: everyone chases the evader
        if pursuer_ttr is None or pursuer_ttr.shape[0] != N:
            last_evader_observe = evader[:2]
            last_match_list = [tuple(evader[:2]) for _ in pursuers]
            return last_match_list

        if not targets:
            last_evader_observe = evader[:2]
            last_match_list = [tuple(evader[:2]) for _ in pursuers]
            return last_match_list

        # grid conversion and prediction setup
        target_grids = [(int(tx / cell_size), int(ty / cell_size)) for (tx, ty) in targets]
        ev_pos = np.array(evader[:2])

        match_list: List[Tuple[float, float]] = [tuple(evader[:2]) for _ in range(N)]

        # Each pursuer decides based on its role using helper functions
        for p_idx in range(N):
            role = role_list[p_idx] if p_idx < len(role_list) else 'encircle'
            if role == 'direct':
                match_list[p_idx] = direct_pursuer(evader, last_evader_observe, match_list,role_list,prediction_steps=2)
                continue

            # encircle: pick best target independently using encircle_pursuer_target
            match_list[p_idx] = encircle_pursuer(
                p_idx,
                pursuers[p_idx],
                targets,
                target_grids,
                pursuer_ttr,
                ev_pos,
                last_evader_observe,
                match_list,
                use_ttr_cost=use_ttr_cost,
                role_list=role_list,
                prediction_steps=10,
            )

        last_evader_observe = evader[:2]
        last_match_list = match_list
        return match_list

def time_criterion():
    global last_pursuit_decision_time
    if time.time() - last_pursuit_decision_time > 0.3:
        last_pursuit_decision_time = time.time()
        return True
    return False

# Update roleplay_pursuer_policy to use commander_roles + assign_targets_decentralized
def roleplay_pursuer_policy(observation):
    """
    Use TTR fields + pure pursuit to produce actions for all pursuers.
    Now uses commander_roles (returns role_list) and a decentralized assignment
    function that assigns targets according to ttrmap, role_list, targets, and observation.
    """
    global last_match_list
    obstacles = observation['obstacles']
    pursuer_actions = []
    path = None

    if time_criterion():
        env_image = env_to_image(observation)
        ttr_result = compute_ttr(env_image, pursuers_world=observation['pursuers_states'])
        targets, dist_map = select_targets(ttr_result, env_image)

        # convert grid targets to world-center coordinates
        cell = ENV_CONFIG['cell_size']
        targets = [((tx * cell + cell / 2), (ty * cell + cell / 2)) for tx, ty in targets]

        # commander decides roles
        role_list = commander_roles(observation)

        # decentralized assignment driven by TTR maps + roles
        match_list = assign_targets_decentralized(observation, targets, ttr_result, role_list)
        last_match_list = match_list
    else:
        match_list = last_match_list

    for i, pursuer in enumerate(observation['pursuers_states']):
        target = match_list[i] if match_list else None
        if target is not None:
            (steering_angle, speed), path, _ = pursuer_track_target(target, pursuer, obstacles, pursuer_idx=i)
        else:
            steering_angle, speed = 0.0, 0.0
        pursuer_actions.append((steering_angle, speed))

    return pursuer_actions

def visualize_roleplay(env_image: np.ndarray,
                       ttr_result: Dict[str, np.ndarray],
                       targets: List[Tuple[float, float]],
                       match_list: List[Tuple[float, float]],
                       role_list: List[str],
                       observe: Dict,
                       scale: int = 20):
    """
    简单 pygame 可视化：显示 TTR 优势场、障碍、targets、分配(match_list) 与角色(role_list)。
    - env_image: from env_to_image(observe) (channels, H, W)
    - ttr_result: result from compute_ttr(...)
    - targets: world coords list
    - match_list: assigned targets per pursuer (world coords)
    - role_list: list of roles per pursuer ('direct'|'encircle')
    - observe: environment observation dict (contains pursuers_states, evader_state, obstacles)
    """
    import pygame
    pygame.init()
    font = pygame.font.Font(None, 20)

    _, H, W = env_image.shape
    screen = pygame.display.set_mode((H * scale, W * scale))
    pygame.display.set_caption("Roleplay TTR Visualization")
    clock = pygame.time.Clock()
    running = True

    adv = ttr_result.get('advantage', None)
    pursuer_fields = ttr_result.get('pursuer_fields', None)

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.fill((255, 255, 255))

        # Draw advantage / TTR background (coarse)
        if adv is not None:
            for r in range(H):
                for c in range(W):
                    v = adv[r, c]
                    if v < -1:
                        color = (0, 155, 0)        # pursuer advantage
                    elif v > 1:
                        color = (155, 0, 0)        # evader advantage
                    else:
                        color = (240, 240, 120)    # neutral
                    pygame.draw.rect(screen, color, (r * scale, c * scale, scale, scale))

        # Draw obstacles / channels from env_image channels (safe checks)
        if env_image.shape[0] >= 3:
            # channel 2 -> obstacles (gray)
            for r, c in zip(*np.where(env_image[2] > 0)):
                pygame.draw.rect(screen, (120, 120, 120), (r * scale, c * scale, scale, scale))
            # channel 0/1 used in other code: draw them if present
            for r, c in zip(*np.where(env_image[0] > 0)):
                pygame.draw.rect(screen, (255, 0, 0), (r * scale, c * scale, scale, scale))
            for r, c in zip(*np.where(env_image[1] > 0)):
                pygame.draw.rect(screen, (0, 0, 255), (r * scale, c * scale, scale, scale))

        # Draw all targets (world coords -> grid centers)
        for tx, ty in targets:
            gx = int(tx / ENV_CONFIG['cell_size'])
            gy = int(ty / ENV_CONFIG['cell_size'])
            pygame.draw.circle(screen, (0, 0, 200), (gx * scale + scale // 2, gy * scale + scale // 2), max(2, scale // 6))

        # Draw pursuers and their assigned targets/arrows + role labels
        pursuers = observe.get('pursuers_states', [])
        for idx, purs in enumerate(pursuers):
            px_w, py_w = purs[0], purs[1]
            px = int(px_w / ENV_CONFIG['cell_size'])
            py = int(py_w / ENV_CONFIG['cell_size'])
            color = (0, 100, 255)

            # draw index
            txt = font.render(str(idx), True, (255, 255, 255))
            screen.blit(txt, (px * scale + 1, py * scale + 1))

            # draw assigned target arrow
            if match_list and idx < len(match_list) and match_list[idx] is not None:
                tx_w, ty_w = match_list[idx]
                txg = int(tx_w / ENV_CONFIG['cell_size'])
                tyg = int(ty_w / ENV_CONFIG['cell_size'])
                pygame.draw.circle(screen, (0, 255, 255), (txg * scale + scale // 2, tyg * scale + scale // 2), max(3, scale // 4), 2)
                pygame.draw.line(screen, (0, 0, 0), (px * scale + scale // 2, py * scale + scale // 2),
                                 (txg * scale + scale // 2, tyg * scale + scale // 2), 2)

            # draw role label
            role = role_list[idx] if role_list and idx < len(role_list) else 'encircle'
            role_color = (255, 200, 0) if role == 'direct' else (0, 0, 100)
            rtxt = font.render(role[0].upper(), True, role_color)
            screen.blit(rtxt, (px * scale + 6, py * scale - 6))
        # Draw evader
        ev = observe.get('evader_state', None)
        if ev is not None:
            ex_w, ey_w = ev[0], ev[1]
            ex = int(ex_w / ENV_CONFIG['cell_size'])
            ey = int(ey_w / ENV_CONFIG['cell_size'])
            pygame.draw.circle(screen, (200, 0, 200), (ex * scale + scale // 2, ey * scale + scale // 2), max(5, scale // 2))
            etxt = font.render("E", True, (255, 255, 255))
            screen.blit(etxt, (ex * scale + 2, ey * scale + 2))

        pygame.display.flip()
        clock.tick(12)

    pygame.quit()


if __name__ == "__main__":
    # 小 demo：创建 env、计算 TTR、分配、并可视化
    import sys
    sys.path.append('.')
    from env import GameENV

    seed = 17
    np.random.seed(seed)
    env = GameENV(num_pursuers=3, seed=seed)
    observe = env.reset()

    env_image = env_to_image(observe)
    ttr_result = compute_ttr(env_image, pursuers_world=observe['pursuers_states'])
    targets, dist_map = select_targets(ttr_result, env_image)

    # world-center targets
    cell = ENV_CONFIG['cell_size']
    targets = [((tx * cell + cell / 2), (ty * cell + cell / 2)) for tx, ty in targets]

    # 角色与分配（使用已有逻辑）
    role_list = commander_roles(observe)
    match_list = assign_targets_decentralized(observe, targets, ttr_result, role_list)

    print("Roles:", role_list)
    print("Assigned targets:", match_list)

    visualize_roleplay(env_image, ttr_result, targets, match_list, role_list, observe)
