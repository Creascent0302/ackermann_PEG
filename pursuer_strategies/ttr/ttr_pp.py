from __future__ import annotations
import sys
import time
from typing import List, Tuple, Dict
import numpy as np

sys.path.append('.')  # ensure relative imports work

from config import ENV_CONFIG
from utils.ttr import compute_ttr, select_targets, env_to_image
from utils.astar_pp_dwa import pursuer_track_target

last_pursuit_decision_time = 0.0
last_match_list: List[Tuple[float, float]] = []
last_evader_observe: List[float] = []

def match_target(observation, targets, ttr_result, use_ttr_cost: bool = True):
    """
    Simple unique assignment:
      1) The pursuer with smallest TTR to the evader goes directly to the evader
      2) Other pursuers pick the best remaining target (prefer TTR cost, else inf)
      3) If targets run out -> chase evader
    Keeps assignments unique.
    """
    global last_evader_observe, last_match_list
    pursuer_ttr = ttr_result['pursuer_fields']  # shape (N, H, W)
    evader = observation['evader_state']
    pursuers = observation['pursuers_states']
    cell_size = ENV_CONFIG['cell_size']

    # evader motion estimate (world units)
    evader_motion = (np.array(evader[:2]) - np.array(last_evader_observe)) if last_evader_observe else np.zeros(2)

    N = len(pursuers)
    if pursuer_ttr.shape[0] != N:
        return [tuple(evader[:2]) for _ in pursuers]

    # clamp evader grid indices
    er = max(0, min(int(evader[0] / cell_size), pursuer_ttr.shape[1] - 1))
    ec = max(0, min(int(evader[1] / cell_size), pursuer_ttr.shape[2] - 1))

    nearest_idx = int(np.argmin(pursuer_ttr[:, er, ec]))

    if not targets:
        last_evader_observe = evader[:2]
        return [tuple(evader[:2]) for _ in pursuers]

    target_grids = [(int(tx / cell_size), int(ty / cell_size)) for (tx, ty) in targets]
    H, W = pursuer_ttr.shape[1], pursuer_ttr.shape[2]

    def cost(p_idx: int, tgt_world, tgt_grid):
        gx, gy = tgt_grid
        if use_ttr_cost and 0 <= gx < H and 0 <= gy < W:
            v = pursuer_ttr[p_idx, gx, gy]
            if np.isfinite(v):
                predicted_evader_pos = np.array(evader[:2]) + 2 * evader_motion
                prediction_cost = np.linalg.norm(predicted_evader_pos - np.array(tgt_world))
                return float(v) + float(prediction_cost)
        return float('inf')

    match_list = [None] * N
    match_list[nearest_idx] = tuple(evader[:2])
    used = set()

    for p_idx in range(N):
        if p_idx == nearest_idx:
            continue
        best_tid = None
        best_c = float('inf')
        for tid, (tw, tg) in enumerate(zip(targets, target_grids)):
            if tid in used:
                continue
            c = cost(p_idx, tw, tg)
            if c < best_c:
                best_c = c
                best_tid = tid
        if best_tid is not None and best_c < float('inf'):
            match_list[p_idx] = targets[best_tid]
            used.add(best_tid)
        else:
            match_list[p_idx] = tuple(evader[:2])

    last_evader_observe = evader[:2]
    last_match_list = match_list
    return match_list

def time_criterion():
    global last_pursuit_decision_time
    if time.time() - last_pursuit_decision_time > 0.3:
        last_pursuit_decision_time = time.time()
        return True
    return False

def ttr_pursuer_policy(observation):
    """
    Use TTR fields + pure pursuit to produce actions for all pursuers.

    Returns:
        List of (steering_angle, speed) for each pursuer.
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

        match_list = match_target(observation, targets, ttr_result)
        last_match_list = match_list
    else:
        match_list = last_match_list

    # default path None to avoid unbound variable
    for i, pursuer in enumerate(observation['pursuers_states']):
        target = match_list[i] if match_list else None
        if target is not None:
            (steering_angle, speed), path, _ = pursuer_track_target(target, pursuer, obstacles, pursuer_idx=i)
        else:
            steering_angle, speed = 0.0, 0.0
        pursuer_actions.append((steering_angle, speed))

    return pursuer_actions

def visualize_ttr(env_image: np.ndarray, ttr_result: Dict[str, np.ndarray],
                  targets: List[Tuple[int, int]], match_list: List[Tuple[int, int]],
                  dist_map, scale: int = 20):
    """
    Simple pygame visualization of TTR and assignments.
    """
    import pygame
    pygame.init()
    font = pygame.font.Font(None, 26)
    _, H, W = env_image.shape
    screen = pygame.display.set_mode((H * scale, W * scale))
    pygame.display.set_caption("TTR Visualization")
    clock = pygame.time.Clock()
    running = True

    # optional global observe for drawing pursuers if available
    observe = globals().get('observe')

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.fill((255, 255, 255))

        # Draw TTR advantage field
        adv = ttr_result.get('advantage')
        if adv is not None:
            for r in range(H):
                for c in range(W):
                    v = adv[r, c]
                    if v < -1:
                        color = (0, 155, 0)
                    elif v > 1:
                        color = (155, 0, 0)
                    else:
                        color = (255, 255, 0)
                    pygame.draw.rect(screen, color, (r * scale, c * scale, scale, scale))

        # Obstacles and channels (safe checks)
        if env_image.shape[0] >= 3:
            for r, c in zip(*np.where(env_image[2] > 0)):
                pygame.draw.rect(screen, (128, 128, 128), (r * scale, c * scale, scale, scale))
            for r, c in zip(*np.where(env_image[0] > 0)):
                pygame.draw.rect(screen, (255, 0, 0), (r * scale, c * scale, scale, scale))
            for r, c in zip(*np.where(env_image[1] > 0)):
                pygame.draw.rect(screen, (0, 0, 255), (r * scale, c * scale, scale, scale))

        # targets (grid coords)
        for tx, ty in targets:
            gx = int(tx / ENV_CONFIG['cell_size'])
            gy = int(ty / ENV_CONFIG['cell_size'])
            pygame.draw.circle(screen, (0, 0, 255), (gx * scale + scale // 2, gy * scale + scale // 2), scale // 8)

        # matched targets and arrows to pursuers if observation available
        for idx, target in enumerate(match_list or []):
            if target is None:
                continue
            tx, ty = target
            gx = int(tx / ENV_CONFIG['cell_size'])
            gy = int(ty / ENV_CONFIG['cell_size'])
            pygame.draw.circle(screen, (0, 255, 255), (gx * scale + scale // 2, gy * scale + scale // 2), scale // 4, 2)

            if observe:
                purs = observe['pursuers_states'][idx]
                px = int(purs[0] / ENV_CONFIG['cell_size'] * scale)
                py = int(purs[1] / ENV_CONFIG['cell_size'] * scale)
                pygame.draw.line(screen, (0, 0, 255), (px, py), (gx * scale + scale // 2, gy * scale + scale // 2), 2)
                txt = font.render(str(idx), True, (255, 255, 0))
                screen.blit(txt, (px, py - 2))

        pygame.display.flip()
        clock.tick(10)

    pygame.quit()

if __name__ == "__main__":
    import sys
    sys.path.append('.')
    from env import GameENV

    seed = 132
    np.random.seed(seed)
    env = GameENV(num_pursuers=3, seed=seed)
    observe = env.reset()
    env_image = env_to_image(observe)

    ttr_result = compute_ttr(env_image, pursuers_world=observe['pursuers_states'])
    targets, dist_map = select_targets(ttr_result, env_image)

    actions = ttr_pursuer_policy(observe)
    print("Pursuer Actions:", actions)

    # visualize using the last_match_list computed by ttr_pursuer_policy
    visualize_ttr(env_image, ttr_result, targets, last_match_list, dist_map)
