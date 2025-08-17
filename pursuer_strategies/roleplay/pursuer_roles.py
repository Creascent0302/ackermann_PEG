from __future__ import annotations
from typing import List, Tuple, Dict
import numpy as np
def direct_pursuer(evader, last_evader_observe,match_list, role_list: List[str] = None, prediction_steps: int = 2):
    """
    Simple direct pursuer target: predict evader future position by linear motion
    and return that as the intercept point.
    """
    ev_pos = np.array(evader[:2])
    prev = np.array(last_evader_observe) if last_evader_observe else ev_pos
    ev_motion = ev_pos - prev
    predicted = ev_pos + prediction_steps * ev_motion
    return tuple(predicted.tolist())

def encircle_pursuer(
    p_idx: int,
    pursuer,
    targets: List[Tuple[float, float]],
    target_grids: List[Tuple[int, int]],
    pursuer_ttr,
    ev_pos: np.ndarray,
    last_evader_observe,
    match_list,
    use_ttr_cost: bool = True,
    role_list: List[str] = None,
    prediction_steps: int = 10,
    ):
    """
    Choose an encircle target for pursuer p_idx.
    Cost = 0.6*TTR(at target cell) + predicted_evader_distance + 0.1 * distance_from_line_between_direct_pursuer_and_evader.
    Falls back to nearest target by Euclidean distance if TTR unavailable.
    Adds a penalty when the same target is already selected in match_list.
    """
    prev = np.array(last_evader_observe) if last_evader_observe else ev_pos
    ev_motion = ev_pos - prev
    predicted_ev = ev_pos + prediction_steps * ev_motion
    pursuer_pos = np.array(pursuer[:2])
    # ensure role_list is accessed (optional parameter); normalize to list for downstream use
    role_list = role_list or []

    # compute direct pursuer intercept point once (use ev_pos as evader input)
    intercept_pt = ev_pos
    best_tid = None
    best_cost = float('inf')

    # shapes
    if pursuer_ttr is not None:
        H, W = pursuer_ttr.shape[1], pursuer_ttr.shape[2]
    else:
        H = W = 0

    # penalty for choosing a target already assigned in match_list
    MATCH_PENALTY_PER_ASSIGNMENT = 100.0

    for tid, (tw, tg) in enumerate(zip(targets, target_grids)):
        gx, gy = tg
        tw_arr = np.array(tw)
        # base distance cost from pursuer to target
        dist_cost = np.linalg.norm(pursuer_pos - tw_arr)

        # distance from target to the infinite line defined by intercept_pt -> ev_pos
        a = intercept_pt
        b = np.array(ev_pos)
        seg = b - a
        seg_norm = np.linalg.norm(seg)
        if seg_norm > 1e-8:
            # 2D cross product magnitude divided by segment length
            line_dist = abs(np.cross(seg, tw_arr - a)) / seg_norm
        else:
            # degenerate line: treat as distance to intercept point
            line_dist = np.linalg.norm(tw_arr - a)

        # convert line distance to a cost contribution (negative to favor smaller distance)
        line_dist_cost = -float(line_dist)

        # match_list penalty: count how many times this target is already matched
        duplicate_count = 0
        if match_list is not None:
            try:
                duplicate_count = sum(1 for m in match_list if m == tw)
            except Exception:
                # If match_list elements are not comparable to tid, ignore
                duplicate_count = 0
        match_penalty = MATCH_PENALTY_PER_ASSIGNMENT * duplicate_count

        if use_ttr_cost and 0 <= gx < H and 0 <= gy < W:
            v = pursuer_ttr[p_idx, gx, gy]
            if np.isfinite(v):
                prediction_cost = np.linalg.norm(predicted_ev - tw_arr)
                # include additional line distance cost and match penalty
                cost = 0.6 * float(v) + 3 * line_dist_cost + match_penalty
            else:
                cost = float('inf')
        else:
            # fallback cost when no valid TTR: distance + line distance penalty + match penalty
            cost = float(dist_cost) + line_dist_cost + match_penalty

        if cost < best_cost:
            best_cost = cost
            best_tid = tid
            print(line_dist)

    if best_tid is not None and best_cost < float('inf'):
        return targets[best_tid]
    # fallback to evader position
    return tuple(ev_pos.tolist())
