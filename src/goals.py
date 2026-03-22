"""
Goal hypothesis sampling for multi-modal MPC.

Paper reference: Section III-F (Goal Sampling and Meta-Cost), Eq. (26).

The batch optimizer runs L trajectory optimisations in parallel, each
directed toward a different goal hypothesis. Diverse goals are critical
for multi-modal behaviour: if all goals are in the same lane as an
obstacle ahead, the optimizer cannot generate lateral avoidance manoeuvres
regardless of the ADMM penalty strength.
"""

import torch
from config import L, N, DT, VMAX, DEVICE, B_OBS

# Lane centre lateral positions in the NGSIM US-101 road frame [m]
LANE_CENTRES = torch.tensor([1.5, 5.5, 9.5, 13.5, 17.5], device=DEVICE)


def sample_goals(state, neighbors=None):
    """
    Generate L world-frame goal hypotheses for the batch optimizer.

    Strategy (Section III-F):
    - Determine the ego's current lane.
    - Identify lanes blocked by neighbors within the planning horizon.
    - If the ego lane is free: place L//2 goals in the ego lane at varied
      longitudinal distances, and L//2 goals in adjacent free lanes.
    - If the ego lane is blocked: place all L goals in adjacent free lanes,
      forcing the optimizer to generate lane-change trajectories.

    Parameters
    ----------
    state : dict
        Ego state with keys 'x', 'y', 'vx'.
    neighbors : list of dict, optional
        Current MPC obstacle neighbors. Each dict has keys 'x', 'y', 'vx'.
        Used to detect blocked lanes. If None, no lane-blocking is assumed.

    Returns
    -------
    goals : (L, 4)
        Each row is (x_goal, y_goal, ψ_goal, v_goal) in world frame.
        ψ_goal = 0 (road-aligned), v_goal = VMAX (Section III-F, Eq. 26).
    """
    x0 = state["x"]
    y0 = state["y"]
    v0 = max(float(state["vx"]), 1.0)

    goal_dist    = v0 * N * DT
    dists        = torch.abs(LANE_CENTRES - y0)
    ego_lane_idx = int(torch.argmin(dists).item())

    # Detect lanes blocked by neighbors ahead within the planning horizon
    blocked = set()
    if neighbors:
        for nb in neighbors:
            if nb.get("vehicle_id", 0) < 0:
                continue   # dummy padding vehicle — ignore
            dx = nb["x"] - x0
            if 0 < dx < goal_dist + 10.0:
                nb_lane = int(torch.argmin(
                    torch.abs(LANE_CENTRES - nb["y"])).item())
                if abs(nb["y"] - LANE_CENTRES[nb_lane].item()) < B_OBS * 1.5:
                    blocked.add(nb_lane)

    ego_blocked = ego_lane_idx in blocked

    # Identify adjacent lanes (left and right of ego lane)
    n_lanes = len(LANE_CENTRES)
    adj = []
    if ego_lane_idx > 0:
        adj.append(ego_lane_idx - 1)
    if ego_lane_idx < n_lanes - 1:
        adj.append(ego_lane_idx + 1)

    free = [i for i in range(n_lanes) if i not in blocked]
    if not free:
        free = adj if adj else [ego_lane_idx]

    x_goals = []
    y_goals = []

    if not ego_blocked:
        # L//2 goals in ego lane at longitudinal distances in [0.8, 1.2] × goal_dist
        for xg in (x0 + goal_dist * torch.linspace(0.8, 1.2, L // 2, device=DEVICE)).tolist():
            x_goals.append(xg)
            y_goals.append(LANE_CENTRES[ego_lane_idx].item())
        # L//2 goals spread across adjacent free lanes at goal_dist
        spread = adj if adj else free
        for i in range(L // 2):
            lane = spread[i % len(spread)]
            x_goals.append(x0 + goal_dist)
            y_goals.append(LANE_CENTRES[lane].item())
    else:
        # Ego lane blocked — all L goals in adjacent free lanes (Eq. 26 strategy)
        spread = [i for i in free if i != ego_lane_idx]
        if not spread:
            spread = free
        for i in range(L):
            lane = spread[i % len(spread)]
            x_goals.append(x0 + goal_dist * (0.8 + 0.4 * i / max(L - 1, 1)))
            y_goals.append(LANE_CENTRES[lane].item())

    x_t = torch.tensor(x_goals[:L], dtype=torch.float32, device=DEVICE)
    y_t = torch.tensor(y_goals[:L], dtype=torch.float32, device=DEVICE)

    return torch.stack([
        x_t,
        y_t,
        torch.zeros(L, device=DEVICE),           # ψ_goal = 0 (road-aligned)
        torch.full((L,), VMAX, device=DEVICE),   # v_goal = VMAX (Eq. 26)
    ], dim=1)