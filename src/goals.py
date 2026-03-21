import torch
from config import L, N, DT, VMAX, DEVICE

# Lane centre-lines in road-attached frame (metres lateral from road centre)
LANE_CENTRES = [-4.0, 0.0, 4.0]   # right, centre, left

def sample_goals(state):
    """
    Generate L goal hypotheses for the batch optimizer.

    state: dict with keys x, y, psi, vx, vy

    Returns:
        goals: (L, 4) tensor of (x_goal, y_goal, psi_goal, v_goal)

    Goal sampling strategy (Section III-F, Eq. 26):
        ~60% of goals on the right lane at varied longitudinal distances.
        Remaining goals spread across all lane centres.
        psi_goal = 0 (aligned with road centre-line).
        v_goal   = VMAX for all instances.
    """
    x0 = state["x"]
    v0 = state["vx"] if state["vx"] > 0 else 10.0
    tf = N * DT                        # planning horizon [s]

    n_right  = round(0.6 * L)         # ~60% on right lane
    n_spread = L - n_right

    # Right-lane goals at varied distances [0.8*v0*tf … 1.2*v0*tf]
    x_right  = x0 + v0 * tf * torch.linspace(0.8, 1.2, n_right, device=DEVICE)
    y_right  = torch.full((n_right,), LANE_CENTRES[0], device=DEVICE)

    # Remaining goals spread across all lanes at v0*tf
    lane_cycle = [LANE_CENTRES[i % len(LANE_CENTRES)] for i in range(n_spread)]
    x_spread   = torch.full((n_spread,), x0 + v0 * tf, device=DEVICE)
    y_spread   = torch.tensor(lane_cycle, dtype=torch.float32, device=DEVICE)

    x_goals  = torch.cat([x_right,  x_spread])   # (L,)
    y_goals  = torch.cat([y_right,  y_spread])
    psi_goals = torch.zeros(L, device=DEVICE)
    v_goals   = torch.full((L,), VMAX, device=DEVICE)

    goals = torch.stack([x_goals, y_goals, psi_goals, v_goals], dim=1)  # (L, 4)
    return goals