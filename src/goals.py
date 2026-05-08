import torch
from config import *

# Based on your config, LANE_CENTRES = [1.5, 5.5, 9.5, 13.5, 17.5]
# Rightmost lane is 17.5 (or 1.5 depending on your coordinate convention)
# Let's assume the rightmost lane is the one with the highest index
LANE_CENTRES = torch.tensor([1.85, 5.55, 9.25, 12.95, 16.65], device=DEVICE)
RIGHT_LANE_Y = LANE_CENTRES[-1] 

def sample_goals(state, t_step, mode="cruise"):
    """
    Generate L world-frame goal hypotheses (Section III-F).
    
    Parameters:
        state: dict with 'x', 'y', 'vx'
        mode: "cruise" or "highway"
    """
    x0 = state["x"]
    y0 = state["y"]
    
    # planning horizon tf = N * DT
    tf = (N-1) * DT
    
    x_goals = []
    y_goals = []

    if mode == "cruise":
        # Scenario 1: Cruise Driving (Eq. 25)
        # Spread goals evenly on different lanes at distance VCRUISE * tf
        dist = VCRUISE * tf
        for i in range(L):
            lane_idx = i % len(LANE_CENTRES)

            x_goals.append(x0 + dist)  # All goals at the same longitudinal distance
            print(x_goals[-1])
            y_goals.append(LANE_CENTRES[lane_idx].item())

    else: # Highway Mode
        # Scenario 2: Max Speed close to Right Lane (Eq. 26)
        # 60% goals on the right lane at different distances
        # 40% goals spread across all lanes at VMAX * tf
        dist_max = VMAX * tf
        n_rl = int(0.6 * L)
        n_spread = L - n_rl
        
        # 60% on Right Lane (varying longitudinal distances)
        # Using a range around the target distance (e.g., 0.8 to 1.2 x target)
        dist_range = torch.linspace(0.8 * dist_max, 1.2 * dist_max, n_rl)
        for d in dist_range:
            x_goals.append(x0 + d.item())
            y_goals.append(RIGHT_LANE_Y.item())
            
        # 40% spread across other lanes
        for i in range(n_spread):
            lane_idx = i % len(LANE_CENTRES)
            x_goals.append(x0 + dist_max)
            y_goals.append(LANE_CENTRES[lane_idx].item())

    x_t = torch.tensor(x_goals[:L], dtype=torch.float32, device=DEVICE)
    y_t = torch.tensor(y_goals[:L], dtype=torch.float32, device=DEVICE)
    
    return torch.stack([
        x_t,
        y_t,
        torch.zeros(L, device=DEVICE),           # psi_goal: road aligned
        torch.full((L,), VCRUISE if mode == "cruise" else VMAX, device=DEVICE),   # v_goal: target speed
    ], dim=1)