import torch
import numpy as np
from config import VCRUISE, VMAX

def compute_meta_cost(v, y, res_obs, mode="cruise"):
    """
    Official Implementation: Weighted Rank Sum.
    v: (L, N)
    y: (N, L)
    res_obs: (N*num_obs, L)
    """
    L_idx = v.shape[0]
    y_LN = y.T # (L, N)
    
    # 1. Define Weights from official config.yaml
    if mode == "cruise":
        w = [50.0, 50.0, 0.0, 0.0] # [Cruise, Optimal/Safety, RightLane, MaxV]
    else: # HSRL
        w = [0.0, 50.0, 25.0, 25.0]

    # 2. Calculate Raw Metrics (Official get_ranks logic)
    # Metric 1: Deviation from Cruise (15.0)
    m1 = torch.norm(v - VCRUISE, p=2, dim=1)
    # Metric 2: Collision Residual (Sum of violations)
    m2 = torch.norm(res_obs, p=2, dim=0) 
    # Metric 3: Rightmost Lane Adherence (official uses y = -10, we use LANE_CENTRES[0])
    m3 = torch.norm(y_LN - 1.85, p=2, dim=1)
    # Metric 4: Max Velocity (24.0)
    m4 = torch.norm(v - VMAX, p=2, dim=1)

    metrics = [m1, m2, m3, m4]
    ranks = torch.zeros((4, L_idx), device=v.device)

    # 3. Convert Metrics to Ranks (1 = best, L = worst)
    for i in range(4):
        # argsort gives indices of sorted values. 
        # argsort of argsort gives the rank of the original index.
        ranks[i] = torch.argsort(torch.argsort(metrics[i])).float() + 1.0

    # 4. Weighted Rank Sum
    total_cost = w[0]*ranks[0] + w[1]*ranks[1] + w[2]*ranks[2] + w[3]*ranks[3]
    
    return total_cost