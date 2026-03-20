import torch
from config import N, NUM_OBS, DEVICE

def get_state(t):
    """
    Return ego + neighbors at timestep t.

    ego:       dict with keys x, y, psi, vx, vy
    neighbors: list of NUM_OBS dicts with same keys
    """

    # TODO:
    # Option A: load from NGSIM dataset (pre-recorded trajectories)
    # Option B: synthetic IDM scenario (neighbors drive parallel,
    #           adapt speed based on distance to vehicle ahead)

    ego = ...
    neighbors = ...

    return ego, neighbors


def predict_obstacles(neighbors):
    """
    Constant-velocity prediction over N timesteps (Section III assumption 3).

    Returns:
        obs_x: (N*NUM_OBS, L)  stacked x positions — repeated L times to broadcast with x (N, L)
        obs_y: (N*NUM_OBS, L)

    Formula: xi(tk) = x0 + vx * tk,  yi(tk) = y0 + vy * tk
    """

    # TODO:
    # for each neighbor: positions = x0 + v * t_vec  where t_vec shape (N,)
    # stack all neighbors -> (N*NUM_OBS,)

    obs_x = ...
    obs_y = ...

    return obs_x, obs_y