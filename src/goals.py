import torch
from config import L, N, DT, VMAX, DEVICE

def sample_goals(state):
    """
    Generate L goal hypotheses for the batch optimizer.

    state: dict or tuple with ego position (x0, y0), heading psi0, velocity v0

    Returns:
        goals: (L, 4) tensor of (x_goal, y_goal, psi_goal, v_goal)
    """

    # tf = N * DT  (planning horizon in seconds)
    # TODO:
    # - spread x_goal across lanes: x0 + v0 * tf
    # - vary y_goal across lane center-lines (e.g. -4, 0, +4 m offsets)
    # - ~60% of goals on right lane at different distances (Eq. 26 strategy)
    # - psi_goal = 0 (aligned with road), v_goal = VMAX or v0

    goals = ...
    return goals