"""
Constraint reformulations (Section III-B, III-C).

Implements:
- Collision avoidance (Eq. 4, Eq. 22)
- Acceleration constraints (Eq. 5)
"""

import torch
from config import A_OBS, B_OBS, AMAX


def collision_reformulation(x, y, obs_x, obs_y):
    """
    Reformulates collision constraints into convex form.

    Args:
        x, y:         (N, L) ego trajectory
        obs_x, obs_y: (N*num_obs, L) predicted obstacles

    Returns:
        alpha: (N*num_obs, L) angle
        d:     (N*num_obs, L) normalized distance (>=1 ensures no collision)
    """

    num_obs = obs_x.shape[0] // x.shape[0]

    x_tiled = x.repeat(num_obs, 1)
    y_tiled = y.repeat(num_obs, 1)

    dx = x_tiled - obs_x
    dy = y_tiled - obs_y

    # Eq. (22)
    alpha = torch.atan2(A_OBS * dy, B_OBS * dx)

    # Eq. (4)
    d = torch.sqrt((dx / A_OBS)**2 + (dy / B_OBS)**2 + 1e-6)
    d = torch.clamp(d, min=1.0)

    return alpha, d


def acceleration_reformulation(x_ddot, y_ddot):
    """
    Reformulates acceleration constraints.

    Returns:
        alpha_a: direction of acceleration
        d_a:     magnitude (clamped to AMAX)
    """

    alpha_a = torch.atan2(y_ddot, x_ddot)
    d_a = torch.sqrt(x_ddot**2 + y_ddot**2)

    d_a = torch.clamp(d_a, max=AMAX)

    return alpha_a, d_a