"""
Constraint reformulations for the multi-convex ADMM solver.

Paper reference: Section III-B (Building Blocks).

Collision avoidance and acceleration bounds are both expressed as
equality constraints of the form f(c) = 0, enabling the alternating
minimisation structure of Algorithm 1.
"""

import torch
from config import A_OBS, B_OBS, AMAX


def collision_reformulation(x, y, obs_x, obs_y):
    """
    Reformulate the quadratic collision avoidance constraint (Eq. 1e) into
    polar form (Eq. 4), and compute the closed-form update for the
    auxiliary variables α and d (Step 15 of Algorithm 1, Eq. 22).

    The constraint Eq. (1e) is:
        -(x - ξx)²/a² - (y - ξy)²/b² + 1 ≤ 0

    This is rephrased as Eq. (4):
        x - ξx = a · d · cos(α)
        y - ξy = b · d · sin(α)

    with d ≥ 1 enforcing clearance. The closed-form solution for α given
    the current trajectory (x, y) is Eq. (22):
        α = atan2(a · (y - ξy),  b · (x - ξx))

    and d is the normalised elliptical distance, clamped to [1, ∞).

    Parameters
    ----------
    x, y : (N, L)
        Ego trajectory positions in ego-relative coordinates.
    obs_x, obs_y : (N * num_obs, L)
        Predicted obstacle positions in ego-relative coordinates.

    Returns
    -------
    alpha : (N * num_obs, L)
        Line-of-sight angle from obstacle to ego (Eq. 22).
    d : (N * num_obs, L)
        Normalised elliptical separation distance; d ≥ 1 (Eq. 4).
    """
    num_obs = obs_x.shape[0] // x.shape[0]

    x_tiled = x.repeat(num_obs, 1)
    y_tiled = y.repeat(num_obs, 1)

    dx = x_tiled - obs_x
    dy = y_tiled - obs_y

    # Eq. (22): closed-form α update
    alpha = torch.atan2(A_OBS * dy, B_OBS * dx)

    # Eq. (4): normalised elliptical distance; clamp enforces d ≥ 1
    d = torch.sqrt((dx / A_OBS)**2 + (dy / B_OBS)**2 + 1e-6)
    d = torch.clamp(d, min=1.0)

    return alpha, d


def acceleration_reformulation(x_ddot, y_ddot):
    """
    Reformulate the quadratic acceleration bound (Eq. 1d) into polar form
    (Eq. 5), and compute the closed-form update for αa and da
    (Step 16 of Algorithm 1).

    The bound ‖ẍ‖² + ‖ÿ‖² ≤ amax² is rephrased as Eq. (5):
        ẍ = da · cos(αa)
        ÿ = da · sin(αa)

    with da ≤ amax. The closed-form solution is:
        αa = atan2(ÿ, ẍ)
        da = ‖(ẍ, ÿ)‖₂, clamped to [0, amax]

    Parameters
    ----------
    x_ddot, y_ddot : (N, L)
        Second τ-derivatives of the ego trajectory (scaled to physical time
        inside the optimizer before being passed here).

    Returns
    -------
    alpha_a : (N, L)
        Direction of the acceleration vector (Eq. 5).
    d_a : (N, L)
        Magnitude of the acceleration vector, clamped to amax (Eq. 5).
    """
    alpha_a = torch.atan2(y_ddot, x_ddot)

    d_a = torch.sqrt(x_ddot**2 + y_ddot**2)
    d_a = torch.clamp(d_a, max=AMAX)

    return alpha_a, d_a