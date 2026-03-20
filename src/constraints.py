import torch
from config import A_OBS, B_OBS, AMAX

# =====================================================
# Eq. (4): Collision Reformulation
# =====================================================

def collision_reformulation(x, y, obs_x, obs_y):
    """
    Inputs:
        x, y:         (N, L)
        obs_x, obs_y: (N*num_obs, L)

    Returns:
        alpha: (N*num_obs, L)  line-of-sight angle per obstacle per timestep
        d:     (N*num_obs, L)  normalized separation distance (enforced >= 1)
    """

    # TODO:
    # 1. dx = x - obs_x (broadcast over num_obs blocks), dy = y - obs_y
    # 2. alpha = atan2(A_OBS * dy, B_OBS * dx)   (Eq. 22)
    #    From Eq. (4): x - ξx = a*d*cos(α), y - ξy = b*d*sin(α)
    #    Eq. (22) explicitly: α = atan2(a*(y-ξy), b*(x-ξx))
    #    => atan2(A_OBS * dy, B_OBS * dx)
    # 3. d = sqrt((dx/A_OBS)**2 + (dy/B_OBS)**2)
    # 4. enforce d >= 1 via clamp: d = clamp(d, min=1.0)

    alpha = ...
    d = ...

    return alpha, d


# =====================================================
# Eq. (5): Acceleration Reformulation
# =====================================================

def acceleration_reformulation(x_ddot, y_ddot):
    """
    Inputs:
        x_ddot, y_ddot: (N, L)

    Returns:
        alpha_a: (N, L)  acceleration angle
        d_a:     (N, L)  acceleration magnitude (enforced <= AMAX)
    """

    # TODO:
    # 1. alpha_a = atan2(y_ddot, x_ddot)
    # 2. d_a     = sqrt(x_ddot**2 + y_ddot**2)
    # 3. enforce d_a <= AMAX via clamp: d_a = clamp(d_a, max=AMAX)

    alpha_a = ...
    d_a = ...

    return alpha_a, d_a