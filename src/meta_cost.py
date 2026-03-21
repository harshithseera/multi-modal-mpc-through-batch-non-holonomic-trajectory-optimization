import torch
from config import VMAX, DEVICE

def compute_meta_cost(v, y, mode="cruise", v_cruise=10.0, y_right=0.0, w1=1.0, w2=1.0):
    """
    Rank trajectories by meta cost (Section III-F).

    v: (L, N)  velocity profiles  — as returned by optimize_batch
    y: (N, L)  lateral position profiles  — P @ cy.T from optimize_batch
               Transposed to (L, N) internally before computing cost.

    Returns:
        cost: (L,)  scalar cost per trajectory (lower is better)
    """
    y_LN = y.T    # (L, N)

    if mode == "cruise":
        # Eq. (25): penalise deviation from cruise speed
        cost = ((v - v_cruise) ** 2).sum(dim=1)   # (L,)

    elif mode == "highway":
        # Eq. (26): penalise deviation from max speed AND distance to right lane
        cost = (
            w1 * (v    - VMAX)    ** 2 +
            w2 * (y_LN - y_right) ** 2
        ).sum(dim=1)                               # (L,)

    else:
        raise ValueError(f"Unknown mode '{mode}'. Use 'cruise' or 'highway'.")

    return cost