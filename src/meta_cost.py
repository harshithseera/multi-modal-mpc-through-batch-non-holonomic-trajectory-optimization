"""
Meta-cost functions for ranking batch trajectory hypotheses.

Paper reference: Section III-F (Goal Sampling and Meta-Cost), Eq. (25)–(26).

After the batch optimizer produces L locally optimal trajectories, a
scalar meta-cost is evaluated for each trajectory. The trajectory with
the lowest meta-cost is selected as the MPC output (receding horizon step).
"""

import torch
from config import VMAX, DEVICE


def compute_meta_cost(v, y, mode="cruise", v_cruise=10.0, y_right=0.0, w1=1.0, w2=1.0):
    """
    Compute a scalar meta-cost for each of the L trajectory hypotheses.

    Parameters
    ----------
    v : (L, N)
        Velocity profile for each batch instance, as returned by optimize_batch.
    y : (N, L)
        Lateral position profile; P @ cy.T from optimize_batch.
        Transposed to (L, N) internally before computing the sum.
    mode : str
        'cruise'  — Eq. (25): penalise deviation from v_cruise.
        'highway' — Eq. (26): penalise deviation from VMAX and lateral offset
                    from the right lane.
    v_cruise : float
        Target cruise speed [m/s] for mode='cruise'.
    y_right : float
        Lateral coordinate of the right lane [m] for mode='highway'.
    w1, w2 : float
        Weights for speed and lane terms in Eq. (26).

    Returns
    -------
    cost : (L,)
        Scalar meta-cost per trajectory; lower is better.
    """
    y_LN = y.T   # (L, N)

    if mode == "cruise":
        # Eq. (25): sum_t (v(t) - v_cruise)²
        cost = ((v - v_cruise) ** 2).sum(dim=1)

    elif mode == "highway":
        # Eq. (26): sum_t  w1·(v(t) - vmax)² + w2·(y(t) - y_right)²
        cost = (
            w1 * (v    - VMAX)    ** 2 +
            w2 * (y_LN - y_right) ** 2
        ).sum(dim=1)

    else:
        raise ValueError(f"Unknown mode '{mode}'. Use 'cruise' or 'highway'.")

    return cost