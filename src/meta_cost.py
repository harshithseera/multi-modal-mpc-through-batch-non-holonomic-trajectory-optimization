import torch
from config import VMAX, DEVICE

def compute_meta_cost(v, y, mode="cruise", v_cruise=10.0, y_right=0.0, w1=1.0, w2=1.0):
    """
    Rank trajectories by meta cost (Section III-F).

    v: (L, N)  velocity profiles  — as returned by optimize_batch
    y: (N, L)  lateral position profiles  — P @ cy.T from optimize_batch
               Must be transposed to (L, N) before computing cost:
               y_LN = y.T  so both v and y are (L, N) when summing over time axis

    Returns:
        cost: (L,)  scalar cost per trajectory (lower is better)
    """

    # TODO:
    # Cruise driving (Eq. 25):
    #   cost = sum_t (v(t) - v_cruise)**2

    # High-speed + right-lane preference (Eq. 26):
    #   cost = sum_t  w1*(v(t) - vmax)**2 + w2*(y(t) - y_right)**2

    cost = ...
    return cost