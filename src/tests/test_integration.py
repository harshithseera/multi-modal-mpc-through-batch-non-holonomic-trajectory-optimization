"""
Integration test for the full MPC loop  —  run with: python src/tests/test_integration.py
Run this last, after all individual module tests pass.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from config import N, K, L, NUM_OBS, DT, VMAX, A_OBS, B_OBS, DEVICE
from basis import build_basis
from matrices import build_F, build_A
from optimizer import optimize_batch
from goals import sample_goals
from data import get_state, predict_obstacles
from meta_cost import compute_meta_cost

P, P_dot, P_ddot = build_basis()
Fmat = build_F(P, P_dot, P_ddot, num_obs=NUM_OBS)
A    = build_A(P)

def test_full_mpc_step():
    """Single MPC cycle completes without error and returns sane values."""
    ego, neighbors  = get_state(0)
    goals           = sample_goals(ego)
    obs_x, obs_y    = predict_obstacles(neighbors)
    cx, cy, cpsi, v = optimize_batch(P, P_dot, P_ddot, Fmat, A, goals, obs_x, obs_y, ego)
    y_pos           = P @ cy.T
    cost            = compute_meta_cost(v, y_pos)
    best            = cost.argmin()

    assert 0 <= best.item() < L,  f"best index {best} out of range"
    assert not cost.isnan().any(), "meta_cost returned NaN"
    assert not v.isnan().any(),    "v contains NaN"
    assert not cx.isnan().any(),   "cx contains NaN"
    print("PASS  single MPC step completes cleanly")

def test_mpc_loop_runs():
    """10 MPC cycles without crash."""
    ego, neighbors = get_state(0)

    for t in range(10):
        goals           = sample_goals(ego)
        obs_x, obs_y    = predict_obstacles(neighbors)
        cx, cy, cpsi, v = optimize_batch(P, P_dot, P_ddot, Fmat, A, goals, obs_x, obs_y, ego)
        y_pos           = P @ cy.T
        cost            = compute_meta_cost(v, y_pos)
        best            = cost.argmin()

        best_cx = cx[best]
        best_cy = cy[best]
        ego = {
            "x":   (P[1]     @ best_cx).item(),
            "y":   (P[1]     @ best_cy).item(),
            "psi": 0.0,
            "vx":  (P_dot[1] @ best_cx).item() / DT,
            "vy":  (P_dot[1] @ best_cy).item() / DT,
        }
        _, neighbors = get_state(t + 1)

    print("PASS  10 MPC cycles complete without error")

def test_ego_moves_forward():
    """Ego x position should be further forward after 5 MPC steps than at the start."""
    ego, neighbors = get_state(0)
    x_start = ego["x"]

    # Use N//4 index — far enough along the polynomial that forward motion
    # is established. P[1] is only t=1/29 into the horizon where the
    # polynomial may still be near the start value.
    step_idx = N // 4

    for t in range(5):
        goals           = sample_goals(ego)
        obs_x, obs_y    = predict_obstacles(neighbors)
        cx, cy, cpsi, v = optimize_batch(P, P_dot, P_ddot, Fmat, A, goals, obs_x, obs_y, ego)
        cost            = compute_meta_cost(v, P @ cy.T)
        best            = cost.argmin()

        best_cx  = cx[best]
        best_cy  = cy[best]
        ego["x"] = (P[step_idx] @ best_cx).item()
        ego["y"] = (P[step_idx] @ best_cy).item()
        ego["vx"] = (P_dot[step_idx] @ best_cx).item() / DT
        ego["vy"] = 0.0
        _, neighbors = get_state(t + 1)

    x_end = ego["x"]
    assert x_end > x_start, \
        f"Ego did not move forward overall: start={x_start:.2f}, end={x_end:.2f}"
    print(f"PASS  ego moved forward: {x_start:.2f} -> {x_end:.2f}")

def test_no_nan_over_loop():
    """No NaN or Inf in any output over 20 steps."""
    ego, neighbors = get_state(0)

    for t in range(20):
        goals           = sample_goals(ego)
        obs_x, obs_y    = predict_obstacles(neighbors)
        cx, cy, cpsi, v = optimize_batch(P, P_dot, P_ddot, Fmat, A, goals, obs_x, obs_y, ego)

        for name, tensor in [("cx", cx), ("cy", cy), ("cpsi", cpsi), ("v", v)]:
            assert not tensor.isnan().any(), f"NaN in {name} at step {t}"
            assert not tensor.isinf().any(), f"Inf in {name} at step {t}"

        cost    = compute_meta_cost(v, P @ cy.T)
        best_cx = cx[cost.argmin()]
        ego["x"] = (P[1] @ best_cx).item()
        _, neighbors = get_state(t + 1)

    print("PASS  no NaN/Inf over 20 MPC steps")

if __name__ == "__main__":
    test_full_mpc_step()
    test_mpc_loop_runs()
    test_ego_moves_forward()
    test_no_nan_over_loop()
    print("\nAll integration tests passed.")