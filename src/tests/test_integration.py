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
    ego, neighbors   = get_state(0)
    goals            = sample_goals(ego)
    obs_x, obs_y     = predict_obstacles(neighbors)
    cx, cy, cpsi, v  = optimize_batch(P, P_dot, P_ddot, Fmat, A, goals, obs_x, obs_y)
    y_pos            = P @ cy.T
    cost             = compute_meta_cost(v, y_pos)
    best             = cost.argmin()

    assert 0 <= best.item() < L,         f"best index {best} out of range"
    assert not cost.isnan().any(),        "meta_cost returned NaN"
    assert not v.isnan().any(),           "v contains NaN"
    assert not cx.isnan().any(),          "cx contains NaN"
    print("PASS  single MPC step completes cleanly")

def test_mpc_loop_runs():
    """10 MPC cycles without crash."""
    ego, _ = get_state(0)

    for t in range(10):
        ego, neighbors  = get_state(t)
        goals           = sample_goals(ego)
        obs_x, obs_y    = predict_obstacles(neighbors)
        cx, cy, cpsi, v = optimize_batch(P, P_dot, P_ddot, Fmat, A, goals, obs_x, obs_y)
        y_pos           = P @ cy.T
        cost            = compute_meta_cost(v, y_pos)
        best            = cost.argmin()

        best_cx = cx[best]
        best_cy = cy[best]
        ego = {
            "x":   (P[1] @ best_cx).item(),
            "y":   (P[1] @ best_cy).item(),
            "psi": 0.0,
            "vx":  (P_dot[1] @ best_cx).item() / DT,
            "vy":  (P_dot[1] @ best_cy).item() / DT,
        }

    print("PASS  10 MPC cycles complete without error")

def test_ego_moves_forward():
    """Ego x position should increase each step."""
    ego, _ = get_state(0)
    x_positions = [ego["x"]]

    for t in range(5):
        ego, neighbors  = get_state(t)
        goals           = sample_goals(ego)
        obs_x, obs_y    = predict_obstacles(neighbors)
        cx, cy, cpsi, v = optimize_batch(P, P_dot, P_ddot, Fmat, A, goals, obs_x, obs_y)
        cost            = compute_meta_cost(v, P @ cy.T)
        best            = cost.argmin()

        best_cx = cx[best]
        ego["x"] = (P[1] @ best_cx).item()
        x_positions.append(ego["x"])

    assert all(x_positions[i] < x_positions[i+1] for i in range(len(x_positions)-1)), \
        f"Ego did not move forward: {x_positions}"
    print("PASS  ego moves forward each MPC step")

def test_no_nan_over_loop():
    """No NaN or Inf in any output over 20 steps."""
    ego, _ = get_state(0)
    cost = None

    for t in range(20):
        ego, neighbors  = get_state(t)
        goals           = sample_goals(ego)
        obs_x, obs_y    = predict_obstacles(neighbors)
        cx, cy, cpsi, v = optimize_batch(P, P_dot, P_ddot, Fmat, A, goals, obs_x, obs_y)

        for name, tensor in [("cx", cx), ("cy", cy), ("cpsi", cpsi), ("v", v)]:
            assert not tensor.isnan().any(), f"NaN in {name} at step {t}"
            assert not tensor.isinf().any(), f"Inf in {name} at step {t}"

        cost = compute_meta_cost(v, P @ cy.T)
        best_cx = cx[cost.argmin()]
        ego["x"] = (P[1] @ best_cx).item()

    print("PASS  no NaN/Inf over 20 MPC steps")

if __name__ == "__main__":
    test_full_mpc_step()
    test_mpc_loop_runs()
    test_ego_moves_forward()
    test_no_nan_over_loop()
    print("\nAll integration tests passed.")
