"""
Tests for optimizer.py  —  run with: python src/tests/test_optimizer.py
Requires basis.py, matrices.py, constraints.py all passing.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from config import N, K, L, NUM_OBS, VMIN, VMAX, A_OBS, B_OBS, DEVICE
from basis import build_basis
from matrices import build_F, build_A
from optimizer import optimize_batch

P, P_dot, P_ddot = build_basis()
Fmat = build_F(P, P_dot, P_ddot, num_obs=NUM_OBS)
A    = build_A(P)

def _make_scene(ego_x=0.0, ego_y=0.0, obs_offset=60.0):
    """Obstacles far ahead, goals at half the obstacle distance, dummy ego dict."""
    obs_x = torch.zeros(N*NUM_OBS, device=DEVICE)
    obs_y = torch.zeros(N*NUM_OBS, device=DEVICE)
    for i in range(NUM_OBS):
        obs_x[i*N:(i+1)*N] = ego_x + obs_offset + i*10.0
        obs_y[i*N:(i+1)*N] = (i - 1) * 4.0
    obs_x = obs_x.unsqueeze(1).expand(-1, L).contiguous()
    obs_y = obs_y.unsqueeze(1).expand(-1, L).contiguous()

    # Goals placed before the obstacles so KKT doesn't pull trajectory into them
    goals = torch.zeros(L, 4, device=DEVICE)
    goals[:, 0] = ego_x + obs_offset / 2        # halfway to obstacles
    goals[:, 1] = torch.linspace(-4, 4, L)

    ego = {"x": ego_x, "y": ego_y, "psi": 0.0, "vx": 10.0, "vy": 0.0}

    return goals, obs_x, obs_y, ego

def test_output_shapes():
    goals, obs_x, obs_y, ego = _make_scene()
    cx, cy, cpsi, v = optimize_batch(P, P_dot, P_ddot, Fmat, A, goals, obs_x, obs_y, ego)
    assert cx.shape   == (L, K), f"cx shape {cx.shape}"
    assert cy.shape   == (L, K), f"cy shape {cy.shape}"
    assert cpsi.shape == (L, K), f"cpsi shape {cpsi.shape}"
    assert v.shape    == (L, N), f"v shape {v.shape}"
    print("PASS  output shapes")

def test_velocity_bounds():
    """All velocities must stay within [VMIN, VMAX]."""
    goals, obs_x, obs_y, ego = _make_scene()
    _, _, _, v = optimize_batch(P, P_dot, P_ddot, Fmat, A, goals, obs_x, obs_y, ego)
    assert (v >= VMIN - 1e-5).all(), f"v below VMIN, min={v.min():.4f}"
    assert (v <= VMAX + 1e-5).all(), f"v above VMAX, max={v.max():.4f}"
    print("PASS  velocity within bounds")

def test_no_collision():
    """Ego should not penetrate any obstacle ellipse after optimization."""
    goals, obs_x, obs_y, ego = _make_scene()
    cx, cy, _, _ = optimize_batch(P, P_dot, P_ddot, Fmat, A, goals, obs_x, obs_y, ego)
    x = P @ cx.T
    y = P @ cy.T
    for i in range(NUM_OBS):
        ox = obs_x[i*N:(i+1)*N]
        oy = obs_y[i*N:(i+1)*N]
        dx = x - ox
        dy = y - oy
        d  = torch.sqrt((dx / A_OBS)**2 + (dy / B_OBS)**2)
        assert (d >= 0.95).all(), \
            f"Collision with obstacle {i}: min d = {d.min():.4f}"
    print("PASS  no collision with obstacles")

def test_convergence():
    """More iterations should reduce constraint violations."""
    goals, obs_x, obs_y, ego = _make_scene()
    import config as cfg
    original = cfg.MAX_ITERS

    cfg.MAX_ITERS = 10
    cx10, cy10, _, _ = optimize_batch(P, P_dot, P_ddot, Fmat, A, goals, obs_x, obs_y, ego)

    cfg.MAX_ITERS = 100
    cx100, cy100, _, _ = optimize_batch(P, P_dot, P_ddot, Fmat, A, goals, obs_x, obs_y, ego)

    cfg.MAX_ITERS = original

    from constraints import collision_reformulation
    x10  = P @ cx10.T;   y10  = P @ cy10.T
    x100 = P @ cx100.T;  y100 = P @ cy100.T
    _, d10  = collision_reformulation(x10,  y10,  obs_x, obs_y)
    _, d100 = collision_reformulation(x100, y100, obs_x, obs_y)

    viol10  = torch.clamp(1.0 - d10,  min=0).sum()
    viol100 = torch.clamp(1.0 - d100, min=0).sum()
    assert viol100 <= viol10 + 1e-3, \
        f"More iterations did not reduce violations: {viol10:.4f} -> {viol100:.4f}"
    print("PASS  more iterations reduce constraint violations")

def test_batch_independence():
    """
    With KKT implemented, different y_goals should produce different cy rows.
    """
    goals, obs_x, obs_y, ego = _make_scene()
    cx, cy, _, _ = optimize_batch(P, P_dot, P_ddot, Fmat, A, goals, obs_x, obs_y, ego)
    cy_std = cy.std(dim=0).max()
    assert cy_std > 1e-4, f"All batch instances produced same cy (std={cy_std:.6f})"
    print("PASS  batch instances produce distinct trajectories")

def test_boundary_conditions():
    """
    With KKT implemented, trajectories should start near ego position
    and end near their respective goals.
    """
    ego_x, ego_y = 50.0, 2.0
    goals, obs_x, obs_y, ego = _make_scene(ego_x=ego_x, ego_y=ego_y)
    cx, cy, _, _ = optimize_batch(P, P_dot, P_ddot, Fmat, A, goals, obs_x, obs_y, ego)

    x_traj = P @ cx.T    # (N, L)
    y_traj = P @ cy.T

    # x(0) should be close to ego_x for all L
    assert torch.allclose(x_traj[0], torch.full((L,), ego_x, device=DEVICE), atol=1.0), \
        f"x start not near ego_x: {x_traj[0]}"
    # y(0) should be close to ego_y for all L
    assert torch.allclose(y_traj[0], torch.full((L,), ego_y, device=DEVICE), atol=1.0), \
        f"y start not near ego_y: {y_traj[0]}"
    # x(T) should be close to each instance's x_goal
    assert torch.allclose(x_traj[-1], goals[:, 0], atol=2.0), \
        f"x end not near x_goals: {x_traj[-1]} vs {goals[:, 0]}"

    print("PASS  boundary conditions enforced by KKT")

def test_output_dtype_device():
    goals, obs_x, obs_y, ego = _make_scene()
    cx, cy, cpsi, v = optimize_batch(P, P_dot, P_ddot, Fmat, A, goals, obs_x, obs_y, ego)
    for name, t in [("cx", cx), ("cy", cy), ("cpsi", cpsi), ("v", v)]:
        assert t.dtype == torch.float32, f"{name} dtype {t.dtype}"
        assert str(t.device).startswith(DEVICE.split(":")[0]), f"{name} on wrong device"
    print("PASS  output dtype and device correct")

if __name__ == "__main__":
    test_output_shapes()
    test_velocity_bounds()
    test_no_collision()
    test_convergence()
    test_batch_independence()
    test_boundary_conditions()
    test_output_dtype_device()
    print("\nAll optimizer tests passed.")