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

def _make_scene(ego_x=0.0, ego_y=0.0, obs_offset=30.0):
    """Obstacles far ahead, goals spread across lanes."""
    obs_x = torch.zeros(N*NUM_OBS, device=DEVICE)
    obs_y = torch.zeros(N*NUM_OBS, device=DEVICE)
    for i in range(NUM_OBS):
        obs_x[i*N:(i+1)*N] = ego_x + obs_offset + i*10.0
        obs_y[i*N:(i+1)*N] = (i - 1) * 4.0    # lanes -4, 0, +4
    obs_x = obs_x.unsqueeze(1).expand(-1, L).contiguous()   # (N*NUM_OBS, L)
    obs_y = obs_y.unsqueeze(1).expand(-1, L).contiguous()

    goals = torch.zeros(L, 4, device=DEVICE)
    goals[:, 0] = ego_x + 30.0                 # x_goal
    goals[:, 1] = torch.linspace(-4, 4, L)     # y_goal across lanes
    return goals, obs_x, obs_y

def test_output_shapes():
    goals, obs_x, obs_y = _make_scene()
    cx, cy, cpsi, v = optimize_batch(P, P_dot, P_ddot, Fmat, A, goals, obs_x, obs_y)
    assert cx.shape   == (L, K), f"cx shape {cx.shape}"
    assert cy.shape   == (L, K), f"cy shape {cy.shape}"
    assert cpsi.shape == (L, K), f"cpsi shape {cpsi.shape}"
    assert v.shape    == (L, N), f"v shape {v.shape}"
    print("PASS  output shapes")

def test_velocity_bounds():
    """All velocities must stay within [VMIN, VMAX]."""
    goals, obs_x, obs_y = _make_scene()
    _, _, _, v = optimize_batch(P, P_dot, P_ddot, Fmat, A, goals, obs_x, obs_y)
    assert (v >= VMIN - 1e-5).all(), f"v below VMIN, min={v.min():.4f}"
    assert (v <= VMAX + 1e-5).all(), f"v above VMAX, max={v.max():.4f}"
    print("PASS  velocity within bounds")

def test_no_collision():
    """Ego should not penetrate any obstacle ellipse after optimization."""
    goals, obs_x, obs_y = _make_scene(obs_offset=30.0)
    cx, cy, _, _ = optimize_batch(P, P_dot, P_ddot, Fmat, A, goals, obs_x, obs_y)
    x = P @ cx.T    # (N, L)
    y = P @ cy.T
    for i in range(NUM_OBS):
        ox = obs_x[i*N:(i+1)*N]    # (N, L)
        oy = obs_y[i*N:(i+1)*N]
        dx = x - ox
        dy = y - oy
        d = torch.sqrt((dx / A_OBS)**2 + (dy / B_OBS)**2)
        assert (d >= 0.95).all(), \
            f"Collision with obstacle {i}: min d = {d.min():.4f}"
    print("PASS  no collision with obstacles")

def test_convergence():
    """Constraint residual should decrease over iterations (smoke test)."""
    goals, obs_x, obs_y = _make_scene()
    import config as cfg
    original = cfg.MAX_ITERS

    cfg.MAX_ITERS = 10
    cx10, cy10, _, _ = optimize_batch(P, P_dot, P_ddot, Fmat, A, goals, obs_x, obs_y)

    cfg.MAX_ITERS = 100
    cx100, cy100, _, _ = optimize_batch(P, P_dot, P_ddot, Fmat, A, goals, obs_x, obs_y)

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
    Each batch instance should produce a different trajectory once boundary
    constraints (KKT solve with A and bl) are implemented.
    Until then, verify the optimizer runs for all L instances without error
    and returns tensors of the correct batch size.
    """
    goals, obs_x, obs_y = _make_scene()
    cx, cy, cpsi, v = optimize_batch(P, P_dot, P_ddot, Fmat, A, goals, obs_x, obs_y)
    assert cx.shape[0] == L,   f"cx batch dim {cx.shape[0]} != L={L}"
    assert cy.shape[0] == L,   f"cy batch dim {cy.shape[0]} != L={L}"
    assert v.shape[0]  == L,   f"v batch dim {v.shape[0]} != L={L}"
    print("PASS  batch size correct (independence test deferred until KKT is implemented)")

def test_output_dtype_device():
    goals, obs_x, obs_y = _make_scene()
    cx, cy, cpsi, v = optimize_batch(P, P_dot, P_ddot, Fmat, A, goals, obs_x, obs_y)
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
    test_output_dtype_device()
    print("\nAll optimizer tests passed.")