"""
Tests for constraints.py  —  run with: python src/tests/test_constraints.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from config import N, L, NUM_OBS, A_OBS, B_OBS, AMAX, DEVICE
from constraints import collision_reformulation, acceleration_reformulation

def _make_obs(x_offset=20.0):
    """Helper: obstacles far from ego."""
    obs_x = torch.full((N*NUM_OBS, L), x_offset, device=DEVICE)
    obs_y = torch.full((N*NUM_OBS, L), 0.0,       device=DEVICE)
    return obs_x, obs_y

def test_collision_shapes():
    x = torch.zeros(N, L, device=DEVICE)
    y = torch.zeros(N, L, device=DEVICE)
    obs_x, obs_y = _make_obs()
    alpha, d = collision_reformulation(x, y, obs_x, obs_y)
    assert alpha.shape == (N*NUM_OBS, L), f"alpha shape {alpha.shape}"
    assert d.shape     == (N*NUM_OBS, L), f"d shape {d.shape}"
    print("PASS  collision output shapes")

def test_collision_far_obstacle():
    """Obstacle far away: d should be >> 1 and not clamped."""
    x = torch.zeros(N, L, device=DEVICE)
    y = torch.zeros(N, L, device=DEVICE)
    obs_x, obs_y = _make_obs(x_offset=100.0)
    _, d = collision_reformulation(x, y, obs_x, obs_y)
    assert (d > 2.0).all(), f"Expected d >> 1 for distant obstacle, got min {d.min():.3f}"
    print("PASS  far obstacle gives d >> 1")

def test_collision_clamp():
    """Ego overlapping obstacle: d must be clamped to >= 1."""
    x = torch.zeros(N, L, device=DEVICE)
    y = torch.zeros(N, L, device=DEVICE)
    obs_x = torch.zeros(N*NUM_OBS, L, device=DEVICE)   # same position as ego
    obs_y = torch.zeros(N*NUM_OBS, L, device=DEVICE)
    _, d = collision_reformulation(x, y, obs_x, obs_y)
    assert (d >= 1.0).all(), f"d clamp failed, min = {d.min():.4f}"
    print("PASS  overlapping obstacle clamped to d >= 1")

def test_collision_alpha_direction():
    """Obstacle to the left: alpha should be near 0 (ego is to the right)."""
    x   = torch.zeros(N, L, device=DEVICE)
    y   = torch.zeros(N, L, device=DEVICE)
    obs_x = torch.full((N*NUM_OBS, L), -10.0, device=DEVICE)  # obstacle to the left
    obs_y = torch.zeros(N*NUM_OBS, L, device=DEVICE)
    alpha, _ = collision_reformulation(x, y, obs_x, obs_y)
    # dx = x - obs_x = 0 - (-10) = +10, dy = 0 => alpha = atan2(0, A_OBS*10) ≈ 0
    assert (alpha.abs() < 0.1).all(), f"Expected alpha~0 for rightward obstacle, got {alpha[0,0]:.4f}"
    print("PASS  collision alpha direction correct")

def test_collision_ellipse_asymmetry():
    """With A_OBS != B_OBS, equal dx/dy should give different d values."""
    x = torch.zeros(N, L, device=DEVICE)
    y = torch.zeros(N, L, device=DEVICE)
    # obstacle displaced purely in x
    obs_x_only = torch.full((N*NUM_OBS, L), -A_OBS * 1.5, device=DEVICE)
    obs_y_zero = torch.zeros(N*NUM_OBS, L, device=DEVICE)
    _, d_x = collision_reformulation(x, y, obs_x_only, obs_y_zero)

    # obstacle displaced purely in y by same physical distance
    obs_x_zero = torch.zeros(N*NUM_OBS, L, device=DEVICE)
    obs_y_only = torch.full((N*NUM_OBS, L), -A_OBS * 1.5, device=DEVICE)
    _, d_y = collision_reformulation(x, y, obs_x_zero, obs_y_only)

    # d along x uses A_OBS, d along y uses B_OBS — they should differ
    assert not torch.allclose(d_x, d_y, atol=0.1), \
        "d_x and d_y should differ due to ellipse asymmetry"
    print("PASS  ellipse asymmetry in d values")

def test_accel_shapes():
    x_ddot = torch.randn(N, L, device=DEVICE)
    y_ddot = torch.randn(N, L, device=DEVICE)
    alpha_a, d_a = acceleration_reformulation(x_ddot, y_ddot)
    assert alpha_a.shape == (N, L), f"alpha_a shape {alpha_a.shape}"
    assert d_a.shape     == (N, L), f"d_a shape {d_a.shape}"
    print("PASS  acceleration output shapes")

def test_accel_clamp():
    """Large acceleration must be clamped to AMAX."""
    x_ddot = torch.full((N, L), 100.0, device=DEVICE)
    y_ddot = torch.full((N, L), 100.0, device=DEVICE)
    _, d_a = acceleration_reformulation(x_ddot, y_ddot)
    assert (d_a <= AMAX + 1e-5).all(), f"d_a clamp failed, max = {d_a.max():.4f}"
    print("PASS  large acceleration clamped to AMAX")

def test_accel_zero():
    """Zero acceleration: d_a = 0 (before clamp, stays 0 since 0 <= AMAX)."""
    x_ddot = torch.zeros(N, L, device=DEVICE)
    y_ddot = torch.zeros(N, L, device=DEVICE)
    _, d_a = acceleration_reformulation(x_ddot, y_ddot)
    assert (d_a == 0.0).all(), f"Expected d_a=0 for zero accel, got {d_a.max():.4f}"
    print("PASS  zero acceleration gives d_a = 0")

def test_accel_alpha_direction():
    """Pure x acceleration: alpha_a should be 0."""
    x_ddot = torch.ones(N, L, device=DEVICE) * 2.0
    y_ddot = torch.zeros(N, L, device=DEVICE)
    alpha_a, _ = acceleration_reformulation(x_ddot, y_ddot)
    assert torch.allclose(alpha_a, torch.zeros(N, L, device=DEVICE), atol=1e-5), \
        f"alpha_a should be 0 for pure x accel, got {alpha_a.abs().max():.4f}"
    print("PASS  pure x acceleration gives alpha_a = 0")

def test_accel_alpha_90():
    """Pure y acceleration: alpha_a should be pi/2."""
    x_ddot = torch.zeros(N, L, device=DEVICE)
    y_ddot = torch.ones(N, L, device=DEVICE) * 2.0
    alpha_a, _ = acceleration_reformulation(x_ddot, y_ddot)
    assert torch.allclose(alpha_a, torch.full((N, L), torch.pi/2, device=DEVICE), atol=1e-5), \
        f"alpha_a should be pi/2 for pure y accel"
    print("PASS  pure y acceleration gives alpha_a = pi/2")

if __name__ == "__main__":
    test_collision_shapes()
    test_collision_far_obstacle()
    test_collision_clamp()
    test_collision_alpha_direction()
    test_collision_ellipse_asymmetry()
    test_accel_shapes()
    test_accel_clamp()
    test_accel_zero()
    test_accel_alpha_direction()
    test_accel_alpha_90()
    print("\nAll constraints tests passed.")
