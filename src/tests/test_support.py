"""
Tests for data.py, goals.py, meta_cost.py  —  run with: python src/tests/test_support.py
These modules have no inter-dependency so they are tested together.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from config import N, K, L, NUM_OBS, DT, VMAX, VMIN, DEVICE
from basis import build_basis

P, P_dot, P_ddot = build_basis()

# ─────────────────────────────────────────────
# data.py
# ─────────────────────────────────────────────

from data import get_state, predict_obstacles

def test_get_state_keys():
    ego, neighbors = get_state(0)
    for key in ("x", "y", "psi", "vx", "vy"):
        assert key in ego, f"ego missing key '{key}'"
    assert len(neighbors) == NUM_OBS, \
        f"Expected {NUM_OBS} neighbors, got {len(neighbors)}"
    for i, n in enumerate(neighbors):
        for key in ("x", "y", "vx", "vy"):
            assert key in n, f"neighbor {i} missing key '{key}'"
    print("PASS  get_state returns correct structure")

def test_get_state_advances():
    """Calling get_state at different t should return different positions (NGSIM/IDM)."""
    ego0, _ = get_state(0)
    ego5, _ = get_state(5)
    changed = any(ego0[k] != ego5[k] for k in ("x", "y"))
    assert changed, "get_state returned identical ego at t=0 and t=5"
    print("PASS  get_state advances with t")

def test_predict_obstacles_shape():
    _, neighbors = get_state(0)
    obs_x, obs_y = predict_obstacles(neighbors)
    assert obs_x.shape == (N * NUM_OBS, L), \
        f"obs_x shape {obs_x.shape}, expected ({N*NUM_OBS}, {L})"
    assert obs_y.shape == (N * NUM_OBS, L), \
        f"obs_y shape {obs_y.shape}"
    print("PASS  predict_obstacles output shape")

def test_predict_obstacles_linear():
    """Predicted positions should be linear in time (constant velocity)."""
    _, neighbors = get_state(0)
    obs_x, _ = predict_obstacles(neighbors)
    traj = obs_x[:N, 0]    # first obstacle, first batch instance
    diffs = traj[1:] - traj[:-1]
    assert torch.allclose(diffs, diffs[0].expand_as(diffs), atol=1e-4), \
        "Obstacle trajectory is not linear (constant velocity violated)"
    print("PASS  obstacle prediction is linear (constant velocity)")

# ─────────────────────────────────────────────
# goals.py
# ─────────────────────────────────────────────

from goals import sample_goals

def test_goals_shape():
    ego, _ = get_state(0)
    goals = sample_goals(ego)
    assert goals.shape == (L, 4), f"goals shape {goals.shape}, expected ({L}, 4)"
    print("PASS  goals shape")

def test_goals_x_ahead():
    """All x_goals should be ahead of ego."""
    ego, _ = get_state(0)
    goals = sample_goals(ego)
    assert (goals[:, 0] > ego["x"]).all(), \
        "Some x_goals are behind ego position"
    print("PASS  all x_goals ahead of ego")

def test_goals_right_lane_bias():
    """Approximately 60% of goals should target the right lane."""
    ego, _ = get_state(0)
    goals = sample_goals(ego)
    right_lane_y = goals[:, 1].min()
    right_lane_count = (goals[:, 1] == right_lane_y).sum().item()
    fraction = right_lane_count / L
    assert fraction >= 0.5, \
        f"Right-lane goal fraction {fraction:.2f} < 0.5 (expected ~0.6)"
    print(f"PASS  right-lane goal bias ({fraction:.0%} on right lane)")

def test_goals_device():
    ego, _ = get_state(0)
    goals = sample_goals(ego)
    assert str(goals.device).startswith(DEVICE.split(":")[0]), \
        f"goals on wrong device {goals.device}"
    print("PASS  goals on correct device")

# ─────────────────────────────────────────────
# meta_cost.py
# ─────────────────────────────────────────────

from meta_cost import compute_meta_cost

def _make_vY(v_values=None, y_values=None):
    """Helper: build v (L,N) and y (N,L) from per-trajectory scalars."""
    v = torch.zeros(L, N, device=DEVICE)
    y = torch.zeros(N, L, device=DEVICE)
    if v_values is not None:
        for i, val in enumerate(v_values):
            v[i] = val
    if y_values is not None:
        for i, val in enumerate(y_values):
            y[:, i] = val
    return v, y

def test_meta_cost_shape():
    v, y = _make_vY()
    cost = compute_meta_cost(v, y)
    assert cost.shape == (L,), f"cost shape {cost.shape}, expected ({L},)"
    print("PASS  meta_cost output shape")

def test_meta_cost_cruise_selects_closest():
    """Trajectory closest to v_cruise should have lowest cost."""
    v_cruise = 10.0
    v_values = [5.0, 8.0, 10.0, 12.0, 15.0, 3.0, 10.5, 9.5]
    v, y = _make_vY(v_values=v_values)
    cost = compute_meta_cost(v, y, mode="cruise", v_cruise=v_cruise)
    best = cost.argmin().item()
    assert v_values[best] in (10.0, 10.5, 9.5), \
        f"Expected trajectory near v_cruise=10, got index {best} (v={v_values[best]})"
    print("PASS  cruise meta_cost selects trajectory closest to v_cruise")

def test_meta_cost_highway_tradeoff():
    """High speed + right lane: should prefer fast trajectory near y=0."""
    v, y = _make_vY(
        v_values=[VMAX]*L,
        y_values=[0.0, 4.0, 8.0, 2.0, 6.0, 1.0, 3.0, 10.0]
    )
    cost = compute_meta_cost(v, y, mode="highway", y_right=0.0, w1=1.0, w2=1.0)
    best = cost.argmin().item()
    assert best == 0, f"Expected trajectory at y=0 to win, got index {best}"
    print("PASS  highway meta_cost selects trajectory closest to right lane")

def test_meta_cost_nonnegative():
    v, y = _make_vY(v_values=[VMAX]*L)
    cost = compute_meta_cost(v, y, mode="cruise", v_cruise=VMAX)
    assert (cost >= 0).all(), "meta_cost returned negative values"
    print("PASS  meta_cost is non-negative")

if __name__ == "__main__":
    print("── data.py ──")
    test_get_state_keys()
    test_get_state_advances()
    test_predict_obstacles_shape()
    test_predict_obstacles_linear()

    print("\n── goals.py ──")
    test_goals_shape()
    test_goals_x_ahead()
    test_goals_right_lane_bias()
    test_goals_device()

    print("\n── meta_cost.py ──")
    test_meta_cost_shape()
    test_meta_cost_cruise_selects_closest()
    test_meta_cost_highway_tradeoff()
    test_meta_cost_nonnegative()

    print("\nAll support module tests passed.")
