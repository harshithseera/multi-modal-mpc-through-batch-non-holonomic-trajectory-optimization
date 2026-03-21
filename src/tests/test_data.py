"""
Tests for data.py  —  run with: python src/tests/test_data.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from config import N, L, NUM_OBS, DEVICE
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
    """Calling get_state at well-separated t values should return different positions."""
    ego0, _ = get_state(0)
    ego50, _ = get_state(50)
    changed = any(ego0[k] != ego50[k] for k in ("x", "y"))
    assert changed, "get_state returned identical ego at t=0 and t=50"
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

if __name__ == "__main__":
    test_get_state_keys()
    test_get_state_advances()
    test_predict_obstacles_shape()
    test_predict_obstacles_linear()
    print("\nAll data tests passed.")