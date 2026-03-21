"""
Tests for goals.py  —  run with: python src/tests/test_goals.py
Requires data.py passing its tests first.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from config import L, DEVICE
from data import get_state
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

if __name__ == "__main__":
    test_goals_shape()
    test_goals_x_ahead()
    test_goals_right_lane_bias()
    test_goals_device()
    print("\nAll goals tests passed.")
