"""
Tests for meta_cost.py  —  run with: python src/tests/test_meta_cost.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from config import N, L, VMAX, DEVICE
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
    test_meta_cost_shape()
    test_meta_cost_cruise_selects_closest()
    test_meta_cost_highway_tradeoff()
    test_meta_cost_nonnegative()
    print("\nAll meta_cost tests passed.")
