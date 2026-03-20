"""
Tests for basis.py  —  run with: python src/tests/test_basis.py
All tests use torch.allclose with rtol=1e-4.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from config import N, K, DEVICE
from basis import build_basis

def test_shapes():
    P, P_dot, P_ddot = build_basis()
    assert P.shape      == (N, K), f"P shape {P.shape}"
    assert P_dot.shape  == (N, K), f"P_dot shape {P_dot.shape}"
    assert P_ddot.shape == (N, K), f"P_ddot shape {P_ddot.shape}"
    print("PASS  shapes")

def test_P_columns():
    """P[:,i] should equal t**i exactly."""
    P, _, _ = build_basis()
    t = torch.linspace(0, 1, N, device=DEVICE)
    for i in range(K):
        assert torch.allclose(P[:, i], t**i, rtol=1e-4), f"P column {i} wrong"
    print("PASS  P columns are t^i")

def test_P_dot_zero_column():
    """Derivative of t^0 = 1 is 0."""
    _, P_dot, _ = build_basis()
    assert torch.allclose(P_dot[:, 0], torch.zeros(N, device=DEVICE), atol=1e-6), \
        "P_dot[:,0] should be zero"
    print("PASS  P_dot[:,0] is zero")

def test_P_ddot_zero_columns():
    """Second derivative of t^0 and t^1 are both 0."""
    _, _, P_ddot = build_basis()
    assert torch.allclose(P_ddot[:, 0], torch.zeros(N, device=DEVICE), atol=1e-6)
    assert torch.allclose(P_ddot[:, 1], torch.zeros(N, device=DEVICE), atol=1e-6)
    print("PASS  P_ddot[:,0] and P_ddot[:,1] are zero")

def test_P_dot_numerical():
    """P_dot should match finite-difference derivative of P to within O(dt)."""
    P, P_dot, _ = build_basis()
    dt = 1.0 / (N - 1)
    fd = (P[1:] - P[:-1]) / dt          # (N-1, K) forward differences
    # compare interior rows only (skip boundary effects)
    assert torch.allclose(P_dot[1:-1], fd[:-1], atol=0.05), \
        "P_dot doesn't match finite difference"
    print("PASS  P_dot matches finite difference (atol=0.05)")

def test_P_ddot_numerical():
    """P_ddot should match second finite-difference of P."""
    P, _, P_ddot = build_basis()
    dt = 1.0 / (N - 1)
    fd2 = (P[2:] - 2*P[1:-1] + P[:-2]) / dt**2   # (N-2, K)
    assert torch.allclose(P_ddot[1:-1], fd2, atol=0.5), \
        "P_ddot doesn't match second finite difference"
    print("PASS  P_ddot matches second finite difference (atol=0.5)")

def test_device():
    P, P_dot, P_ddot = build_basis()
    for name, mat in [("P", P), ("P_dot", P_dot), ("P_ddot", P_ddot)]:
        assert str(mat.device).startswith(DEVICE.split(":")[0]), \
            f"{name} on wrong device {mat.device}"
    print("PASS  all tensors on correct device")

if __name__ == "__main__":
    test_shapes()
    test_P_columns()
    test_P_dot_zero_column()
    test_P_ddot_zero_columns()
    test_P_dot_numerical()
    test_P_ddot_numerical()
    test_device()
    print("\nAll basis tests passed.")
