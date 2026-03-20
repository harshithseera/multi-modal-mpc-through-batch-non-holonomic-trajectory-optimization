"""
Tests for matrices.py  —  run with: python src/tests/test_matrices.py
Depends on basis.py passing its tests first.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from config import N, K, NUM_OBS, DEVICE
from basis import build_basis
from matrices import build_F, build_A

P, P_dot, P_ddot = build_basis()
F_ROWS = N*NUM_OBS*2 + N*2 + N*2   # expected row count of F

def test_F_shape():
    F = build_F(P, P_dot, P_ddot, num_obs=NUM_OBS)
    assert F.shape == (F_ROWS, K*2), f"F shape {F.shape}, expected ({F_ROWS}, {K*2})"
    print("PASS  F shape")

def test_F_block_structure():
    """
    Verify the six blocks are placed correctly.
    Fo = P repeated num_obs times. Blocks are all x-rows then all y-rows.
    """
    F = build_F(P, P_dot, P_ddot, num_obs=NUM_OBS)
    Fo = P.repeat(NUM_OBS, 1)          # (N*NUM_OBS, K)
    half = F_ROWS // 2

    # x-block (left half of columns, top half of rows)
    assert torch.allclose(F[:N*NUM_OBS, :K],        Fo,     atol=1e-6), "Fo x-block wrong"
    assert torch.allclose(F[N*NUM_OBS:N*NUM_OBS+N, :K], P_ddot, atol=1e-6), "P_ddot x-block wrong"
    assert torch.allclose(F[N*NUM_OBS+N:half, :K],  P_dot,  atol=1e-6), "P_dot x-block wrong"

    # x-block right half should be zero
    assert torch.allclose(F[:half, K:], torch.zeros(half, K, device=DEVICE), atol=1e-6), \
        "x-block right half not zero"

    # y-block (right half of columns, bottom half of rows)
    assert torch.allclose(F[half:half+N*NUM_OBS, K:],        Fo,     atol=1e-6), "Fo y-block wrong"
    assert torch.allclose(F[half+N*NUM_OBS:half+N*NUM_OBS+N, K:], P_ddot, atol=1e-6), "P_ddot y-block wrong"
    assert torch.allclose(F[half+N*NUM_OBS+N:, K:],  P_dot,  atol=1e-6), "P_dot y-block wrong"

    # y-block left half should be zero
    assert torch.allclose(F[half:, :K], torch.zeros(half, K, device=DEVICE), atol=1e-6), \
        "y-block left half not zero"

    print("PASS  F block structure")

def test_F_num_obs_scaling():
    """F row count should scale linearly with num_obs."""
    F2 = build_F(P, P_dot, P_ddot, num_obs=2)
    F4 = build_F(P, P_dot, P_ddot, num_obs=4)
    rows2 = N*2*2 + N*2 + N*2
    rows4 = N*4*2 + N*2 + N*2
    assert F2.shape[0] == rows2, f"F num_obs=2 rows {F2.shape[0]} != {rows2}"
    assert F4.shape[0] == rows4, f"F num_obs=4 rows {F4.shape[0]} != {rows4}"
    print("PASS  F scales with num_obs")

def test_A_shape():
    A = build_A(P)
    assert A.shape == (4, 2*K), f"A shape {A.shape}, expected (4, {2*K})"
    print("PASS  A shape")

def test_A_block_structure():
    """A should be block-diagonal: top-left = P[[0,-1],:], bottom-right = P[[0,-1],:]."""
    A = build_A(P)
    A_block = P[[0, -1], :]     # (2, K)
    assert torch.allclose(A[:2, :K],  A_block, atol=1e-6), "A top-left block wrong"
    assert torch.allclose(A[2:, K:],  A_block, atol=1e-6), "A bottom-right block wrong"
    assert torch.allclose(A[:2, K:],  torch.zeros(2, K, device=DEVICE), atol=1e-6), \
        "A top-right should be zero"
    assert torch.allclose(A[2:, :K],  torch.zeros(2, K, device=DEVICE), atol=1e-6), \
        "A bottom-left should be zero"
    print("PASS  A block structure")

def test_A_boundary_enforcement():
    """A @ [cx; cy] = bl should recover start/end positions."""
    A = build_A(P)
    # construct cx, cy whose trajectory starts at x=1, ends at x=5
    # simplest: use degree-1 polynomial cx = [1, 4, 0, ...] so x(0)=1, x(1)=5
    cx = torch.zeros(K, device=DEVICE); cx[0] = 1.0; cx[1] = 4.0
    cy = torch.zeros(K, device=DEVICE); cy[0] = 2.0; cy[1] = 1.0
    xy = torch.cat([cx, cy])           # (2K,)
    bl = A @ xy                        # (4,)
    assert torch.allclose(bl[0], torch.tensor(1.0), atol=1e-5), f"x(0)={bl[0]:.4f} != 1.0"
    assert torch.allclose(bl[1], torch.tensor(5.0), atol=1e-5), f"x(T)={bl[1]:.4f} != 5.0"
    assert torch.allclose(bl[2], torch.tensor(2.0), atol=1e-5), f"y(0)={bl[2]:.4f} != 2.0"
    assert torch.allclose(bl[3], torch.tensor(3.0), atol=1e-5), f"y(T)={bl[3]:.4f} != 3.0"
    print("PASS  A enforces boundary positions")

if __name__ == "__main__":
    test_F_shape()
    test_F_block_structure()
    test_F_num_obs_scaling()
    test_A_shape()
    test_A_block_structure()
    test_A_boundary_enforcement()
    print("\nAll matrices tests passed.")
