import torch
from config import N, K, DEVICE

def build_basis():
    t = torch.linspace(0, 1, N, device=DEVICE)

    # =====================================================
    # Polynomial basis (Eq. 6 in paper)
    # x(τ) = P(τ) @ c
    # =====================================================
    P = torch.stack([t**i for i in range(K)], dim=1)

    # =====================================================
    # Derivatives with respect to τ (normalized time)
    # IMPORTANT:
    # These are NOT scaled to real time.
    # Tests expect τ-derivatives, not physical-time derivatives.
    #
    # Real-time scaling is done inside optimizer:
    #   dx/dt = (dx/dτ) / T
    #   d²x/dt² = (d²x/dτ²) / T²
    # =====================================================

    # First derivative: d/dτ (t^i) = i * t^(i-1)
    P_dot = torch.stack([
        i * t**(i-1) if i > 0 else torch.zeros_like(t)
        for i in range(K)
    ], dim=1)

    # Second derivative: d²/dτ² (t^i) = i*(i-1)*t^(i-2)
    P_ddot = torch.stack([
        i * (i-1) * t**(i-2) if i > 1 else torch.zeros_like(t)
        for i in range(K)
    ], dim=1)

    return P, P_dot, P_ddot