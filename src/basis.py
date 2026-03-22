"""
Polynomial basis construction.

Paper reference: Section III-B, Eq. (6).

The trajectory variables x(t), y(t), ψ(t) are parameterised as:
    x(τ) = P(τ) @ cx
    y(τ) = P(τ) @ cy
    ψ(τ) = P(τ) @ cψ

where τ ∈ [0, 1] is normalised time and P is the K-column Vandermonde
matrix evaluated at N evenly-spaced knots.

Derivatives are with respect to τ (not physical time t).
Physical-time scaling is applied inside the optimizer:
    dx/dt  = (dx/dτ) / T
    d²x/dt² = (d²x/dτ²) / T²
where T = (N-1) * DT is the total planning horizon in seconds.
"""

import torch
from config import N, K, DEVICE


def build_basis():
    """
    Build the polynomial basis matrix and its first and second τ-derivatives.

    Returns
    -------
    P : (N, K)
        Basis matrix. Row i is [1, τ_i, τ_i², ..., τ_i^(K-1)].
    P_dot : (N, K)
        First τ-derivative of P. Row i is [0, 1, 2τ_i, ..., (K-1)τ_i^(K-2)].
    P_ddot : (N, K)
        Second τ-derivative of P. Row i is [0, 0, 2, 6τ_i, ..., (K-1)(K-2)τ_i^(K-3)].
    """
    t = torch.linspace(0, 1, N, device=DEVICE)

    # Eq. (6): P[:,i] = τ^i
    P = torch.stack([t**i for i in range(K)], dim=1)

    # First derivative: d/dτ (τ^i) = i * τ^(i-1), with i=0 term = 0
    P_dot = torch.stack([
        i * t**(i - 1) if i > 0 else torch.zeros_like(t)
        for i in range(K)
    ], dim=1)

    # Second derivative: d²/dτ² (τ^i) = i(i-1) * τ^(i-2), with i=0,1 terms = 0
    P_ddot = torch.stack([
        i * (i - 1) * t**(i - 2) if i > 1 else torch.zeros_like(t)
        for i in range(K)
    ], dim=1)

    return P, P_dot, P_ddot