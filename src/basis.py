import torch
from scipy.special import comb
from config import N, K, DEVICE

def build_basis():
    t = torch.linspace(0, 1, N, device=DEVICE)
    n = K - 1
    P = torch.zeros((N, K), device=DEVICE)
    P_dot = torch.zeros((N, K), device=DEVICE)
    P_ddot = torch.zeros((N, K), device=DEVICE)

    for i in range(K):
        # Bernstein Basis Function: B_{i,n}(t) = comb(n, i) * t^i * (1-t)^{n-i}
        coeff = comb(n, i)
        P[:, i] = coeff * (t**i) * ((1 - t)**(n - i))

        # First Derivative (Manual derivation for speed/accuracy)
        # d/dt B_{i,n}(t) = n * (B_{i-1, n-1}(t) - B_{i, n-1}(t))
        if i > 0:
            P_dot[:, i] += n * comb(n-1, i-1) * (t**(i-1)) * ((1-t)**(n-i))
        if i < n:
            P_dot[:, i] -= n * comb(n-1, i) * (t**i) * ((1-t)**(n-1-i))

        # Second Derivative
        # d2/dt2 B_{i,n}(t) = n(n-1) * (B_{i-2, n-2} - 2B_{i-1, n-2} + B_{i, n-2})
        if i > 1:
            P_ddot[:, i] += n*(n-1) * comb(n-2, i-2) * (t**(i-2)) * ((1-t)**(n-i))
        if i > 0 and i < n:
            P_ddot[:, i] -= 2 * n*(n-1) * comb(n-2, i-1) * (t**(i-1)) * ((1-t)**(n-i-1))
        if i < n-1:
            P_ddot[:, i] += n*(n-1) * comb(n-2, i) * (t**i) * ((1-t)**(n-i-2))

    return P, P_dot, P_ddot