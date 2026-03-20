import torch
from config import N, K, DEVICE

def build_basis():
    t = torch.linspace(0, 1, N, device=DEVICE)

    P = torch.stack([t**i for i in range(K)], dim=1)

    # TODO: implement proper derivatives (Eq. 6) - done
    # P_dot  row i = i * t**(i-1), with row 0 = 0
    # P_ddot row i = i*(i-1) * t**(i-2), with rows 0,1 = 0
    P_dot = torch.stack([i * t**(i-1) if i > 0 else torch.zeros_like(t) for i in range(K)], dim=1)
    P_ddot = torch.stack([i * (i-1) * t**(i-2) if i > 1 else torch.zeros_like(t) for i in range(K)], dim=1)

    return P, P_dot, P_ddot