import torch
from config import N, K, DEVICE

def build_basis():
    t = torch.linspace(0, 1, N, device=DEVICE)

    P = torch.stack([t**i for i in range(K)], dim=1)

    # TODO: implement proper derivatives (Eq. 6)
    # P_dot  row i = i * t**(i-1), with row 0 = 0
    # P_ddot row i = i*(i-1) * t**(i-2), with rows 0,1 = 0
    P_dot = ...
    P_ddot = ...

    return P, P_dot, P_ddot