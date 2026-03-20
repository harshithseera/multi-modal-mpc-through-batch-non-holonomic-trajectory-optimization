import torch
from config import N, K, NUM_OBS, DEVICE

def build_F(P, P_dot, P_ddot, num_obs):
    """
    Build full F matrix (Eq. 9)

    P:       (N, K)
    P_dot:   (N, K)
    P_ddot:  (N, K)

    Fo:      (N*num_obs, K)  -- P stacked num_obs times
    F shape: (N*num_obs*2 + N*2 + N*2, K*2)
    """

    # TODO:
    # 1. Fo = P stacked num_obs times along dim=0  -> (N*num_obs, K)
    # 2. Build block matrix (Eq. 9):
    #    [Fo        0    ]
    #    [0         Fo   ]
    #    [P_ddot    0    ]
    #    [0         P_ddot]
    #    [P_dot     0    ]
    #    [0         P_dot]

    F = ...
    return F


def build_A(P):
    """
    Boundary constraints for x and y (Eq. 8b): A @ [cx; cy] = bl

    Enforces x(0), x(T), y(0), y(T) via rows 0 and -1 of P.
    A shape: (4, 2*K)

    Note: heading boundary conditions ψ(0), ψ(T) are handled separately
    via the independent system A @ cpsi = bpsi (Eq. 19), not here.
    """

    # TODO:
    # A_block = P[[0, -1], :]        -> (2, K)
    # A = block_diag(A_block, A_block) -> (4, 2*K)

    A = ...
    return A