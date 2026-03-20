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

    # TODO: - done
    # 1. Fo = P stacked num_obs times along dim=0  -> (N*num_obs, K)
    # 2. Build block matrix (Eq. 9):
    #    [Fo        0    ]
    #    [0         Fo   ]
    #    [P_ddot    0    ]
    #    [0         P_ddot]
    #    [P_dot     0    ]
    #    [0         P_dot]

    Fo = P.repeat(num_obs, 1)
    zeros_Fo   = torch.zeros_like(Fo)
    zeros_P    = torch.zeros_like(P_dot)

    F = torch.cat([
        torch.cat([Fo,     zeros_Fo], dim=1),   # x collision
        torch.cat([P_ddot, zeros_P],  dim=1),   # x accel
        torch.cat([P_dot,  zeros_P],  dim=1),   # x kin
        torch.cat([zeros_Fo, Fo],     dim=1),   # y collision
        torch.cat([zeros_P,  P_ddot], dim=1),   # y accel
        torch.cat([zeros_P,  P_dot],  dim=1),   # y kin
    ], dim=0)

    return F


def build_A(P):
    """
    Boundary constraints for x and y (Eq. 8b): A @ [cx; cy] = bl

    Enforces x(0), x(T), y(0), y(T) via rows 0 and -1 of P.
    A shape: (4, 2*K)

    Note: heading boundary conditions ψ(0), ψ(T) are handled separately
    via the independent system A @ cpsi = bpsi (Eq. 19), not here.
    """

    # TODO: - done
    # A_block = P[[0, -1], :]        -> (2, K)
    # A = block_diag(A_block, A_block) -> (4, 2*K)

    # Get boundary rows (first and last)
    A_block = P[[0, -1], :]  # (2, K)

    # Create block diagonal matrix [A_block, 0; 0, A_block]
    K_val = P.shape[1]
    A = torch.zeros(4, 2*K_val, device=P.device)
    A[:2, :K_val] = A_block
    A[2:, K_val:] = A_block

    return A