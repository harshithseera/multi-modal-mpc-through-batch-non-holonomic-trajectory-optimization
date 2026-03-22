import torch
from config import N, K, NUM_OBS, DEVICE, WEIGHT_SMOOTHNESS


def build_F(P, P_dot, P_ddot, num_obs):
    """
    Build F matrix (Eq. 9) using RAW τ-derivatives (not real-time scaled).
    Real-time scaling is done inside the optimizer.
    Row order: obs_x, accel_x, kin_x, obs_y, accel_y, kin_y
    """
    Fo      = P.repeat(num_obs, 1)
    z_Fo    = torch.zeros_like(Fo)
    z_P     = torch.zeros_like(P_dot)

    return torch.cat([
        torch.cat([Fo,     z_Fo], dim=1),
        torch.cat([P_ddot, z_P],  dim=1),
        torch.cat([P_dot,  z_P],  dim=1),
        torch.cat([z_Fo,   Fo],   dim=1),
        torch.cat([z_P,    P_ddot], dim=1),
        torch.cat([z_P,    P_dot],  dim=1),
    ], dim=0)


def build_A(P):
    """
    Position-only boundary constraints: A @ [cx; cy] = bl
    Enforces x(0), x(T), y(0), y(T).
    Shape: (4, 2K)
    """
    K_val   = P.shape[1]
    A_block = P[[0, -1], :]
    A = torch.zeros(4, 2 * K_val, device=P.device)
    A[:2, :K_val] = A_block
    A[2:, K_val:] = A_block
    return A


def build_Q(P_ddot):
    Q_s = WEIGHT_SMOOTHNESS * (P_ddot.T @ P_ddot)
    return Q_s, torch.block_diag(Q_s, Q_s)