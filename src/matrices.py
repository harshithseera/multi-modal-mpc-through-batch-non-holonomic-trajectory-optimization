"""
Matrix construction for the batch trajectory optimizer.

Paper reference:
  - F matrix: Section III-B (Matrix Representation), Eq. (9)
  - A matrix: Section III-C (Multi-Convex Reformulation), Eq. (8b)
  - Q matrix: cost function, Eq. (8a)
"""

import torch
from config import N, K, NUM_OBS, DEVICE, WEIGHT_SMOOTHNESS


def build_F(P, P_dot, P_ddot, num_obs):
    """
    Build the constraint matrix F (Eq. 9).

    F maps the stacked coefficient vector [cx; cy] to the trajectory
    quantities that appear in the non-convex constraints (Eq. 7a-7c):
        Fo @ cx  =  ξx + a·d·cos(α)      (collision x,  Eq. 7a)
        P̈  @ cx  =  da·cos(αa)           (acceleration x, Eq. 7b)
        Ṗ  @ cx  =  v·cos(Pψ)            (kinematics x,  Eq. 7c)
    and their y-counterparts.

    Row order follows Eq. (9):
        [Fo   0  ]   collision x
        [P̈   0  ]   acceleration x
        [Ṗ   0  ]   kinematics x
        [0    Fo ]   collision y
        [0    P̈ ]   acceleration y
        [0    Ṗ ]   kinematics y

    Note: P_dot and P_ddot passed here are raw τ-derivatives.
    Real-time scaling (÷T and ÷T²) is applied inside optimize_batch.

    Parameters
    ----------
    P, P_dot, P_ddot : (N, K)
        Basis matrix and its τ-derivatives from build_basis().
    num_obs : int
        Number of obstacles; Fo = P stacked num_obs times → (N*num_obs, K).

    Returns
    -------
    F : (N*num_obs*2 + N*2 + N*2, 2K)
    """
    Fo   = P.repeat(num_obs, 1)
    z_Fo = torch.zeros_like(Fo)
    z_P  = torch.zeros_like(P_dot)

    return torch.cat([
        torch.cat([Fo,     z_Fo],   dim=1),   # collision x  (Eq. 7a)
        torch.cat([P_ddot, z_P],    dim=1),   # acceleration x (Eq. 7b)
        torch.cat([P_dot,  z_P],    dim=1),   # kinematics x  (Eq. 7c)
        torch.cat([z_Fo,   Fo],     dim=1),   # collision y   (Eq. 7a)
        torch.cat([z_P,    P_ddot], dim=1),   # acceleration y (Eq. 7b)
        torch.cat([z_P,    P_dot],  dim=1),   # kinematics y  (Eq. 7c)
    ], dim=0)


def build_A(P, P_dot=None):
    """
    Build the boundary constraint matrix A (Eq. 8b).

    Enforces position boundary conditions on the trajectory:
        A @ [cx; cy] = bl
    where bl = [x(0), x(T), y(0), y(T)].

    Only position constraints are included (start and end positions).
    Shape: (4, 2K).

    Parameters
    ----------
    P : (N, K)
        Basis matrix from build_basis().
    P_dot : ignored
        Accepted for API compatibility; not used.

    Returns
    -------
    A : (4, 2K)
    """
    K_val   = P.shape[1]
    A_block = P[[0, -1], :]   # rows for τ=0 and τ=1
    A = torch.zeros(4, 2 * K_val, device=P.device)
    A[:2, :K_val] = A_block   # x(0) and x(T)
    A[2:, K_val:] = A_block   # y(0) and y(T)
    return A


def build_Q(P_ddot):
    """
    Build the smoothness cost matrix Q (Eq. 8a).

    The cost function minimises squared accelerations:
        (1/2) cx^T Q cx + (1/2) cy^T Q cy

    where Q = WEIGHT_SMOOTHNESS * P̈^T P̈.

    Returns
    -------
    Q_single : (K, K)
        Per-axis cost matrix.
    Q_block : (2K, 2K)
        Block-diagonal matrix for the joint [cx; cy] system.
    """
    Q_single = WEIGHT_SMOOTHNESS * (P_ddot.T @ P_ddot)
    return Q_single, torch.block_diag(Q_single, Q_single)