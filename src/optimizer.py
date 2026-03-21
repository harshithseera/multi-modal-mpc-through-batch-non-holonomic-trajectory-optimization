import torch
from config import L, K, N, RHO, VMIN, VMAX, MAX_ITERS, A_OBS, B_OBS, DEVICE
from constraints import collision_reformulation, acceleration_reformulation

def optimize_batch(P, P_dot, P_ddot, Fmat, A, goals, obs_x, obs_y, ego):
    """
    A:     (4, 2*K)  boundary constraint matrix from build_A(P)
    goals: (L, 4)    each row is (x_goal, y_goal, psi_goal, v_goal)
    ego:   dict with keys x, y — used to construct bl for KKT boundary constraints
    """

    # ==========================================
    # INIT (Section IV)
    # ==========================================

    cx   = torch.zeros(L, K, device=DEVICE)
    cy   = torch.zeros(L, K, device=DEVICE)
    cpsi = torch.zeros(L, K, device=DEVICE)

    # v shape is (L, N) to match meta_cost expected input (L, N)
    # Note: x_dot = P_dot @ cx.T is (N, L), so v = sqrt(...).T gives (L, N)
    v = torch.ones(L, N, device=DEVICE) * 5.0

    lam_xy  = torch.zeros(L, 2*K, device=DEVICE)
    lam_psi = torch.zeros(L,   K, device=DEVICE)

    # ==========================================
    # PRECOMPUTE (Eq. 17-18)
    # ==========================================

    Q_single = P_ddot.T @ P_ddot                     # (K, K)
    Q        = torch.block_diag(Q_single, Q_single)   # (2K, 2K)

    # Full KKT matrix (Eq. 17) — precomputed once, inverted once
    n_bc = A.shape[0]                                  # 4
    Qxy  = torch.zeros(2*K + n_bc, 2*K + n_bc, device=DEVICE)
    Qxy[:2*K, :2*K] = Q + RHO * (Fmat.T @ Fmat)
    Qxy[:2*K, 2*K:] = A.T
    Qxy[2*K:, :2*K] = A
    Qxy_inv = torch.linalg.inv(Qxy)                   # (2K+4, 2K+4)

    # bl shape (4, L): boundary values per batch instance
    # rows = [x(0), x(T), y(0), y(T)]
    bl      = torch.zeros(n_bc, L, device=DEVICE)
    bl[0]   = ego["x"]           # x start — same for all L
    bl[1]   = goals[:, 0]        # x goal  — differs per instance
    bl[2]   = ego["y"]           # y start — same for all L
    bl[3]   = goals[:, 1]        # y goal  — differs per instance

    # Precompute Qpsi_inv for heading update (Eq. 19/20)
    Qpsi     = Q_single + RHO * (P.T @ P)
    Qpsi_inv = torch.linalg.inv(Qpsi)                 # (K, K)

    # infer num_obs from obs_x shape
    num_obs = obs_x.shape[0] // N

    # ==========================================
    # MAIN LOOP (Algorithm 1)
    # ==========================================

    for _ in range(MAX_ITERS):

        # --------------------------------------
        # STEP (12): update cx, cy
        # --------------------------------------

        x      = P      @ cx.T     # (N, L)
        y      = P      @ cy.T
        x_dot  = P_dot  @ cx.T     # (N, L)
        y_dot  = P_dot  @ cy.T
        x_ddot = P_ddot @ cx.T     # (N, L)
        y_ddot = P_ddot @ cy.T

        # collision variables (Eq. 4)
        alpha, d = collision_reformulation(x, y, obs_x, obs_y)

        # acceleration variables (Eq. 5)
        alpha_a, d_a = acceleration_reformulation(x_ddot, y_ddot)

        # --------------------------------------
        # Build g (Eq. 9 RHS)
        # all x-rows first, then all y-rows — must match F row order exactly
        # --------------------------------------

        psi = P @ cpsi.T    # (N, L)

        gx_col = obs_x + A_OBS * d * torch.cos(alpha)   # (N*num_obs, L)
        gx_acc = d_a           * torch.cos(alpha_a)      # (N, L)
        gx_kin = v.T           * torch.cos(psi)          # (N, L)
        gy_col = obs_y + B_OBS * d * torch.sin(alpha)
        gy_acc = d_a           * torch.sin(alpha_a)
        gy_kin = v.T           * torch.sin(psi)

        g = torch.cat([gx_col, gx_acc, gx_kin,
                       gy_col, gy_acc, gy_kin], dim=0)   # (F_rows, L)

        # --------------------------------------
        # Solve full KKT system (Eq. 18)
        # --------------------------------------

        rhs = torch.cat([RHO * Fmat.T @ g + lam_xy.T, bl], dim=0)  # (2K+4, L)
        sol = Qxy_inv @ rhs                                           # (2K+4, L)

        cx = sol[:K].T      # (L, K)
        cy = sol[K:2*K].T

        # --------------------------------------
        # STEP (13): update heading (Eq. 19/20)
        # --------------------------------------

        psi_target = torch.atan2(y_dot, x_dot)              # (N, L)
        rhs_psi    = RHO * P.T @ psi_target + lam_psi.T     # (K, L)
        cpsi       = (Qpsi_inv @ rhs_psi).T                 # (L, K)

        # --------------------------------------
        # STEP (14): velocity (Eq. 21)
        # --------------------------------------

        # x_dot, y_dot are (N, L); take .T to get v as (L, N) for meta_cost
        v = torch.sqrt(x_dot**2 + y_dot**2).T
        v = torch.clamp(v, VMIN, VMAX)

        # --------------------------------------
        # STEP (23): lambda_xy (Eq. 23)
        # --------------------------------------

        xy_stack    = torch.cat([cx.T, cy.T], dim=0)            # (2K, L)
        residual_xy = Fmat @ xy_stack - g                        # (F_rows, L)
        lam_xy      = lam_xy - (RHO * Fmat.T @ residual_xy).T   # (L, 2K)

        # --------------------------------------
        # STEP (24): lambda_psi (Eq. 24)
        # --------------------------------------

        residual_psi = torch.atan2(y_dot, x_dot) - P @ cpsi.T   # (N, L)
        lam_psi      = lam_psi - (RHO * P.T @ residual_psi).T   # (L, K)

    return cx, cy, cpsi, v