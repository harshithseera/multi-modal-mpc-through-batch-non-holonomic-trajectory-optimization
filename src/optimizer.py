import torch
from config import L, K, N, RHO, VMIN, VMAX, MAX_ITERS, A_OBS, B_OBS, DEVICE
from constraints import collision_reformulation, acceleration_reformulation

def optimize_batch(P, P_dot, P_ddot, Fmat, A, goals, obs_x, obs_y):
    """
    A:     (4, 2*K)  boundary constraint matrix from build_A(P)
    goals: (L, 4)    each row is (x_goal, y_goal, psi_goal, v_goal)

    A and goals together define bl for the KKT system (Eq. 17):
        bl[l] = [x0, x_goal[l], y0, y_goal[l]]  shape (4,) per batch instance
    The full KKT solve (Eq. 18) requires stacking bl across the batch.
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
    # PRECOMPUTE (Eq. 18 insight)
    # ==========================================

    Q_single = P_ddot.T @ P_ddot                    # (K, K)
    Q        = torch.block_diag(Q_single, Q_single)  # (2K, 2K)

    # -----------------------------------------------------------------------
    # TODO: Full KKT solve (Eq. 17-18) — currently only the reduced system
    # is precomputed below. Boundary constraints (A, bl) are NOT yet enforced.
    #
    # When implemented, replace H and H_inv with the augmented KKT matrix:
    #
    #   n_bc  = A.shape[0]                              # 4
    #   # Q is already block_diag(Q_single, Q_single), shape (2K, 2K)
    #   Qxy   = torch.zeros(2*K + n_bc, 2*K + n_bc)
    #   Qxy[:2*K, :2*K] = Q + RHO * Fmat.T @ Fmat
    #   Qxy[:2*K, 2*K:] = A.T
    #   Qxy[2*K:, :2*K] = A
    #   Qxy_inv = torch.linalg.inv(Qxy)                # precompute once
    #
    # bl must also be constructed from ego state and goals before the loop:
    #   bl shape (4, L): rows = [x0, x_goal, y0, y_goal]
    #   bl[0] = ego_x0  (scalar, broadcast over L)
    #   bl[1] = goals[:, 0]   # x_goal per instance
    #   bl[2] = ego_y0
    #   bl[3] = goals[:, 1]   # y_goal per instance
    #
    # Inside the loop, rhs and sol then become:
    #   rhs = torch.cat([RHO * Fmat.T @ g + lam_xy.T, bl], dim=0)
    #   sol = Qxy_inv @ rhs
    #   cx  = sol[:K].T    cy = sol[K:2*K].T
    # -----------------------------------------------------------------------

    # Reduced (unconstrained) system — placeholder until KKT is implemented
    # TODO: replace with Qxy_inv above
    H     = Q + RHO * (Fmat.T @ Fmat)
    H_inv = torch.linalg.inv(H)          # TODO: use solve() instead of inverse

    # Precompute Qpsi_inv for heading update (Eq. 19/20) outside the loop
    Qpsi     = Q_single + RHO * (P.T @ P)
    Qpsi_inv = torch.linalg.inv(Qpsi)   # TODO: use solve() instead of inverse

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

        # TODO:
        # compute collision variables (Eq. 4)
        alpha, d = collision_reformulation(x, y, obs_x, obs_y)

        # TODO:
        # compute acceleration variables (Eq. 5)
        alpha_a, d_a = acceleration_reformulation(x_ddot, y_ddot)

        # --------------------------------------
        # Build g (Eq. 9 RHS)
        # --------------------------------------

        psi  = P @ cpsi.T    # (N, L)

        # TODO:
        # Stack g to match F block structure (Eq. 9) — all x-rows first, then all y-rows:
        # [ ξx + A_OBS * d * cos(alpha)  ]  ← collision x     (N*num_obs, L)
        # [ d_a * cos(alpha_a)           ]  ← acceleration x   (N, L)
        # [ v.T * cos(P @ cpsi.T)        ]  ← kinematics x     (N, L)
        # [ ξy + B_OBS * d * sin(alpha)  ]  ← collision y     (N*num_obs, L)
        # [ d_a * sin(alpha_a)           ]  ← acceleration y   (N, L)
        # [ v.T * sin(P @ cpsi.T)        ]  ← kinematics y     (N, L)
        # g shape: (F_rows, L)  — must match F row count exactly

        gx_col = obs_x + A_OBS * d   * torch.cos(alpha)
        gx_acc = d_a                 * torch.cos(alpha_a)
        gx_kin = v.T                 * torch.cos(psi)

        gy_col = obs_y + B_OBS * d   * torch.sin(alpha)
        gy_acc = d_a                 * torch.sin(alpha_a)
        gy_kin = v.T                 * torch.sin(psi)

        g = torch.cat([gx_col, gx_acc, gx_kin,
                       gy_col, gy_acc, gy_kin], dim=0)

        # --------------------------------------
        # Solve linear system (Eq. 18)
        # --------------------------------------

        rhs = RHO * (Fmat.T @ g) + lam_xy.T    # (2K, L)
        sol = H_inv @ rhs                      # (2K, L)

        cx = sol[:K].T
        cy = sol[K:].T

        # --------------------------------------
        # STEP (13): update heading (Eq. 19/20)
        # --------------------------------------

        # TODO:
        # psi_target = atan2(y_dot, x_dot)   shape: (N, L)
        psi_target = torch.atan2(y_dot, x_dot)

        # TODO:
        # Solve equality-constrained QP (Eq. 19/20)
        rhs_psi = RHO * P.T @ psi_target + lam_psi.T
        cpsi    = (Qpsi_inv @ rhs_psi).T

        # --------------------------------------
        # STEP (14): velocity (Eq. 21)
        # --------------------------------------

        v = torch.sqrt(x_dot**2 + y_dot**2).T
        v = torch.clamp(v, VMIN, VMAX)

        # --------------------------------------
        # STEP (23): lambda_xy (Eq. 23)
        # --------------------------------------

        xy_stack    = torch.cat([cx.T, cy.T], dim=0)
        residual_xy = Fmat @ xy_stack - g
        lam_xy      = lam_xy - (RHO * Fmat.T @ residual_xy).T

        # --------------------------------------
        # STEP (24): lambda_psi (Eq. 24)
        # --------------------------------------

        residual_psi = torch.atan2(y_dot, x_dot) - P @ cpsi.T
        lam_psi      = lam_psi - (RHO * P.T @ residual_psi).T

    return cx, cy, cpsi, v