import torch
from config import (
    L, K, N,
    RHO_OBS, RHO_NONHOL, RHO_INEQ,
    VMIN, VMAX, MAX_ITERS,
    A_OBS, B_OBS, DEVICE, DT,
    WEIGHT_SMOOTHNESS,
)
from constraints import collision_reformulation, acceleration_reformulation


def optimize_batch(P, P_dot, P_ddot, Fmat, A, goals, obs_x, obs_y, ego):
    """
    Batch non-holonomic trajectory optimizer (Eq. 17-24, Algorithm 1).

    IMPORTANT: all computation is done in ego-relative coordinates.
    obs_x, obs_y and goals arrive in world frame — they are shifted here.
    Returned cx, cy are world-frame (ox/oy baked into cx[:,0], cy[:,0]).
    """

    # ------------------------------------------------------------------
    # 1. Ego-relative shift — MUST happen before anything else
    #    Without this, collision_reformulation sees dx = ~0 - 150 = -150
    #    so d >> 1 always and the collision constraint is never active.
    # ------------------------------------------------------------------
    ox  = float(ego["x"])
    oy  = float(ego["y"])
    vx0 = max(float(ego.get("vx", 5.0)), VMIN)

    goals_r  = goals.clone()
    goals_r[:, 0] -= ox
    goals_r[:, 1] -= oy

    obs_xr = obs_x - ox   # (N*num_obs, L)
    obs_yr = obs_y - oy

    # ------------------------------------------------------------------
    # 2. Real-time scaled basis
    # ------------------------------------------------------------------
    num_obs = obs_x.shape[0] // N
    Fo      = P.repeat(num_obs, 1)    # (N*num_obs, K)
    T       = (N - 1) * DT
    Pd      = P_dot  / T
    Pdd     = P_ddot / (T * T)

    # ------------------------------------------------------------------
    # 3. F matrix — row order: obs_x, acc_x, kin_x, obs_y, acc_y, kin_y
    # ------------------------------------------------------------------
    z_Fo = torch.zeros_like(Fo)
    z_P  = torch.zeros_like(Pd)

    F = torch.cat([
        torch.cat([Fo,  z_Fo], dim=1),   # obs x
        torch.cat([Pdd, z_P],  dim=1),   # acc x
        torch.cat([Pd,  z_P],  dim=1),   # kin x
        torch.cat([z_Fo, Fo],  dim=1),   # obs y
        torch.cat([z_P,  Pdd], dim=1),   # acc y
        torch.cat([z_P,  Pd],  dim=1),   # kin y
    ], dim=0)   # (F_rows, 2K)

    n_obs_r = Fo.shape[0]   # N * num_obs

    # Per-row rho
    rho_vec = torch.cat([
        torch.full((n_obs_r,), RHO_OBS,    device=DEVICE),
        torch.full((N,),       RHO_INEQ,   device=DEVICE),
        torch.full((N,),       RHO_NONHOL, device=DEVICE),
        torch.full((n_obs_r,), RHO_OBS,    device=DEVICE),
        torch.full((N,),       RHO_INEQ,   device=DEVICE),
        torch.full((N,),       RHO_NONHOL, device=DEVICE),
    ])   # (F_rows,)

    F_w   = F * rho_vec.unsqueeze(1)   # (F_rows, 2K)
    FtF_w = F.T @ F_w                  # (2K, 2K)

    # ------------------------------------------------------------------
    # 4. KKT matrix — precomputed once (constant across iterations)
    # ------------------------------------------------------------------
    Qs    = WEIGHT_SMOOTHNESS * (Pdd.T @ Pdd)
    Q     = torch.block_diag(Qs, Qs)
    n_bc  = A.shape[0]   # 4

    Qxy = torch.zeros(2*K + n_bc, 2*K + n_bc, dtype=torch.float64, device=DEVICE)
    Qxy[:2*K, :2*K] = (Q + FtF_w).double()
    Qxy[:2*K, 2*K:] = A.double().T
    Qxy[2*K:, :2*K] = A.double()
    Qxy_inv = torch.linalg.inv(Qxy)   # float64, once

    # ------------------------------------------------------------------
    # 5. Boundary conditions — relative frame: start=(0,0), end=goal_rel
    # ------------------------------------------------------------------
    bl = torch.zeros(n_bc, L, device=DEVICE)
    bl[0] = 0.0;             bl[2] = 0.0
    bl[1] = goals_r[:, 0];   bl[3] = goals_r[:, 1]

    # ------------------------------------------------------------------
    # 6. Heading KKT — precomputed once
    # ------------------------------------------------------------------
    Qpsi_m   = (WEIGHT_SMOOTHNESS * Pdd.T @ Pdd + RHO_NONHOL * P.T @ P).double()
    Qpsi_inv = torch.linalg.inv(Qpsi_m)

    # ------------------------------------------------------------------
    # 7. Initialise: straight line from (0,0) to each goal in rel frame
    # ------------------------------------------------------------------
    cx = torch.zeros(L, K, device=DEVICE)
    cy = torch.zeros(L, K, device=DEVICE)
    for l in range(L):
        cx[l, 1] = float(goals_r[l, 0])
        cy[l, 1] = float(goals_r[l, 1])

    cpsi   = torch.zeros(L, K, device=DEVICE)
    v      = torch.full((L, N), vx0, device=DEVICE)
    lam_xy = torch.zeros(2*K, L, device=DEVICE)

    # ------------------------------------------------------------------
    # 8. ADMM iterations
    # ------------------------------------------------------------------
    for _ in range(MAX_ITERS):

        # trajectory in relative frame
        x   = P   @ cx.T   # (N, L)
        y   = P   @ cy.T
        xd  = Pd  @ cx.T
        yd  = Pd  @ cy.T
        xdd = Pdd @ cx.T
        ydd = Pdd @ cy.T

        # auxiliary proximal steps — use RELATIVE obs
        alpha,   d   = collision_reformulation(x, y, obs_xr, obs_yr)
        alpha_a, d_a = acceleration_reformulation(xdd, ydd)
        psi          = P @ cpsi.T

        # g uses RELATIVE obs positions
        g = torch.cat([
            obs_xr + A_OBS * d * torch.cos(alpha),
            d_a             * torch.cos(alpha_a),
            v.T             * torch.cos(psi),
            obs_yr + B_OBS * d * torch.sin(alpha),
            d_a             * torch.sin(alpha_a),
            v.T             * torch.sin(psi),
        ], dim=0)   # (F_rows, L)

        # KKT solve (Eq. 18)
        rhs_top = F_w.T @ g + lam_xy
        rhs     = torch.cat([rhs_top, bl], dim=0).double()
        sol     = Qxy_inv @ rhs
        cx      = sol[:K].T.float()
        cy      = sol[K:2*K].T.float()

        # heading update
        xd = Pd @ cx.T
        yd = Pd @ cy.T
        psi_tgt  = torch.atan2(yd, xd)
        rhs_psi  = (RHO_NONHOL * P.T @ psi_tgt).double()
        cpsi     = (Qpsi_inv @ rhs_psi).T.float()

        # velocity update
        v = torch.clamp(torch.sqrt(xd**2 + yd**2).T, VMIN, VMAX)

        # dual update (Eq. 23): lam -= F_w^T @ (F*c - g)
        cxy      = torch.cat([cx.T, cy.T], dim=0)
        residual = F @ cxy - g
        lam_xy   = lam_xy - F_w.T @ residual

    # ------------------------------------------------------------------
    # 9. Shift back to world frame
    # ------------------------------------------------------------------
    cx[:, 0] += ox
    cy[:, 0] += oy

    return cx, cy, cpsi, v