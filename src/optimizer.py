"""
Batch non-holonomic trajectory optimizer.

Paper reference: Section III (Main Results), Algorithm 1.

Implements the alternating minimisation (split-Bregman / ADMM) solver
for the multi-convex reformulation of the non-holonomic trajectory
optimisation problem (Eq. 8a–8c).

The L problem instances (one per goal hypothesis) share the same F, Q,
and A matrices. Their solutions differ only through the right-hand side
vector g (which depends on the current auxiliary variables) and the
boundary vector bl (which encodes each instance's goal position). This
structure enables the batch update of Eq. (18) and (20).
"""

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
    Solve L goal-directed non-holonomic trajectory optimisation problems
    in parallel using the alternating minimisation scheme of Algorithm 1.

    All internal computation is performed in the ego-relative coordinate
    frame (origin at the ego vehicle's current position). World-frame
    coordinates are restored before returning.

    Parameters
    ----------
    P, P_dot, P_ddot : (N, K)
        Polynomial basis and τ-derivatives from build_basis().
    Fmat : (F_rows, 2K)
        Constraint matrix from build_F() — passed for API compatibility
        but rebuilt internally with real-time scaling.
    A : (4, 2K)
        Boundary constraint matrix from build_A().
    goals : (L, 4)
        World-frame goal hypotheses; each row is (x_goal, y_goal, ψ_goal, v_goal).
    obs_x, obs_y : (N * NUM_OBS, L)
        World-frame obstacle trajectory predictions from predict_obstacles().
    ego : dict
        Current ego state with keys 'x', 'y', 'vx'.

    Returns
    -------
    cx, cy : (L, K)
        World-frame polynomial coefficients. P @ cx.T gives world-frame x.
    cpsi : (L, K)
        Heading polynomial coefficients.
    v : (L, N)
        Velocity profile for each batch instance.
    """

    # ── Ego-relative coordinate shift ─────────────────────────────────
    # Shifting to ego origin keeps polynomial coefficients O(goal_dist)
    # rather than O(absolute_road_position), which prevents numerical
    # issues in the KKT solve for large NGSIM coordinate values.
    ox  = float(ego["x"])
    oy  = float(ego["y"])
    vx0 = max(float(ego.get("vx", 5.0)), VMIN)

    goals_r = goals.clone()
    goals_r[:, 0] -= ox
    goals_r[:, 1] -= oy

    obs_xr = obs_x - ox   # (N*num_obs, L)
    obs_yr = obs_y - oy

    # ── Real-time scaled basis (Section III-B, Eq. 6) ─────────────────
    # P_dot and P_ddot from build_basis() are τ-derivatives.
    # Scaling by T and T² converts them to physical-time derivatives:
    #   dx/dt = P_dot/T @ cx,   d²x/dt² = P_ddot/T² @ cx
    num_obs = obs_x.shape[0] // N
    Fo      = P.repeat(num_obs, 1)   # (N*num_obs, K); Eq. (7a)
    T       = (N - 1) * DT
    Pd      = P_dot  / T
    Pdd     = P_ddot / (T * T)

    # ── F matrix (Eq. 9) ──────────────────────────────────────────────
    # Built with real-time-scaled Pd and Pdd so that all constraint rows
    # operate in the same physical units (m, m/s, m/s²).
    z_Fo = torch.zeros_like(Fo)
    z_P  = torch.zeros_like(Pd)

    F = torch.cat([
        torch.cat([Fo,  z_Fo], dim=1),   # Fo @ cx = ξx + a·d·cos(α)    (Eq. 7a)
        torch.cat([Pdd, z_P],  dim=1),   # P̈ @ cx = da·cos(αa)          (Eq. 7b)
        torch.cat([Pd,  z_P],  dim=1),   # Ṗ @ cx = v·cos(Pψ)           (Eq. 7c)
        torch.cat([z_Fo, Fo],  dim=1),   # Fo @ cy = ξy + b·d·sin(α)    (Eq. 7a)
        torch.cat([z_P,  Pdd], dim=1),   # P̈ @ cy = da·sin(αa)          (Eq. 7b)
        torch.cat([z_P,  Pd],  dim=1),   # Ṗ @ cy = v·sin(Pψ)           (Eq. 7c)
    ], dim=0)   # (F_rows, 2K)

    n_obs_rows = Fo.shape[0]   # N * num_obs

    # Per-constraint-row penalty weights ρ (Section III-D)
    rho_vec = torch.cat([
        torch.full((n_obs_rows,), RHO_OBS,    device=DEVICE),
        torch.full((N,),          RHO_INEQ,   device=DEVICE),
        torch.full((N,),          RHO_NONHOL, device=DEVICE),
        torch.full((n_obs_rows,), RHO_OBS,    device=DEVICE),
        torch.full((N,),          RHO_INEQ,   device=DEVICE),
        torch.full((N,),          RHO_NONHOL, device=DEVICE),
    ])   # (F_rows,)

    F_w   = F * rho_vec.unsqueeze(1)   # diag(ρ) F; (F_rows, 2K)
    FtF_w = F.T @ F_w                  # F^T diag(ρ) F; (2K, 2K)

    # ── KKT matrix for [cx; cy] update (Eq. 17) ──────────────────────
    # The equality-constrained QP (Step 12, Eq. 12) reduces to the linear
    # system of Eq. (17). The left-hand side is constant across all L
    # instances and all ADMM iterations, so it is inverted once here.
    #
    # KKT structure:
    #   [ Q + F^T diag(ρ) F    A^T ] [ cx; cy ]   [ F^T diag(ρ) g + λ ]
    #   [ A                    0   ] [  μ     ] = [ bl                  ]
    Qs   = WEIGHT_SMOOTHNESS * (Pdd.T @ Pdd)   # (K, K); Eq. (8a)
    Q    = torch.block_diag(Qs, Qs)            # (2K, 2K)
    n_bc = A.shape[0]                          # 4

    Qxy = torch.zeros(2*K + n_bc, 2*K + n_bc, dtype=torch.float64, device=DEVICE)
    Qxy[:2*K, :2*K] = (Q + FtF_w).double()
    Qxy[:2*K, 2*K:] = A.double().T
    Qxy[2*K:, :2*K] = A.double()
    Qxy_inv = torch.linalg.inv(Qxy)   # precomputed in float64 for numerical stability

    # ── Boundary condition vector bl (Eq. 8b) ─────────────────────────
    # In ego-relative frame: start = (0, 0), end = goal_rel per instance.
    # Rows: [x(0), x(T), y(0), y(T)] matching the A matrix structure.
    bl = torch.zeros(n_bc, L, device=DEVICE)
    bl[0] = 0.0;            bl[2] = 0.0
    bl[1] = goals_r[:, 0];  bl[3] = goals_r[:, 1]

    # ── Heading KKT matrix (Eq. 19) ───────────────────────────────────
    # The heading update (Step 13) also reduces to a linear system.
    # Q_ψ = WEIGHT_SMOOTHNESS * P̈^T P̈ + ρ * P^T P  (Eq. 19, left-hand side)
    Qpsi_m   = (WEIGHT_SMOOTHNESS * Pdd.T @ Pdd + RHO_NONHOL * P.T @ P).double()
    Qpsi_inv = torch.linalg.inv(Qpsi_m)

    # ── Initialisation (Section IV) ───────────────────────────────────
    # Straight-line trajectory from (0,0) to each goal in relative frame.
    # With P(τ) = [1, τ, τ², ...], a linear ramp x(τ) = x_goal·τ
    # is encoded as cx = [0, x_goal, 0, 0, ...].
    cx = torch.zeros(L, K, device=DEVICE)
    cy = torch.zeros(L, K, device=DEVICE)
    for l in range(L):
        cx[l, 1] = float(goals_r[l, 0])
        cy[l, 1] = float(goals_r[l, 1])

    cpsi   = torch.zeros(L, K, device=DEVICE)
    v      = torch.full((L, N), vx0, device=DEVICE)
    lam_xy = torch.zeros(2*K, L, device=DEVICE)   # Lagrange multipliers λ (Eq. 10)

    # ── Algorithm 1: Alternating Minimisation ─────────────────────────
    for _ in range(MAX_ITERS):

        # Evaluate trajectory quantities in ego-relative frame
        x   = P   @ cx.T   # (N, L)  position
        y   = P   @ cy.T
        xd  = Pd  @ cx.T   # (N, L)  velocity
        yd  = Pd  @ cy.T
        xdd = Pdd @ cx.T   # (N, L)  acceleration
        ydd = Pdd @ cy.T

        # Steps 15–16: closed-form updates for auxiliary variables
        # α, d from Eq. (22) and Step 15; αa, da from Step 16
        alpha,   d   = collision_reformulation(x, y, obs_xr, obs_yr)
        alpha_a, d_a = acceleration_reformulation(xdd, ydd)
        psi          = P @ cpsi.T   # (N, L)

        # Build g (Eq. 9 RHS) — right-hand side of F @ [cx; cy] = g
        g = torch.cat([
            obs_xr + A_OBS * d * torch.cos(alpha),   # ξx + a·d·cos(α)  (Eq. 7a)
            d_a            * torch.cos(alpha_a),      # da·cos(αa)        (Eq. 7b)
            v.T            * torch.cos(psi),          # v·cos(Pψ)         (Eq. 7c)
            obs_yr + B_OBS * d * torch.sin(alpha),   # ξy + b·d·sin(α)  (Eq. 7a)
            d_a            * torch.sin(alpha_a),      # da·sin(αa)        (Eq. 7b)
            v.T            * torch.sin(psi),          # v·sin(Pψ)         (Eq. 7c)
        ], dim=0)   # (F_rows, L)

        # Step 12: solve KKT system for cx, cy (Eq. 18)
        # RHS top block = F^T diag(ρ) g + λ
        rhs_top = F_w.T @ g + lam_xy
        rhs     = torch.cat([rhs_top, bl], dim=0).double()
        sol     = Qxy_inv @ rhs
        cx      = sol[:K].T.float()
        cy      = sol[K:2*K].T.float()

        # Step 13: heading update — convex surrogate (Eq. 13, last line; Eq. 20)
        # The non-holonomic heading penalty is replaced by the convex surrogate
        # ‖atan2(ẏ, ẋ) - Pψ‖² for a given (ẋ, ẏ), giving a linear system.
        xd = Pd @ cx.T
        yd = Pd @ cy.T
        psi_tgt  = torch.atan2(yd, xd)                       # Eq. (13), last line
        rhs_psi  = (RHO_NONHOL * P.T @ psi_tgt).double()
        cpsi     = (Qpsi_inv @ rhs_psi).T.float()            # Eq. (20)

        # Step 14: velocity update — element-wise clip (Eq. 21)
        v = torch.clamp(torch.sqrt(xd**2 + yd**2).T, VMIN, VMAX)

        # Multiplier update — split-Bregman dual ascent (Eq. 23)
        cxy      = torch.cat([cx.T, cy.T], dim=0)   # (2K, L)
        residual = F @ cxy - g                       # constraint residual (F_rows, L)
        lam_xy   = lam_xy - F_w.T @ residual         # λ ← λ - F^T diag(ρ)(Fc - g)

    # ── Restore world-frame coordinates ───────────────────────────────
    # Adding ox to cx[:,0] is equivalent to shifting x(τ) = P @ cx.T
    # by ox for all τ, since P[:,0] = 1 always.
    cx[:, 0] += ox
    cy[:, 0] += oy

    return cx, cy, cpsi, v