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
import math

from config import (
    L, K, N,
    RHO_OBS,
    RHO_NONHOL,
    RHO_INEQ,
    VMIN,
    VMAX,
    MAX_ITERS,
    A_OBS,
    B_OBS,
    DEVICE,
    DT,
    WEIGHT_SMOOTHNESS,
    WEIGHT_SMOOTHNESS_PSI,
    AMAX,
    VCRUISE
)


def collision_reformulation(x, y, obs_x, obs_y):
    """
    Official obstacle reformulation.
    """

    num_obs = obs_x.shape[0] // N

    x_t = x.repeat(num_obs, 1)
    y_t = y.repeat(num_obs, 1)

    dx = x_t - obs_x
    dy = y_t - obs_y

    alpha = torch.atan2(A_OBS * dy, B_OBS * dx)

    cos_a = torch.cos(alpha)
    sin_a = torch.sin(alpha)

    c1 = (
        (A_OBS ** 2) * (cos_a ** 2)
        + (B_OBS ** 2) * (sin_a ** 2)
    )

    c2 = (
        A_OBS * dx * cos_a
        + B_OBS * dy * sin_a
    )

    d = torch.clamp(c2 / (c1 + 1e-6), min=1.0)

    return alpha, d


def acceleration_reformulation(xdd, ydd):

    alpha_a = torch.atan2(ydd, xdd)

    mag = torch.sqrt(xdd**2 + ydd**2 + 1e-6)

    d_a = torch.clamp(mag, max=AMAX)

    return alpha_a, d_a


def solve_boundary_conditions(ego, goals_r, T):

    n = K - 1

    cx = torch.zeros((L, K), device=DEVICE)
    cy = torch.zeros((L, K), device=DEVICE)

    cx[:, 0] = 0.0
    cy[:, 0] = 0.0

    cx[:, -1] = goals_r[:, 0]
    cy[:, -1] = goals_r[:, 1]

    cx[:, 1] = (ego["vx"] * T / n)
    cy[:, 1] = (ego["vy"] * T / n)

    for i in range(2, n):

        frac = i / n

        cx[:, i] = (
            cx[:, 1]
            + frac * (cx[:, -1] - cx[:, 1])
        )

        cy[:, i] = (
            cy[:, 1]
            + frac * (cy[:, -1] - cy[:, 1])
        )

    return cx, cy


def optimize_batch(
    P,
    P_dot,
    P_ddot,
    Fmat_unused,
    A_unused,
    goals,
    obs_x,
    obs_y,
    ego
):

    # ========================================================
    # Setup
    # ========================================================

    ox = float(ego["x"])
    oy = float(ego["y"])

    vx0 = float(ego["vx"])
    vy0 = float(ego["vy"])

    psi0 = float(ego["psi"])

    T = N  * DT

    # IMPORTANT:
    # official C++ already physicalizes using /l
    Pd = P_dot / T
    Pdd = P_ddot / (T**2)

    num_obs = obs_x.shape[0] // N

    Fo = P.repeat(num_obs, 1)

    # ========================================================
    # Relative coordinates
    # ========================================================

    goals_r = goals.clone()

    goals_r[:, 0] -= ox
    goals_r[:, 1] -= oy

    obs_xr = obs_x - ox
    obs_yr = obs_y - oy
    
    # ========================================================
    # Equality Constraints
    # ========================================================

    n_order = K - 1

    A_single = torch.zeros((3, K), device=DEVICE)

    A_single[0, 0] = 1.0

    A_single[1, 0] = -n_order / T
    A_single[1, 1] = n_order / T

    A_single[2, -1] = 1.0

    A_block = torch.block_diag(A_single, A_single)

    bl = torch.zeros((L, 6), device=DEVICE)

    bl[:, 1] = vx0
    bl[:, 4] = vy0

    bl[:, 2] = goals_r[:, 0]
    bl[:, 5] = goals_r[:, 1]

    # ========================================================
    # Initial Guess
    # ========================================================

    cx, cy = solve_boundary_conditions(
        ego,
        goals_r,
        T
    )

    cpsi = torch.zeros((L, K), device=DEVICE)
    cpsi[:, 0] = psi0

    # ========================================================
    # ADMM Variables
    # ========================================================

    lam_obs_x = torch.zeros((L, K), device=DEVICE)
    lam_obs_y = torch.zeros((L, K), device=DEVICE)

    lam_nonhol_x = torch.zeros((L, K), device=DEVICE)
    lam_nonhol_y = torch.zeros((L, K), device=DEVICE)

    lam_acc_x = torch.zeros((L, K), device=DEVICE)
    lam_acc_y = torch.zeros((L, K), device=DEVICE)

    lam_psi = torch.zeros((L, K), device=DEVICE)

    rho_obs = RHO_OBS
    rho_nonhol = RHO_NONHOL
    rho_ineq = RHO_INEQ
    rho_psi = 1.0

    rho_growth = 1.06

    # ========================================================
    # Smoothness Costs
    # ========================================================

    Qs = WEIGHT_SMOOTHNESS * (Pdd.T @ Pdd)

    Q = torch.block_diag(Qs, Qs)

    # ========================================================
    # ADMM LOOP
    # ========================================================

    for it in range(MAX_ITERS):

        # ----------------------------------------------------
        # Current trajectories
        # ----------------------------------------------------

        x = P @ cx.T
        y = P @ cy.T

        xd = Pd @ cx.T
        yd = Pd @ cy.T

        xdd = Pdd @ cx.T
        ydd = Pdd @ cy.T

        # ----------------------------------------------------
        # Velocity projection
        # ----------------------------------------------------

        v_profile = torch.sqrt(
            xd**2 + yd**2 + 1e-6
        ).T

        v_profile = torch.clamp(
            v_profile,
            VMIN,
            VMAX
        )

        # VERY IMPORTANT
        # official implementation anchors initial speed
        v_profile[:, 0] = vx0

        # ----------------------------------------------------
        # Heading update
        # ----------------------------------------------------

        psi_tgt = torch.atan2(yd, xd)

        # ----------------------------------------------------
        # Collision reformulation
        # ----------------------------------------------------

        alpha_obs, d_obs = collision_reformulation(
            x,
            y,
            obs_xr,
            obs_yr
        )

        # ----------------------------------------------------
        # Acceleration reformulation
        # ----------------------------------------------------

        alpha_a, d_a = acceleration_reformulation(
            xdd,
            ydd
        )

        # ----------------------------------------------------
        # Constraint targets
        # ----------------------------------------------------

        psi = P @ cpsi.T

        # NONHOLONOMIC TARGETS
        b_nonhol_x = v_profile.T * torch.cos(psi)
        b_nonhol_y = v_profile.T * torch.sin(psi)

        # OBSTACLE TARGETS
        b_obs_x = (
            obs_xr
            + A_OBS * d_obs * torch.cos(alpha_obs)
        )

        b_obs_y = (
            obs_yr
            + B_OBS * d_obs * torch.sin(alpha_obs)
        )

        # ACC TARGETS
        b_acc_x = d_a * torch.cos(alpha_a)
        b_acc_y = d_a * torch.sin(alpha_a)

        # ====================================================
        # Build weighted system
        # ====================================================

        F_obs = Fo
        F_nonhol = Pd
        F_acc = Pdd

        cost_x = (
            Qs
            + rho_nonhol * (F_nonhol.T @ F_nonhol)
            + rho_obs * (F_obs.T @ F_obs)
            + rho_ineq * (F_acc.T @ F_acc)
        )

        cost = torch.block_diag(cost_x, cost_x)

        # ====================================================
        # KKT SYSTEM
        # ====================================================

        KKT = torch.zeros(
            (2*K + 6, 2*K + 6),
            dtype=torch.float64,
            device=DEVICE
        )

        KKT[:2*K, :2*K] = cost.double()

        KKT[:2*K, 2*K:] = A_block.T.double()

        KKT[2*K:, :2*K] = A_block.double()

        KKT[2*K:, 2*K:] = (
            -1e-8 * torch.eye(
                6,
                dtype=torch.float64,
                device=DEVICE
            )
        )

        # ====================================================
        # Linear costs
        # ====================================================

        lincost_x = (
            -lam_nonhol_x
            - rho_nonhol * (
                F_nonhol.T @ b_nonhol_x
            ).T

            - rho_obs * (
                F_obs.T @ b_obs_x
            ).T

            - rho_ineq * (
                F_acc.T @ b_acc_x
            ).T
        )

        lincost_y = (
            -lam_nonhol_y
            - rho_nonhol * (
                F_nonhol.T @ b_nonhol_y
            ).T

            - rho_obs * (
                F_obs.T @ b_obs_y
            ).T

            - rho_ineq * (
                F_acc.T @ b_acc_y
            ).T
        )

        rhs_top = torch.cat([
            -lincost_x.T,
            -lincost_y.T
        ], dim=0)

        rhs = torch.cat([
            rhs_top,
            bl.T
        ], dim=0).double()

        sol = torch.linalg.solve(KKT, rhs)

        cx = sol[:K].T.float()
        cy = sol[K:2*K].T.float()

        # ====================================================
        # PSI UPDATE
        # ====================================================

        A_psi = torch.zeros((1, K), device=DEVICE)
        A_psi[0, 0] = 1.0

        b_psi = torch.full(
            (L, 1),
            psi0,
            device=DEVICE
        )

        KKT_p = torch.zeros(
            (K+1, K+1),
            dtype=torch.float64,
            device=DEVICE
        )

        KKT_p[:K, :K] = (
            WEIGHT_SMOOTHNESS_PSI * (Pdd.T @ Pdd)
            + rho_psi * (P.T @ P)
        ).double()

        KKT_p[:K, K:] = A_psi.T.double()

        KKT_p[K:, :K] = A_psi.double()

        rhs_p = torch.cat([
            (
                rho_psi * P.T @ psi_tgt
                + lam_psi.T
            ).double(),

            b_psi.T.double()

        ], dim=0)

        sol_psi = torch.linalg.solve(
            KKT_p,
            rhs_p
        )

        cpsi = sol_psi[:K].T.float()

        # ====================================================
        # Residuals
        # ====================================================

        x = P @ cx.T
        y = P @ cy.T

        xd = Pd @ cx.T
        yd = Pd @ cy.T

        xdd = Pdd @ cx.T
        ydd = Pdd @ cy.T

        psi = P @ cpsi.T

        res_nonhol_x = xd - b_nonhol_x
        res_nonhol_y = yd - b_nonhol_y

        x_rep = x.repeat(num_obs, 1)
        y_rep = y.repeat(num_obs, 1)

        res_obs_x = (
            x_rep
            - b_obs_x
        )

        res_obs_y = (
            y_rep
            - b_obs_y
        )

        res_acc_x = xdd - b_acc_x
        res_acc_y = ydd - b_acc_y

        # ====================================================
        # Dual Updates
        # ====================================================

        lam_nonhol_x = (
            lam_nonhol_x
            - rho_nonhol * (
                F_nonhol.T @ res_nonhol_x
            ).T
        )

        lam_nonhol_y = (
            lam_nonhol_y
            - rho_nonhol * (
                F_nonhol.T @ res_nonhol_y
            ).T
        )

        lam_obs_x = (
            lam_obs_x
            - rho_obs * (
                F_obs.T @ res_obs_x
            ).T
        )

        lam_obs_y = (
            lam_obs_y
            - rho_obs * (
                F_obs.T @ res_obs_y
            ).T
        )

        lam_acc_x = (
            lam_acc_x
            - rho_ineq * (
                F_acc.T @ res_acc_x
            ).T
        )

        lam_acc_y = (
            lam_acc_y
            - rho_ineq * (
                F_acc.T @ res_acc_y
            ).T
        )

        res_psi = (
            (P @ cpsi.T)
            - psi_tgt
        )

        lam_psi = (
            lam_psi
            - rho_psi * (
                P.T @ res_psi
            ).T
        )

        # ====================================================
        # Rho Updates
        # ====================================================

        rho_obs *= rho_growth
        rho_nonhol *= rho_growth
        rho_ineq *= rho_growth
        rho_psi *= rho_growth

    # ========================================================
    # Final residual capture
    # ========================================================

    res_obs = torch.cat([
        res_obs_x,
        res_obs_y
    ], dim=0)

    # ========================================================
    # Restore world coordinates
    # ========================================================

    cx += ox
    cy += oy

    return (
        cx,
        cy,
        cpsi,
        v_profile,
        res_obs
    )