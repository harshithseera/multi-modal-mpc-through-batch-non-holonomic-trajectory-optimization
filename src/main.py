"""
Receding-horizon MPC loop.

Paper reference: Section III (MPC formulation), Section IV (validation).

At each step t the MPC:
  1. Samples L goal hypotheses (Section III-F).
  2. Predicts obstacle trajectories (Section III, Assumption 3).
  3. Runs the batch optimizer to produce L locally optimal trajectories
     (Algorithm 1).
  4. Selects the best trajectory via meta-cost (Section III-F, Eq. 25/26),
     discarding trajectories with heading change > HEADING_LIMIT_DEG.
  5. Applies the first control step (receding horizon).
"""

import math
import torch
from basis import build_basis
from matrices import build_F, build_A
from optimizer import optimize_batch
from goals import sample_goals
from data import get_state, predict_obstacles
from meta_cost import compute_meta_cost
from config import NUM_OBS, DT, VMIN, VMAX, N, HEADING_LIMIT_DEG

T_START = 100   # NGSIM frame index to begin from (skip sparse early frames)


def main():
    P, P_dot, P_ddot = build_basis()
    Fmat = build_F(P, P_dot, P_ddot, num_obs=NUM_OBS)
    A    = build_A(P)   # (4, 2K) position-only boundary constraints (Eq. 8b)

    T_total = (N - 1) * DT   # physical planning horizon [s]
    vel_idx = N // 2          # mid-horizon index for stable velocity readout

    ego, neighbors = get_state(T_START)

    for t in range(200):

        # Step 1: goal sampling (Section III-F, Eq. 26)
        goals = sample_goals(ego, neighbors)

        # Step 2: constant-velocity obstacle prediction (Section III, Assumption 3)
        obs_x, obs_y = predict_obstacles(neighbors)

        # Step 3: batch trajectory optimisation (Algorithm 1)
        cx, cy, cpsi, v = optimize_batch(
            P, P_dot, P_ddot, Fmat, A, goals, obs_x, obs_y, ego
        )

        # Step 4: meta-cost ranking with heading filter
        y_pos = P @ cy.T   # (N, L) world-frame lateral positions
        cost  = compute_meta_cost(v, y_pos)

        # Discard trajectories with total heading change > HEADING_LIMIT_DEG.
        # Heading is derived from velocity direction (chain rule: Pd/T @ cx).
        xd_t      = (P_dot / T_total) @ cx.T            # (N, L) physical velocity
        yd_t      = (P_dot / T_total) @ cy.T
        psi_t     = torch.atan2(yd_t, xd_t)             # (N, L) heading angle
        delta_deg = torch.abs(psi_t[-1] - psi_t[0]) * 180.0 / torch.pi  # (L,)
        cost[delta_deg > HEADING_LIMIT_DEG] = float("inf")

        best = cost.argmin().item()

        # Step 5: receding-horizon state advance
        # Position: evaluate polynomial at τ = 1/(N-1) (one DT step ahead)
        # Velocity: evaluate τ-derivative at mid-horizon, scale by T
        bcx, bcy = cx[best], cy[best]
        x_next  = (P[1]           @ bcx).item()
        y_next  = (P[1]           @ bcy).item()
        vx_next = (P_dot[vel_idx] @ bcx).item() / T_total
        vy_next = (P_dot[vel_idx] @ bcy).item() / T_total

        speed = (vx_next**2 + vy_next**2) ** 0.5
        if speed > 1e-6:
            sc = min(max(speed, VMIN), VMAX) / speed
            vx_next *= sc
            vy_next *= sc

        ego = {
            "x":          x_next,
            "y":          y_next,
            "psi":        math.atan2(vy_next, vx_next) if speed > 0.01 else ego["psi"],
            "vx":         vx_next,
            "vy":         vy_next,
            "vehicle_id": ego.get("vehicle_id"),
        }

        _, neighbors = get_state(T_START + t + 1, ego_id=ego.get("vehicle_id"))

        print(f"Step {t:3d} | best={best} | delta_psi={delta_deg[best]:.1f}° | "
              f"x={x_next:.1f}  y={y_next:.2f}  vx={vx_next:.2f}")


if __name__ == "__main__":
    main()