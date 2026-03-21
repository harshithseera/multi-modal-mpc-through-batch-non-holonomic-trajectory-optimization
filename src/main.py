import torch
from basis import build_basis
from matrices import build_F, build_A
from optimizer import optimize_batch
from goals import sample_goals
from data import get_state, predict_obstacles
from meta_cost import compute_meta_cost
from config import N, NUM_OBS, DT, VMIN, VMAX

def main():

    P, P_dot, P_ddot = build_basis()

    Fmat = build_F(P, P_dot, P_ddot, num_obs=NUM_OBS)
    A    = build_A(P)

    # Ego state — updated each MPC step via receding horizon
    ego, neighbors = get_state(0)

    for t in range(100):

        goals        = sample_goals(ego)
        obs_x, obs_y = predict_obstacles(neighbors)

        cx, cy, cpsi, v = optimize_batch(
            P, P_dot, P_ddot, Fmat, A, goals, obs_x, obs_y, ego
        )

        y_pos = P @ cy.T                       # (N, L)
        cost  = compute_meta_cost(v, y_pos, mode="highway")
        best  = cost.argmin()

        # ── Receding horizon: apply first step of best trajectory ──────────

        best_cx   = cx[best]                   # (K,)
        best_cy   = cy[best]
        best_cpsi = cpsi[best]

        # Use N//4 index — far enough along the polynomial that forward motion
        # is established. P[1] is only t=1/29 into the horizon where the
        # polynomial may still be near or below the starting position.
        step_idx = N // 4
        x_next   = (P[step_idx]     @ best_cx).item()
        y_next   = (P[step_idx]     @ best_cy).item()
        psi_next = (P[step_idx]     @ best_cpsi).item()
        vx_next  = (P_dot[step_idx] @ best_cx).item() / DT
        vy_next  = (P_dot[step_idx] @ best_cy).item() / DT

        # Clamp velocity to physical limits
        speed = (vx_next**2 + vy_next**2) ** 0.5
        if speed > 0:
            scale    = min(max(speed, VMIN), VMAX) / speed
            vx_next *= scale
            vy_next *= scale

        # Update ego state for next MPC cycle
        ego = {
            "x":   x_next,
            "y":   y_next,
            "psi": psi_next,
            "vx":  vx_next,
            "vy":  vy_next,
        }

        # Advance data loader to next frame
        _, neighbors = get_state(t + 1)

        print(
            f"Step {t:3d} | best={best.item()} | "
            f"x={x_next:8.2f}  y={y_next:6.2f}  "
            f"vx={vx_next:5.2f}  cost={cost[best].item():.4f}"
        )

if __name__ == "__main__":
    main()