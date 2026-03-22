import torch
from basis import build_basis
from matrices import build_F, build_A
from optimizer import optimize_batch
from goals import sample_goals
from data import get_state, predict_obstacles
from meta_cost import compute_meta_cost
from config import NUM_OBS, DT, VMIN, VMAX, N


def main():
    P, P_dot, P_ddot = build_basis()
    Fmat = build_F(P, P_dot, P_ddot, num_obs=NUM_OBS)
    A    = build_A(P)   # 4-row position-only

    T_total = (N - 1) * DT
    vel_idx = N // 2

    ego, neighbors = get_state(0)

    for t in range(100):
        goals        = sample_goals(ego)
        obs_x, obs_y = predict_obstacles(neighbors)

        cx, cy, cpsi, v = optimize_batch(
            P, P_dot, P_ddot, Fmat, A, goals, obs_x, obs_y, ego
        )

        y_pos = P @ cy.T
        best  = compute_meta_cost(v, y_pos).argmin().item()

        best_cx = cx[best]
        best_cy = cy[best]

        x_next  = (P[1]           @ best_cx).item()
        y_next  = (P[1]           @ best_cy).item()
        vx_next = (P_dot[vel_idx] @ best_cx).item() / T_total
        vy_next = (P_dot[vel_idx] @ best_cy).item() / T_total

        speed = (vx_next**2 + vy_next**2) ** 0.5
        if speed > 1e-6:
            sc = min(max(speed, VMIN), VMAX) / speed
            vx_next *= sc; vy_next *= sc

        ego = {
            "x": x_next, "y": y_next,
            "psi": float(torch.atan2(torch.tensor(vy_next), torch.tensor(vx_next))),
            "vx": vx_next, "vy": vy_next,
            "vehicle_id": ego.get("vehicle_id"),
        }
        _, neighbors = get_state(t + 1, ego_id=ego.get("vehicle_id"))
        print(f"Step {t:3d} | best={best} | x={x_next:.1f} y={y_next:.2f} vx={vx_next:.2f}")


if __name__ == "__main__":
    main()