from basis import build_basis
from matrices import build_F, build_A
from optimizer import optimize_batch
from goals import sample_goals
from data import get_state, predict_obstacles
from meta_cost import compute_meta_cost
from config import NUM_OBS

def main():

    P, P_dot, P_ddot = build_basis()

    Fmat = build_F(P, P_dot, P_ddot, num_obs=NUM_OBS)
    A = build_A(P)

    for t in range(100):

        ego, neighbors = get_state(t)

        goals = sample_goals(ego)

        obs_x, obs_y = predict_obstacles(neighbors)

        cx, cy, cpsi, v = optimize_batch(
            P, P_dot, P_ddot, Fmat, A, goals, obs_x, obs_y
        )

        y_pos = P @ cy.T          # (N, L) lateral positions, as meta_cost expects
        cost = compute_meta_cost(v, y_pos)

        best = cost.argmin()

        # TODO (receding horizon MPC):
        # - extract first control point from best trajectory (cx[best], cy[best])
        # - apply to vehicle dynamics to advance ego state by one timestep DT
        # - update ego state for next call to get_state / sample_goals
        # - optionally warm-start next optimize_batch with shifted cx[best], cy[best]

        print(f"Step {t}: best trajectory = {best}")

if __name__ == "__main__":
    main()