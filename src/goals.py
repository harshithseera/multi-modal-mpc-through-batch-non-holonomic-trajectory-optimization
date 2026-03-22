import torch
from config import L, N, DT, VMAX, DEVICE

LANE_CENTRES = torch.tensor([1.5, 5.5, 9.5, 13.5, 17.5], device=DEVICE)


def sample_goals(state):
    x0 = state["x"]
    y0 = state["y"]
    v0 = max(float(state["vx"]), 1.0)

    goal_dist    = v0 * N * DT
    dists        = torch.abs(LANE_CENTRES - y0)
    ego_lane_idx = int(torch.argmin(dists).item())

    n_right  = round(0.6 * L)
    n_spread = L - n_right

    x_right = x0 + goal_dist * torch.linspace(0.8, 1.2, n_right, device=DEVICE)
    y_right = torch.full((n_right,), LANE_CENTRES[ego_lane_idx].item(), device=DEVICE)

    lo = max(0, ego_lane_idx - 1)
    hi = min(len(LANE_CENTRES), ego_lane_idx + 2)
    adj = LANE_CENTRES[lo:hi]
    y_spread = torch.tensor(
        [adj[i % len(adj)].item() for i in range(n_spread)],
        dtype=torch.float32, device=DEVICE,
    )
    x_spread = torch.full((n_spread,), x0 + goal_dist, device=DEVICE)

    x_g = torch.cat([x_right, x_spread])
    y_g = torch.cat([y_right, y_spread])
    return torch.stack([x_g, y_g,
                        torch.zeros(L, device=DEVICE),
                        torch.full((L,), VMAX, device=DEVICE)], dim=1)