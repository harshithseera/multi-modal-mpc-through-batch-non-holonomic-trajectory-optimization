"""
Bird's-eye visualisation of the multi-modal MPC system.

Paper reference: Section IV (validation figures).

Displays:
  - Candidate trajectories (grey) and optimal trajectory (red) from the
    batch optimizer (Section III-F).
  - All vehicles in the current NGSIM frame (blue).
  - MPC obstacle neighbors (same blue as background).
  - Ego vehicle (pink), scrolling viewport centred on ego.
  - Stats bar: collision flag, average speed, heading, trajectory count.
"""

import sys, os, math
sys.path.insert(0, os.path.dirname(__file__))

import torch
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

from basis import build_basis
from matrices import build_F, build_A
from optimizer import optimize_batch
from goals import sample_goals
from data import get_state, predict_obstacles, get_all_vehicles
from meta_cost import compute_meta_cost
from config import NUM_OBS, DT, VMIN, VMAX, L, N, A_OBS, B_OBS, HEADING_LIMIT_DEG

# ── Display parameters ────────────────────────────────────────────────────────
VIS_A = 2.5   # vehicle ellipse half-length for display [m]
VIS_B = 1.0   # vehicle ellipse half-width for display [m]

MAX_STEPS   = 200
T_START     = 100     # NGSIM frame index to begin from
META_MODE   = "highway"
Y_RIGHT     = 1.5     # right-lane lateral coordinate [m] for Eq. (26)
W1, W2      = 1.0, 0.5
PAUSE       = 0.001    # matplotlib pause between frames [s]

VIEW_HALF_X     = 120.0   # half-width of scrolling viewport [m]
ROAD_Y_MIN      = -1.0
ROAD_Y_MAX      =  22.0
LANE_YS         = [0.0, 3.7, 7.4, 11.1, 14.8, 18.5]

MIN_LABEL_SEP_X = 10.0   # minimum longitudinal gap between speed labels [m]
MIN_LABEL_SEP_Y =  2.0   # minimum lateral gap between speed labels [m]

# ── Colour palette ────────────────────────────────────────────────────────────
BG        = "#f8f9fa"
ROAD_FILL = "#ffffff"
LANE_LINE = "#cccccc"
CAND_COL  = "#aaaaaa"
BEST_COL  = "#cc2222"
EGO_COL   = "#f4aaaa"
EGO_EDGE  = "#cc2222"
OBS_COL   = "#4a90d9"
OBS_EDGE  = "#2a6aaa"
OBS_TXT   = "#ffffff"
STAT_COL  = "#333333"


def _safe(v, fb=0.0):
    """Return fb if v is NaN or Inf, else return v."""
    return fb if (math.isnan(v) or math.isinf(v)) else v


def _check_collision(ego, vehicles):
    """
    Check whether the ego vehicle penetrates any other vehicle's safety
    ellipse (Eq. 1e) at the current timestep.
    """
    for v in vehicles:
        if v.get("vehicle_id") == ego.get("vehicle_id"):
            continue
        dx = ego["x"] - v["x"]
        dy = ego["y"] - v["y"]
        if (dx / A_OBS)**2 + (dy / B_OBS)**2 < 1.0:
            return True
    return False


def _draw_vehicles(ax, vehicles, ego_id, vlo, vhi):
    """
    Draw all non-ego vehicles.
    Draw all non-ego vehicles as blue ellipses with speed labels.
    """
    placed = []
    for v in vehicles:
        vid = v.get("vehicle_id")
        if vid == ego_id:
            continue
        lon, lat = v["x"], v["y"]
        if not (vlo - A_OBS <= lon <= vhi + A_OBS):
            continue


        fc = OBS_COL
        ec = OBS_EDGE
        zo = 4

        ax.add_patch(mpatches.Ellipse(
            (lon, lat), width=2*VIS_A, height=2*VIS_B,
            facecolor=fc, edgecolor=ec, linewidth=1.0, zorder=zo))

        if not any(abs(lon - px) < MIN_LABEL_SEP_X and abs(lat - py) < MIN_LABEL_SEP_Y
                   for px, py in placed):
            ax.text(lon, lat, f"{v['vx']:.1f}", color=OBS_TXT, fontsize=7,
                    ha="center", va="center", zorder=zo + 1, fontweight="bold")
            placed.append((lon, lat))


def main():
    P, P_dot, P_ddot = build_basis()
    Fmat    = build_F(P, P_dot, P_ddot, num_obs=NUM_OBS)
    A       = build_A(P)   # (4, 2K) position-only boundary constraints (Eq. 8b)
    T_total = (N - 1) * DT
    vel_idx = N // 2

    ego, neighbors = get_state(T_START)

    fig, ax = plt.subplots(figsize=(16, 5), facecolor=BG)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.78, bottom=0.12)
    plt.ion()
    plt.show()

    for t in range(MAX_STEPS):

        ox, oy = ego["x"], ego["y"]

        # Goal sampling (Section III-F, Eq. 26) — neighbors passed to
        # detect blocked lanes and force lateral goal hypotheses
        goals = sample_goals(ego, neighbors)

        # Constant-velocity obstacle prediction (Section III, Assumption 3)
        obs_x_w, obs_y_w = predict_obstacles(neighbors)

        _, all_vehicles = get_all_vehicles(T_START + t, ego_id=ego.get("vehicle_id"))

        # Batch optimisation (Algorithm 1) — world-frame inputs/outputs
        cx, cy, cpsi, v = optimize_batch(
            P, P_dot, P_ddot, Fmat, A, goals, obs_x_w, obs_y_w, ego)

        # World-frame trajectory points for plotting (cx[:,0] has ox baked in)
        x_trajs = (P @ cx.T).detach().cpu().numpy()   # (N, L)
        y_trajs = (P @ cy.T).detach().cpu().numpy()

        # Meta-cost ranking with heading filter (Section III-F, Eq. 26)
        y_pos = P @ cy.T
        cost  = compute_meta_cost(v, y_pos, mode=META_MODE,
                                  y_right=Y_RIGHT, w1=W1, w2=W2)

        # Discard trajectories with total heading change > HEADING_LIMIT_DEG
        xd_t      = (P_dot / T_total) @ cx.T
        yd_t      = (P_dot / T_total) @ cy.T
        psi_t     = torch.atan2(yd_t, xd_t)
        delta_deg = torch.abs(psi_t[-1] - psi_t[0]) * 180.0 / torch.pi
        cost[delta_deg > HEADING_LIMIT_DEG] = float("inf")

        best = cost.argmin().item()

        bcx, bcy = cx[best], cy[best]

        # Receding-horizon state advance (one DT step)
        x_next  = _safe((P[1]           @ bcx).item(), ox)
        y_next  = _safe((P[1]           @ bcy).item(), oy)
        vx_next = _safe((P_dot[vel_idx] @ bcx).item() / T_total, ego["vx"])
        vy_next = _safe((P_dot[vel_idx] @ bcy).item() / T_total, 0.0)

        spd = (vx_next**2 + vy_next**2) ** 0.5
        if spd > 1e-6:
            sc = min(max(spd, VMIN), VMAX) / spd
            vx_next *= sc
            vy_next *= sc

        psi_next    = float(np.arctan2(vy_next, vx_next)) if spd > 0.01 else ego["psi"]
        collision   = _check_collision(ego, all_vehicles)
        avg_speed   = _safe(float(v[best].mean()), 0.0)
        orientation = float(np.degrees(psi_next))

        # ── Drawing ───────────────────────────────────────────────────────────
        vlo, vhi = ox - VIEW_HALF_X, ox + VIEW_HALF_X

        ax.cla()
        ax.set_facecolor(ROAD_FILL)
        for sp in ax.spines.values():
            sp.set_edgecolor("#dddddd")
        ax.tick_params(colors=STAT_COL, labelsize=8)

        for ly in LANE_YS:
            ls = "-" if ly in [LANE_YS[0], LANE_YS[-1]] else "--"
            ax.axhline(ly, color=LANE_LINE, linewidth=0.8, linestyle=ls, zorder=1)

        # Candidate trajectories from batch optimizer
        for li in range(L):
            if li == best:
                continue
            ax.plot(x_trajs[:, li], y_trajs[:, li],
                    color=CAND_COL, linewidth=1.0, alpha=0.5, zorder=2)

        # Optimal trajectory (lowest meta-cost after heading filter)
        ax.plot(x_trajs[:, best], y_trajs[:, best],
                color=BEST_COL, linewidth=2.5, zorder=3, solid_capstyle="round")

        # All vehicles — uniform blue
        _draw_vehicles(ax, all_vehicles, ego.get("vehicle_id"), vlo, vhi)

        # Ego vehicle
        ax.add_patch(mpatches.Ellipse(
            (ox, oy), width=2*VIS_A, height=2*VIS_B,
            facecolor=EGO_COL, edgecolor=EGO_EDGE, linewidth=1.5, zorder=7))
        ax.text(ox, oy, f"{ego['vx']:.1f}", color="#882222", fontsize=8,
                ha="center", va="center", zorder=8, fontweight="bold")

        ax.set_xlim(vlo, vhi)
        ax.set_ylim(ROAD_Y_MIN, ROAD_Y_MAX)
        ax.set_xlabel("Longitudinal (m)", fontsize=9, color=STAT_COL)
        ax.set_ylabel("Lateral (m)",      fontsize=9, color=STAT_COL)
        ax.set_aspect("auto")

        ax.legend(handles=[
            Line2D([0], [0], color=BEST_COL, lw=2.5, label="Optimal trajectory"),
            Line2D([0], [0], color=CAND_COL, lw=1.0, alpha=0.6,
                   label="Candidate trajectories"),
            mpatches.Patch(facecolor=OBS_COL, edgecolor=OBS_EDGE,
                           label="Background vehicles"),
            mpatches.Patch(facecolor=EGO_COL, edgecolor=EGO_EDGE, label="Ego"),
        ], fontsize=7.5, loc="upper right", facecolor="white",
           edgecolor="#cccccc", framealpha=0.9)

        col_c = "#cc2222" if collision else "#226622"
        for i, (txt, col) in enumerate([
            (f"Collision= {collision}",             col_c),
            (f"Avg speed= {avg_speed:.2f} m/s",     STAT_COL),
            (f"Orientation= {orientation:.1f} deg", STAT_COL),
            (f"Trajectories= {L}",                  STAT_COL),
        ]):
            ax.text(0.02 + i * 0.25, 1.10, txt, transform=ax.transAxes,
                    color=col, fontsize=9, ha="left", va="top",
                    bbox=dict(boxstyle="round,pad=0.3",
                              facecolor="white" if i > 0
                              else ("#ffe0e0" if collision else "#e0ffe0"),
                              edgecolor="#cccccc", alpha=0.9))

        fig.suptitle("Highway environment", fontsize=12, color=STAT_COL, y=0.98)
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(PAUSE)

        # Advance to next NGSIM frame; update ego from optimizer output
        _, neighbors = get_state(T_START + t + 1, ego_id=ego.get("vehicle_id"))
        ego = {
            "x":          x_next,
            "y":          y_next,
            "psi":        psi_next,
            "vx":         vx_next,
            "vy":         vy_next,
            "vehicle_id": ego.get("vehicle_id"),
        }

        print(f"Step {t:3d} | best={best} | hdg={delta_deg[best].item():.1f}° | "
              f"collision={collision} | ego=({x_next:.1f},{y_next:.1f}) | "
              f"vx={vx_next:.2f}")

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()