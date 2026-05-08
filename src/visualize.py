"""
Highway Bird's-Eye MPC Visualization  —  run with: python src/visualize.py

Saves output to mpc_output.mp4 via FFmpeg.
Requires: pip install matplotlib and brew install ffmpeg (macOS).
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
from matplotlib.animation import FFMpegWriter

from basis import build_basis
from matrices import build_F, build_A
from optimizer import optimize_batch
from goals import sample_goals
from data import get_state, predict_obstacles, get_all_vehicles
from meta_cost import compute_meta_cost
from config import *

# Visual proportions match MPC collision ellipses
VIS_A = A_OBS
VIS_B = B_OBS

MAX_STEPS   = 200
T_START     = 100
META_MODE   = "cruise"
PAUSE       = 0.01

VIEW_HALF_X = 100.0
ROAD_Y_MIN  = -2.0
ROAD_Y_MAX  =  22.0
LANE_YS     = [3.7 * i for i in range(6)]

VIDEO_PATH  = "cruise_idm.mp4"
VIDEO_FPS   = 10

# Color Palette
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
    return fb if (math.isnan(v) or math.isinf(v)) else v

def _check_collision(ego, vehicles):
    for v in vehicles:
        if v.get("vehicle_id") == ego.get("vehicle_id"):
            continue
        dx = ego["x"] - v["x"]
        dy = ego["y"] - v["y"]
        if (dx/A_OBS)**2 + (dy/B_OBS)**2 < 1.0:
            return True
    return False

def _draw_vehicles(ax, vehicles, ego_id, vlo, vhi):
    for v in vehicles:
        if v.get("vehicle_id") == ego_id:
            continue
        lon, lat = v["x"], v["y"]
        if not (vlo - VIS_A <= lon <= vhi + VIS_A):
            continue
        ax.add_patch(mpatches.Ellipse(
            (lon, lat), width=VIS_A, height=VIS_B,
            facecolor=OBS_COL, edgecolor=OBS_EDGE,
            linewidth=1.0, zorder=4))
        ax.text(lon, lat, f"{v['vx']:.1f}", color=OBS_TXT, fontsize=6,
                ha="center", va="center", zorder=5, fontweight="bold")

def main():
    P, P_dot, P_ddot = build_basis()
    Fmat = build_F(P, P_dot, P_ddot, num_obs=NUM_OBS)
    A = build_A(P, P_dot)
    T_total =N  * DT
    Pd = P_dot / T_total  # Physical derivative basis

    ego, all_neighbors = get_state(T_START)
    neighbors = all_neighbors[:NUM_OBS]

    # Trip Statistics
    total_speed_sum = 0.0
    frame_count = 0

    fig, ax = plt.subplots(figsize=(16, 5), facecolor=BG)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.78, bottom=0.12)
    plt.ion()
    plt.show()

    writer = FFMpegWriter(fps=VIDEO_FPS)

    with writer.saving(fig, VIDEO_PATH, dpi=120):
        for t in range(MAX_STEPS):
            goals = sample_goals(ego,t, mode='cruise')
            obs_x_w, obs_y_w = predict_obstacles(neighbors)
            _, all_vehicles = get_all_vehicles(T_START + t, ego_id=ego)

            cx, cy, cpsi, v_profile, res_obs = optimize_batch(
                P, P_dot, P_ddot, Fmat, A, goals, obs_x_w, obs_y_w, ego)

            y_pos = P @ cy.T
            cost = compute_meta_cost(v_profile, y_pos, res_obs, mode="cruise")
            best = cost.argmin().item()

            # average of top-3 trajectories for smoother control commands as per official implementation
            # Linear Velocity (v)
            v_cmd = v_profile[best, :3].mean().item()
            
            # Heading Rate (w)
            # Physical psidot = Pd @ cpsi
            psidot_profile = Pd @ cpsi[best]
            w_cmd = psidot_profile[:3].mean().item()

            # Current state
            ox, oy = ego["x"], ego["y"]
            psi_curr = float(ego["psi"])

            # 1. Advance position based on current heading and speed
        
            vx_cmd = v_cmd * math.cos(psi_curr)
            vy_cmd = v_cmd * math.sin(psi_curr)
            x_next = ox + vx_cmd * DT
            
            y_next = oy + vy_cmd * DT
            
            # 2. Advance orientation based on heading rate
            psi_next = psi_curr + w_cmd * DT
            v_cmd = math.sqrt(vx_cmd**2 + vy_cmd**2)

            # Update Trip Stats
            total_speed_sum += v_cmd
            frame_count += 1
            trip_avg_speed = total_speed_sum / frame_count

            # --- VISUALIZATION ---
            ax.cla()
            ax.set_facecolor(ROAD_FILL)
            for ly in LANE_YS:
                ls = "-" if ly in [LANE_YS[0], LANE_YS[-1]] else "--"
                ax.axhline(ly, color=LANE_LINE, linewidth=0.8, linestyle=ls, zorder=1)

            x_trajs = (P @ cx.T).detach().cpu().numpy()
            y_trajs = (P @ cy.T).detach().cpu().numpy()
            for li in range(L):
                col = BEST_COL if li == best else CAND_COL
                lw = 2.5 if li == best else 1.0
                z = 3 if li == best else 2
                ax.plot(x_trajs[:, li], y_trajs[:, li], color=col, linewidth=lw, zorder=z)

            vlo, vhi = ox - VIEW_HALF_X, ox + VIEW_HALF_X
            _draw_vehicles(ax, all_vehicles, ego.get("vehicle_id"), vlo, vhi)
            
            ax.add_patch(mpatches.Ellipse((ox, oy), width=VIS_A, height=VIS_B, facecolor=EGO_COL, edgecolor=EGO_EDGE, linewidth=1.5, zorder=7))
            ax.text(ox, oy, f"{v_cmd:.1f}", color="#882222", fontsize=6, ha="center", va="center", zorder=8, fontweight="bold")

            ax.set_xlim(vlo, vhi)
            ax.set_ylim(ROAD_Y_MIN, ROAD_Y_MAX)
            ax.set_aspect("equal")

            collision = _check_collision(ego, all_vehicles)
            col_c = "#cc2222" if collision else "#226622"
            
            labels = [
                (f"Collision: {collision}", col_c),
                (f"Trip Avg Speed: {trip_avg_speed:.2f} m/s", STAT_COL),
                (f"Current Speed: {v_cmd:.2f} m/s", STAT_COL),
                (f"Trajectories: {L}", STAT_COL),
            ]
            for i, (txt, col) in enumerate(labels):
                ax.text(0.02 + i*0.23, 1.10, txt, transform=ax.transAxes, color=col, fontsize=10, fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#cccccc", alpha=0.9))

            fig.canvas.draw()
            writer.grab_frame()
            plt.pause(PAUSE)

            # Step environment
            ego = {
                "x": x_next, 
                "y": y_next, 
                "psi": psi_next, 
                "vx": vx_cmd, 
                "vy": vy_cmd, 
                "vehicle_id": 0
            }
            _, all_neighbors = get_state(T_START + t + 1, ego_id=ego, advance=True)
            neighbors = all_neighbors[:NUM_OBS]
            
            print(f"Step {t:3d} | best={best} | x={x_next:.1f} y={y_next:.2f} | v={v_cmd:.1f}")

    print(f"Video saved to {VIDEO_PATH}")
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()