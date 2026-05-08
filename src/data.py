"""
Data loading and obstacle prediction.

Paper reference:
  - Section III (Assumption 3): constant-velocity obstacle prediction.
  - Section IV: NGSIM dataset and IDM synthetic scenario.

Supports two data sources:
  1. NGSIM US-101 CSV (real trajectories, chunked loading for large files).
  2. Synthetic IDM scenario (fallback when data.csv is absent).
"""

import os
import torch
import pandas as pd
from config import *

_CSV_PATH   = os.path.join(os.path.dirname(__file__), "data.csv")
_CHUNK_SIZE = 50_000

# NGSIM column names
COL_FRAME = "Frame_ID"
COL_ID    = "Vehicle_ID"
COL_X     = "Local_Y"   # longitudinal position [ft] — mapped to paper's x axis
COL_Y     = "Local_X"   # lateral position [ft]      — mapped to paper's y axis
COL_VEL   = "v_Vel"     # scalar speed [ft/s]

_FT_TO_M = 0.3048   # feet-to-metres conversion

# Module-level cache: populated once on first call to get_state()
_frames: dict      = {}
_frame_index: list = []
_LOCATION_FILTER   = "us-101"


# ── NGSIM loading ─────────────────────────────────────────────────────────────

def _init_ngsim():
    """
    Load data.csv in chunks, filter to _LOCATION_FILTER, group by Frame_ID,
    and build a sorted frame index for O(1) lookup in get_state().
    Called once; subsequent calls return immediately.
    """
    global _frames, _frame_index
    if _frame_index:
        return

    print(f"[data] Loading {_CSV_PATH} (location={_LOCATION_FILTER}) ...")

    chunks = []
    for chunk in pd.read_csv(_CSV_PATH, chunksize=_CHUNK_SIZE):
        if "Location" in chunk.columns:
            chunk = chunk[chunk["Location"] == _LOCATION_FILTER]
        chunks.append(chunk)

    df = pd.concat(chunks, ignore_index=True)
    print(f"[data] Rows after location filter: {len(df):,}")

    for fid, grp in df.groupby(COL_FRAME):
        if len(grp) >= 1 + NUM_OBS:
            _frames[fid] = grp.sort_values(COL_X, ascending=False).reset_index(drop=True)

    _frame_index = sorted(_frames.keys())
    print(f"[data] Loaded {len(_frame_index)} valid frames.")


def _load_frame(frame_id):
    """Return the cached DataFrame for a given Frame_ID."""
    if frame_id not in _frames:
        raise ValueError(f"frame_id {frame_id} not in loaded frames.")
    return _frames[frame_id]


def _row_to_dict(row):
    """
    Convert a NGSIM CSV row to an ego/neighbor state dict.

    NGSIM Local_Y is longitudinal (paper's x), Local_X is lateral (paper's y).
    All values converted from feet to metres.
    """
    return {
        "x":          float(row[COL_X]) * _FT_TO_M,
        "y":          float(row[COL_Y]) * _FT_TO_M,
        "psi":        0.0,
        "vx":         float(row[COL_VEL]) * _FT_TO_M,
        "vy":         0.0,
        "vehicle_id": int(row[COL_ID]),
    }


def _pick_ego_row(frame, ego_id=None):
    """
    Select the ego vehicle row from a frame DataFrame.
    If ego_id is given and present, returns that vehicle's row.
    Otherwise picks a vehicle at ~40% from the front of the sorted frame
    (sorted descending by longitudinal position), ensuring traffic ahead.
    """
    if ego_id is not None and ego_id in frame[COL_ID].values:
        return frame[frame[COL_ID] == ego_id].iloc[0]
    n = len(frame)
    return frame.iloc[max(1, n * 2 // 5)]


def _pick_neighbors(frame, ego, num):
    """
    Select the NUM_OBS closest forward vehicles to the ego for use as
    MPC obstacles (Section IV: obstacle selection).

    Selection criteria:
    - Forward vehicles only (0 < dx < 100 m).
    - Within two lane-widths laterally (|dy| < 7.4 m).
    - Ranked by elliptical distance matching the collision model (Eq. 4).

    Pads with dummy far-ahead vehicles if fewer than num candidates exist.
    """
    ego_x = ego["x"]
    ego_y = ego["y"]

    rest = frame[frame[COL_ID] != ego["vehicle_id"]].copy()

    rest["_xm"] = rest[COL_X] * _FT_TO_M
    rest["_ym"] = rest[COL_Y] * _FT_TO_M

    dx = rest["_xm"] - ego_x
    dy = rest["_ym"] - ego_y

    lane_width   = 3.7
    mask_forward = (dx > 0) & (dx < 100.0)
    mask_lane    = dy.abs() < (2.0 * lane_width)

    candidates = rest[mask_forward & mask_lane].copy()

    # Elliptical distance score matching Eq. (4)
    candidates["_score"] = ((dx.loc[candidates.index] / A_OBS)**2 +
                            (dy.loc[candidates.index] / B_OBS)**2)
    candidates = candidates.nsmallest(num, "_score")

    neighbors = [_row_to_dict(candidates.iloc[i]) for i in range(len(candidates))]

    # Pad with far-ahead dummy vehicles if insufficient real neighbors
    while len(neighbors) < num:
        neighbors.append({
            "x":          ego_x + 80.0 + 20.0 * len(neighbors),
            "y":          ego_y,
            "psi":        0.0,
            "vx":         ego["vx"],
            "vy":         0.0,
            "vehicle_id": -(len(neighbors) + 1),
        })

    return neighbors


# ── Public API ────────────────────────────────────────────────────────────────

def predict_obstacles(neighbors):
    """
    Predict obstacle trajectories over the planning horizon using the
    constant-velocity model (Section III, Assumption 3).

    Formula: ξx(tk) = x0 + vx·tk,  ξy(tk) = y0 + vy·tk

    Parameters
    ----------
    neighbors : list of dict
        Each dict has keys 'x', 'y', 'vx', 'vy'.

    Returns
    -------
    obs_x, obs_y : (N * NUM_OBS, L)
        Obstacle position predictions, replicated across L batch instances.
    
    Crucially, this uses only current vx, vy.
    """
    num_obs = len(neighbors)
    obs_x = torch.zeros(N * num_obs, L, device=DEVICE)
    obs_y = torch.zeros(N * num_obs, L, device=DEVICE)

    for j, nb in enumerate(neighbors):
        x0, y0 = nb["x"], nb["y"]
        vx, vy = nb["vx"], nb["vy"]
        for k in range(N):
            # Extrapolate linearly
            obs_x[j * N + k, :] = x0 + vx * (k * DT)
            obs_y[j * N + k, :] = y0 + vy * (k * DT)

    return obs_x, obs_y


def get_state(t, ego_id=None, advance=False):
    """
    Return the ego state and MPC neighbor list at MPC step t.

    Uses NGSIM data if data.csv exists; falls back to synthetic IDM otherwise.

    Parameters
    ----------
    t : int
        MPC time index (0-based index into the sorted frame list).
    ego_id : int, optional
        Vehicle ID to use as ego across frames for trajectory continuity.
    advance : bool, optional
        Whether to advance the synthetic scenario by one time step.

    Returns
    -------
    ego : dict
        State dict with keys 'x', 'y', 'psi', 'vx', 'vy', 'vehicle_id'.
    neighbors : list of dict
        List of NUM_OBS neighbor state dicts.
    """
    if os.path.exists(_CSV_PATH):
        return _get_state_ngsim(t, ego_id)
    return _get_state_synthetic(t,current_ego = ego_id,advance=advance)


def _get_state_ngsim(t, ego_id=None):
    """Load ego and neighbors from the NGSIM frame at index t."""
    _init_ngsim()

    if t >= len(_frame_index):
        raise ValueError(f"t={t} out of range ({len(_frame_index)} frames)")

    frame     = _load_frame(_frame_index[t])
    ego_row   = _pick_ego_row(frame, ego_id)
    ego       = _row_to_dict(ego_row)
    neighbors = _pick_neighbors(frame, ego, NUM_OBS)

    return ego, neighbors


_synthetic_neighbors_init = None
_synthetic_cache = {} # Persistent state for synthetic neighbor velocities

def _get_state_synthetic(t, current_ego=None, advance=False):
    """
    Synthetic IDM-style scenario (Section IV).
    Neighbors move parallel to centerline but adapt vx to the car in front.
    """
    global _synthetic_cache, _synthetic_neighbors_init
    
    # 1. Ego State (centered in Lane 1: 5.55m)
    if current_ego is not None:
        ego = current_ego
    else:
        ego = {"x": 0.0, "y": 5.55, "psi": 0.0, "vx": 10.0, "vy": 0.0, "vehicle_id": 0}

    # 2. Initialization (One-time)
    if _synthetic_neighbors_init is None:
        import random
        rng = random.Random(42)
        lane_options = [1.85, 5.55, 9.25, 12.95, 16.65]
        lane_reach = {lane: 10.0 for lane in lane_options}
        _synthetic_neighbors_init = []
        
        for i in range(15): # Generate a pool of 15 vehicles
            lane = rng.choice(lane_options)
            dx = lane_reach[lane] + rng.uniform(25.0, 40.0)
            lane_reach[lane] = dx
            v_target = rng.uniform(8.0, 10.0)
            _synthetic_neighbors_init.append({
                "id": i + 1, "x": dx, "y": lane, "v": v_target, "v_target": v_target
            })
        _synthetic_cache = {n["id"]: n for n in _synthetic_neighbors_init}

    # 3. Simulate IDM Step (Behavioral Reality)
    # This runs every step t to update the "true" positions of neighbors
    if advance:
        dt_sim = DT
        for nid in _synthetic_cache:
            n = _synthetic_cache[nid]
            # Find car in front in the SAME lane
            lead_car = None
            min_gap = 1000.0
            for other_id in _synthetic_cache:
                if other_id == nid: continue
                other = _synthetic_cache[other_id]
                if other["y"] == n["y"] and other["x"] > n["x"]:
                    gap = other["x"] - n["x"]
                    if gap < min_gap:
                        min_gap = gap
                        lead_car = other
            
            # Simple IDM speed adaptation
            if lead_car and min_gap < 20.0:
                # Slow down to match lead car speed with a safety buffer
                n["v"] = max(n["v"] - 0.5, lead_car["v"]) 
            else:
                # Accelerate back to target speed
                n["v"] = min(n["v"] + 0.1, n["v_target"])
                
            n["x"] += n["v"] * dt_sim

    # 4. Filter for MPC (Section IV: Selection of NUM_OBS neighbors)
    all_neighbors = [
        {"x": n["x"], "y": n["y"], "psi": 0.0, "vx": n["v"], "vy": 0.0, "vehicle_id": n["id"]}
        for n in _synthetic_cache.values()
    ]
    
    # Sort by distance to ego and pick NUM_OBS
    all_neighbors.sort(key=lambda v: (v["x"] - ego["x"])**2 + (v["y"] - ego["y"])**2)
    return ego, all_neighbors


def get_all_vehicles(t, ego_id=None):
    """
    Return the ego state and all vehicles in frame t (for visualisation).

    Unlike get_state(), this returns every vehicle in the frame rather than
    only the NUM_OBS MPC neighbors.

    Parameters
    ----------
    t : int
        MPC time index.
    ego_id : int, optional
        Vehicle ID to use as ego.

    Returns
    -------
    ego : dict
    vehicles : list of dict
        All vehicles in the frame (including non-MPC background vehicles).
    """
    if os.path.exists(_CSV_PATH):
        _init_ngsim()

        if t >= len(_frame_index):
            raise ValueError(f"t={t} out of range ({len(_frame_index)} frames)")

        frame   = _load_frame(_frame_index[t])
        ego_row = _pick_ego_row(frame, ego_id)
        ego     = _row_to_dict(ego_row)

        vehicles = [
            {
                "x":          float(row[COL_X]) * _FT_TO_M,
                "y":          float(row[COL_Y]) * _FT_TO_M,
                "psi":        0.0,
                "vx":         float(row[COL_VEL]) * _FT_TO_M,
                "vy":         0.0,
                "vehicle_id": int(row[COL_ID]),
            }
            for _, row in frame.iterrows()
        ]

        return ego, vehicles

    ego, neighbors = _get_state_synthetic(t,ego_id,advance=False)
    return ego, [ego] + neighbors