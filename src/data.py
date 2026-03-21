import os
import torch
import pandas as pd
from config import N, NUM_OBS, DT, L, DEVICE

# ─────────────────────────────────────────────────────────────
# CSV loading — chunked to handle large files (~1.5 GB)
# ─────────────────────────────────────────────────────────────

# Expected NGSIM-style columns (adapt names if your CSV differs):
#   frame_id, vehicle_id, local_x, local_y, v_length, v_lat, psi
# local_x = lateral position, local_y = longitudinal position

_CSV_PATH   = os.path.join(os.path.dirname(__file__), "data.csv")
_CHUNK_SIZE = 50_000   # rows per chunk — tune to available RAM

# Column name mapping — change these if your CSV uses different headers
COL_FRAME  = "Frame_ID"
COL_ID     = "Vehicle_ID"
COL_X      = "Local_Y"    # longitudinal position in road-frame
COL_Y      = "Local_X"    # lateral position in road-frame
COL_VEL    = "v_Vel"      # scalar speed (ft/s in NGSIM)
COL_LANE   = "Lane_ID"    # used to derive lateral heading assumption

# Note: NGSIM does not provide vx/vy or psi directly.
# We assume vehicles drive parallel to the road (psi=0), so:
#   vx = v_Vel,  vy = 0
# This matches Section III assumption 3 (constant-velocity, road-aligned).


# In-memory frame store — populated once on first call to get_state
# Filtered to rows that have enough vehicles, keyed by sorted frame index
_frames: dict = {}          # frame_id -> pd.DataFrame
_frame_index: list = []     # sorted list of valid frame_ids

def _init_ngsim():
    """
    Load the full CSV into memory once, grouped by Frame_ID.
    Only retains frames that have at least 1 + NUM_OBS vehicles.
    Called automatically on first get_state() call.
    ~50-100 MB in memory after filtering — acceptable for a 1.5 GB source file.
    """
    global _frames, _frame_index
    if _frame_index:
        return   # already initialised

    print(f"[data] Loading {_CSV_PATH} into memory (one-time, please wait)...")
    all_chunks = []
    for chunk in pd.read_csv(_CSV_PATH, chunksize=_CHUNK_SIZE):
        all_chunks.append(chunk)
    df = pd.concat(all_chunks, ignore_index=True)

    # Group by frame, keep only frames with enough vehicles
    grouped = df.groupby(COL_FRAME)
    valid = []
    for fid, grp in grouped:
        if len(grp) >= 1 + NUM_OBS:
            # sort by longitudinal position descending once at load time
            _frames[fid] = grp.sort_values(
                COL_X, ascending=False
            ).reset_index(drop=True)
            valid.append(fid)

    _frame_index = sorted(valid)
    print(f"[data] Loaded {len(_frame_index)} valid frames.")


def _load_frame(frame_id: int) -> pd.DataFrame:
    """Return the pre-loaded DataFrame for frame_id (O(1) dict lookup)."""
    if frame_id not in _frames:
        raise ValueError(f"frame_id {frame_id} not in loaded frames.")
    return _frames[frame_id]


_FT_TO_M = 0.3048   # NGSIM positions are in feet, velocities in ft/s

def _row_to_dict(row: pd.Series) -> dict:
    """Convert a DataFrame row to the ego/neighbor state dict.
    Converts from NGSIM feet/ft·s⁻¹ to metres/m·s⁻¹."""
    return {
        "x":   float(row[COL_X])   * _FT_TO_M,
        "y":   float(row[COL_Y])   * _FT_TO_M,
        "psi": 0.0,
        "vx":  float(row[COL_VEL]) * _FT_TO_M,
        "vy":  0.0,
    }


# ─────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────

def get_state(t: int):
    """
    Return ego + NUM_OBS neighbors at MPC timestep t.

    t is a 0-based index into the sorted unique Frame_IDs in the CSV,
    not the raw Frame_ID value itself. So get_state(0) always returns
    the first available frame regardless of what number it has.

    Falls back to synthetic IDM scenario if data.csv is absent.
    """
    if os.path.exists(_CSV_PATH):
        return _get_state_ngsim(t)
    else:
        return _get_state_synthetic(t)


def _get_state_ngsim(t: int):
    """Load one MPC step from the in-memory NGSIM frames using t as index."""
    _init_ngsim()
    if t >= len(_frame_index):
        raise ValueError(
            f"t={t} out of range — {len(_frame_index)} valid frames (0…{len(_frame_index)-1})"
        )
    frame_id = _frame_index[t]
    frame    = _load_frame(frame_id)   # O(1) dict lookup
    ego       = _row_to_dict(frame.iloc[0])
    neighbors = [_row_to_dict(frame.iloc[i + 1]) for i in range(NUM_OBS)]
    return ego, neighbors


def _get_state_synthetic(t: int):
    """
    Synthetic IDM-style scenario.
    Ego starts at origin doing 10 m/s; neighbors are ahead in three lanes.
    State advances by one DT step each call so the MPC loop progresses.
    """
    x0 = float(t) * 10.0 * DT

    ego = {"x": x0, "y": 0.0, "psi": 0.0, "vx": 10.0, "vy": 0.0}

    lane_offsets = [-4.0, 0.0, 4.0]
    neighbors = []
    for i in range(NUM_OBS):
        lane = lane_offsets[i % len(lane_offsets)]
        neighbors.append({
            "x":   x0 + 20.0 + i * 10.0,
            "y":   lane,
            "psi": 0.0,
            "vx":  8.0,
            "vy":  0.0,
        })

    return ego, neighbors


def predict_obstacles(neighbors):
    """
    Constant-velocity prediction over N timesteps (Section III assumption 3).

    Returns:
        obs_x: (N*NUM_OBS, L)  longitudinal positions tiled across batch
        obs_y: (N*NUM_OBS, L)  lateral positions tiled across batch

    Formula: xi(tk) = x0 + vx * tk,  yi(tk) = y0 + vy * tk
    """
    t_vec = torch.arange(N, device=DEVICE).float() * DT   # (N,)

    xs, ys = [], []
    for n in neighbors:
        xs.append(n["x"] + n["vx"] * t_vec)   # (N,)
        ys.append(n["y"] + n["vy"] * t_vec)

    obs_x = torch.stack(xs, dim=0).reshape(-1)              # (N*NUM_OBS,)
    obs_y = torch.stack(ys, dim=0).reshape(-1)

    obs_x = obs_x.unsqueeze(1).expand(-1, L).contiguous()   # (N*NUM_OBS, L)
    obs_y = obs_y.unsqueeze(1).expand(-1, L).contiguous()

    return obs_x, obs_y