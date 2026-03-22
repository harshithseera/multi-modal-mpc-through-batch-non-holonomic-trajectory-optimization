import os
import torch
import pandas as pd
from config import N, NUM_OBS, DT, L, DEVICE, A_OBS, B_OBS

_CSV_PATH   = os.path.join(os.path.dirname(__file__), "data.csv")
_CHUNK_SIZE = 50_000

COL_FRAME = "Frame_ID"
COL_ID    = "Vehicle_ID"
COL_X     = "Local_Y"   # longitudinal (ft)
COL_Y     = "Local_X"   # lateral      (ft)
COL_VEL   = "v_Vel"     # speed        (ft/s)

_FT_TO_M = 0.3048

_frames: dict = {}
_frame_index: list = []
_LOCATION_FILTER = "us-101"


def _init_ngsim():
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
    if frame_id not in _frames:
        raise ValueError(f"frame_id {frame_id} not in loaded frames.")
    return _frames[frame_id]


def _row_to_dict(row):
    return {
        "x":          float(row[COL_X]) * _FT_TO_M,
        "y":          float(row[COL_Y]) * _FT_TO_M,
        "psi":        0.0,
        "vx":         float(row[COL_VEL]) * _FT_TO_M,
        "vy":         0.0,
        "vehicle_id": int(row[COL_ID]),
    }


def _pick_ego_row(frame, ego_id=None):
    if ego_id is not None and ego_id in frame[COL_ID].values:
        return frame[frame[COL_ID] == ego_id].iloc[0]

    n = len(frame)
    return frame.iloc[max(1, n * 2 // 5)]


def _pick_neighbors(frame, ego, num):
    """
    Improved obstacle selection for MPC:
    - Uses elliptical distance (matches collision model)
    - Filters forward vehicles only
    - Filters nearby lanes only
    """

    ego_x = ego["x"]
    ego_y = ego["y"]

    rest = frame[frame[COL_ID] != ego["vehicle_id"]].copy()

    # convert to meters
    rest["_xm"] = rest[COL_X] * _FT_TO_M
    rest["_ym"] = rest[COL_Y] * _FT_TO_M

    dx = rest["_xm"] - ego_x
    dy = rest["_ym"] - ego_y

    # =====================================================
    # FILTER 1: forward vehicles only
    # =====================================================
    mask_forward = (dx > 0) & (dx < 100.0)

    # =====================================================
    # FILTER 2: lane relevance (within ~2 lanes)
    # =====================================================
    lane_width = 3.7
    mask_lane = dy.abs() < (2.0 * lane_width)

    candidates = rest[mask_forward & mask_lane].copy()

    # =====================================================
    # SCORE: elliptical distance (matches Eq. 4)
    # =====================================================
    candidates["_score"] = ((dx.loc[candidates.index] / A_OBS)**2 +
                            (dy.loc[candidates.index] / B_OBS)**2)

    # select closest in collision sense
    candidates = candidates.nsmallest(num, "_score")

    neighbors = [_row_to_dict(candidates.iloc[i]) for i in range(len(candidates))]

    # =====================================================
    # PAD if fewer than required
    # =====================================================
    while len(neighbors) < num:
        neighbors.append({
            "x": ego_x + 80.0 + 20.0 * len(neighbors),
            "y": ego_y,
            "psi": 0.0,
            "vx": ego["vx"],
            "vy": 0.0,
            "vehicle_id": -(len(neighbors) + 1),
        })

    return neighbors


def predict_obstacles(neighbors):
    """
    Predict obstacle trajectories assuming constant velocity.

    Returns:
        obs_x, obs_y: (N * num_obs, L)
    """

    num_obs = len(neighbors)

    obs_x = torch.zeros(N * num_obs, L, device=DEVICE)
    obs_y = torch.zeros(N * num_obs, L, device=DEVICE)

    for j, n in enumerate(neighbors):
        x0, y0 = n["x"], n["y"]
        vx, vy = n["vx"], n["vy"]

        for t in range(N):
            xt = x0 + vx * t * DT
            yt = y0 + vy * t * DT

            obs_x[j * N + t, :] = xt
            obs_y[j * N + t, :] = yt

    return obs_x, obs_y


def get_state(t: int, ego_id: int = None):
    if os.path.exists(_CSV_PATH):
        return _get_state_ngsim(t, ego_id)
    return _get_state_synthetic(t)


def _get_state_ngsim(t: int, ego_id: int = None):
    _init_ngsim()

    if t >= len(_frame_index):
        raise ValueError(f"t={t} out of range ({len(_frame_index)} frames)")

    frame     = _load_frame(_frame_index[t])
    ego_row   = _pick_ego_row(frame, ego_id)

    ego       = _row_to_dict(ego_row)
    neighbors = _pick_neighbors(frame, ego, NUM_OBS)

    return ego, neighbors


def _get_state_synthetic(t: int):
    x0 = float(t) * 10.0 * DT

    ego = {
        "x": x0,
        "y": 5.5,
        "psi": 0.0,
        "vx": 10.0,
        "vy": 0.0,
        "vehicle_id": 0
    }

    neighbors = [
        {
            "x": x0 + 20 + i * 10,
            "y": [1.5, 5.5, 9.5][i % 3],
            "psi": 0.0,
            "vx": 8.0,
            "vy": 0.0,
            "vehicle_id": i + 1
        }
        for i in range(NUM_OBS)
    ]

    return ego, neighbors

def get_all_vehicles(t: int, ego_id: int = None):
    """
    Returns ALL vehicles in the frame (for visualization only).

    Output:
        ego, vehicles
    """

    if os.path.exists(_CSV_PATH):
        _init_ngsim()

        if t >= len(_frame_index):
            raise ValueError(f"t={t} out of range ({len(_frame_index)} frames)")

        frame = _load_frame(_frame_index[t])

        # pick ego (same logic as get_state)
        ego_row = _pick_ego_row(frame, ego_id)
        ego     = _row_to_dict(ego_row)

        vehicles = []
        for _, row in frame.iterrows():
            vehicles.append({
                "x": float(row[COL_X]) * _FT_TO_M,
                "y": float(row[COL_Y]) * _FT_TO_M,
                "psi": 0.0,
                "vx": float(row[COL_VEL]) * _FT_TO_M,
                "vy": 0.0,
                "vehicle_id": int(row[COL_ID]),
            })

        return ego, vehicles

    # synthetic fallback
    ego, neighbors = _get_state_synthetic(t)
    return ego, [ego] + neighbors