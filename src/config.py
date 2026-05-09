"""
Global hyperparameters for the multi-modal MPC system.

Paper reference: Section IV (Implementation Details).
"""

import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
T_START = 100  # Initial time step for goal sampling in visualize.py
LANE_WIDTH = 3.7
# Logic midpoints: [1.85, 5.55, 9.25, 12.95, 16.65]
LANE_CENTRES = [LANE_WIDTH * (i + 0.5) for i in range(5)]

# ── Trajectory parameterisation (Section III-B, Eq. 6) ──────────────────────
N  = 100    # number of discrete time knots in the planning horizon
K  = 11     # polynomial degree (number of basis coefficients per axis)
DT = 0.08   # time step [s]; planning horizon T = (N-1)*DT = 2.9 s

# ── Batch size (Section III-F) ───────────────────────────────────────────────
L = 11      # number of parallel goal-directed trajectory instances

# ── Obstacle model (Section II-B, Eq. 1e) ───────────────────────────────────
NUM_OBS = 6   # number of obstacles considered per MPC step

# ── ADMM solver (Section III-D) +─────────────────────────────────────────────
MAX_ITERS  = 100    # Algorithm 1 iteration count

# Penalty weights ρ per constraint type.
# RHO_OBS must be large enough to overcome the smoothness cost — empirically
# WFtF(obs) trace ≈ 190 vs smoothness trace ≈ 2400, so RHO_OBS >= 10 is needed.
# All values kept ≤ 10 to avoid ill-conditioning the float64 KKT inversion.
RHO_OBS    = 100.0   # ρ applied to collision constraint rows
RHO_NONHOL = 1.0    # ρ applied to non-holonomic kinematic rows
RHO_INEQ   = 1.0    # ρ applied to acceleration constraint rows

# ── Cost weights ─────────────────────────────────────────────────────────────
WEIGHT_SMOOTHNESS = 2.5   # weight on ‖ẍ‖² + ‖ÿ‖² + ‖ψ̈‖² (Eq. 1a)
WEIGHT_SMOOTHNESS_PSI = 5.0
# ── Physical limits (Section II-B, Eq. 1d) ──────────────────────────────────
AMAX = 4.0    # maximum total acceleration [m/s²]
VMIN = 0.1    # minimum forward speed [m/s]
VMAX = 24.0   # maximum forward speed [m/s]
VCRUISE = 15.0 # target cruise speed [m/s] for meta-cost in "cruise" mode

# ── Elliptical safety region (Section II-B, Eq. 1e; Section IV) ─────────────
A_OBS = 2.8   # semi-axis along longitudinal direction [m] (includes ego inflation)
B_OBS = 1.5   # semi-axis along lateral direction [m]

# ── Heading filter (post-optimisation trajectory selection) ──────────────────
# Trajectories whose total heading change exceeds this threshold are discarded
# before meta-cost ranking. Prevents selection of trajectories with large
# lateral swerves that satisfy the cost but are physically unrealistic.
HEADING_LIMIT_DEG = 13.0