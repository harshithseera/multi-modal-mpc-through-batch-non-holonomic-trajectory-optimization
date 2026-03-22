"""
Global hyperparameters for the multi-modal MPC system.

Paper reference: Section IV (Implementation Details).
"""

import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Trajectory parameterisation (Section III-B, Eq. 6) ──────────────────────
N  = 30    # number of discrete time knots in the planning horizon
K  = 8     # polynomial degree (number of basis coefficients per axis)
DT = 0.1   # time step [s]; planning horizon T = (N-1)*DT = 2.9 s

# ── Batch size (Section III-F) ───────────────────────────────────────────────
L = 8      # number of parallel goal-directed trajectory instances

# ── Obstacle model (Section II-B, Eq. 1e) ───────────────────────────────────
NUM_OBS = 6   # number of obstacles considered per MPC step

# ── ADMM solver (Section III-D) ─────────────────────────────────────────────
MAX_ITERS  = 100    # Algorithm 1 iteration count

# Penalty weights ρ per constraint type.
# RHO_OBS must be large enough to overcome the smoothness cost — empirically
# WFtF(obs) trace ≈ 190 vs smoothness trace ≈ 2400, so RHO_OBS >= 10 is needed.
# All values kept ≤ 10 to avoid ill-conditioning the float64 KKT inversion.
RHO_OBS    = 10.0   # ρ applied to collision constraint rows
RHO_NONHOL = 1.0    # ρ applied to non-holonomic kinematic rows
RHO_INEQ   = 1.0    # ρ applied to acceleration constraint rows

# ── Cost weights ─────────────────────────────────────────────────────────────
WEIGHT_SMOOTHNESS = 1.0   # weight on ‖ẍ‖² + ‖ÿ‖² + ‖ψ̈‖² (Eq. 1a)

# ── Physical limits (Section II-B, Eq. 1d) ──────────────────────────────────
AMAX = 4.0    # maximum total acceleration [m/s²]
VMIN = 0.1    # minimum forward speed [m/s]
VMAX = 15.0   # maximum forward speed [m/s]

# ── Elliptical safety region (Section II-B, Eq. 1e; Section IV) ─────────────
A_OBS = 5.6   # semi-axis along longitudinal direction [m] (includes ego inflation)
B_OBS = 3.1   # semi-axis along lateral direction [m]

# ── Heading filter (post-optimisation trajectory selection) ──────────────────
# Trajectories whose total heading change exceeds this threshold are discarded
# before meta-cost ranking. Prevents selection of trajectories with large
# lateral swerves that satisfy the cost but are physically unrealistic.
HEADING_LIMIT_DEG = 13.0