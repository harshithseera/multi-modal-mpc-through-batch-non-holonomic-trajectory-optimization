import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Horizon
N = 30
K = 8
DT = 0.1          # timestep [s]; planning horizon tf = N * DT = 3.0 s

# Batch size (number of goals)
L = 8

# Obstacles
NUM_OBS = 3

# MPC
MAX_ITERS = 100  # paper Fig. 2(a): ~100 iterations for residuals ~1e-3
RHO = 1.0

# Physical limits
AMAX = 3.0
VMIN = 0.1
VMAX = 20.0

# Vehicle ellipse params (Section IV: a=5.6, b=3.1 includes ego inflation)
A_OBS = 5.6
B_OBS = 3.1