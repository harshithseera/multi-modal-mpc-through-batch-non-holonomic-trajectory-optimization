import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

N  = 30
K  = 8
DT = 0.1

L       = 8
NUM_OBS = 3

MAX_ITERS = 100

# ADMM penalty — must be large enough that the collision proximal pull
# dominates the smoothness cost. With WEIGHT_SMOOTHNESS=1 and the
# polynomial basis scaled to [0,1], RHO=100 gives strong enough penalty.
RHO        = 100.0
RHO_OBS    = RHO
RHO_NONHOL = RHO
RHO_INEQ   = RHO

WEIGHT_SMOOTHNESS = 1.0

AMAX = 4.0
VMIN = 0.1
VMAX = 15.0

A_OBS = 5.6
B_OBS = 3.1