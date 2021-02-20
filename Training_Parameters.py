"""This gets read by Training.py"""


import numpy as np

## TRAINING NETWORK PARAMS
numIters = np.floor(1e4).astype(int)  # 1e4
# depth_h = 0  # don't change this! lots of dependencies on it
width_h = 20   # truly this needs to be about 500
width_g = 15   # maybe like 25
depth_g = 1
batch_size = 32
iterprintinterval = max(1,np.ceil(numIters/100))
animate_on  = False
plots_on    = False
log_on      = False
ws_on, ws_dir = True, "1613292657.9177542"  # warm_start for network
