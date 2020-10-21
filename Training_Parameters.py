"""This gets read by Training.py"""


import numpy as np

## TRAINING NETWORK PARAMS
numIters = np.floor(1e5).astype(int)  # 1e4
stepsize = .05
    # .01 for size0=40000, w_h=50,w_g=25,d_g=25
# depth_h = 0
    # don't change this! there are a bunch of dependencies below
width_h = 50  # 50
    #truly this needs to be about 500
width_g = 25  # 25
depth_g = 1
batch_size = 10000
runtype = "ONESTEP"
iterprintinterval = max(1,np.ceil(numIters/100))
animate_on  = True
plots_on    = False
log_on      = True
Adamb1 = .99  # for TF optimizer. default term 0.99
ws_on, ws_dir = False, "1585068292.6331022"
    # warm_start for network
