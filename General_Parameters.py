"""This gets read by Training.py and Evaluate.py"""


import numpy as np

##DATA PARAMS
dataset_id = 0
    # 0 is TRAINING.
    # 1 is PHOTOMETRIC.
    # 2 is COQUADRATIC.
subset = True
size0  = 1000
ttfrac = 0.9
codeis = "kh"
uniquetime = "all"
prefix = "";
    #"Dropbox/Research/Experiments/" # change depending on if running from command line, or from within the mathematica gui

## Params for inside DataLoadingMain
verbose  = False
pruning1 = False  # only executes if dataset_id==0
pruning2 = False  # only executes if dataset_id==0
pruning3 = True   # only executes if dataset_id==0
loadflags = verbose, pruning1, pruning2, pruning3

scale_discrim = 10  # scale Y up/down so it's easier to resolve.
## Nicole, this is the parameter to focus on for scale-invariance.
