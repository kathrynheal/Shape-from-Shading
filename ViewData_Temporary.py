
#combines Evaluate.py and Training.py loading parts

KHsockets = ['dhcp-10-250-168-155.harvard.edu','Kathryns-MacBook-Pro-2.local']
import matplotlib, ast, sys, time, os, socket, warnings

sys.path.insert(1, os.path.join(sys.path[0], 'Utilities'))
sys.path.insert(1, os.path.join(sys.path[0], 'Data Generation'))
from Utilities1 import *
from Utilities2 import *
from Utilities3 import *
from Utilities4 import *

if socket.gethostname() not in KHsockets: #if not KH's MBP
    matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
if socket.gethostname() in KHsockets: #if KH's MBP
    from tensorflow.python.util import deprecation
    deprecation._PRINT_DEPRECATION_WARNINGS = False
    try:
        from tensorflow.python.util import module_wrapper as deprecation2
    except ImportError:
        from tensorflow.python.util import deprecation_wrapper as deprecation2
    deprecation2._PER_MODULE_WARNING_LIMIT = 0
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.pyplot import *
from mpl_toolkits.mplot3d import *
np.set_printoptions(suppress=True)
from time import time
from numpy.matlib import repmat

print("finished loading packages")

print("****************************")
print("****************************")
print("******** training **********")
print("****************************")
print("****************************")


##DATA PARAMS
toyopt = 5 #0 is training data, #3 is a good toy case, #5 is synth folder
subset = False
size0  = 40000
ttfrac = 1
codeis = "_TOY"
uniquetime = "all" #"Synth/"#"036051_small" #"036051_large"   #"367616"#"998128"
prefix = "";            #"Dropbox/Research/Experiments/" # change this depending on whether you're running from command line, or from within the mathematica gui
uniquenum = str(time())

dataout,dataout_,dataout_t,etcout = DataLoadingMain(time(),ttfrac,toyopt,size0,subset,uniquetime)
X,   Y,   I   = dataout


scaledYtodiscriminate = etcout[4]
Y = Y/scaledYtodiscriminate
info00 = etcout[-1]
T = info00[:,1]

fout = np.real(np.concatenate((X,Y),axis=1))
#print("\nfvects after COMPRESSion:\n",fout)

#print("\n\nfvects after compression\n",np.round(fout,decimals=3))
#print("\n\nIvects after compression\n",np.round(I,decimals=3),"\n")

#print("\n\n********\n\n")
#print("fout",fout)
#print("\n\n********\n\n")

expand_I = [(1,1,0,i[0],i[1],-1) for i in I]
decompf=[]
decompI=[]
for e_i in range(20):
    ughhhh  = inv_exploit_symmetries(expand_I[e_i],fout[e_i],info00[e_i])
    decompI = np.append(decompI, np.round(np.real(ughhhh[0]),decimals=3))
    decompf = np.append(decompf, np.round(np.real(ughhhh[1]),decimals=3))
decompf = np.reshape(decompf,(20,5))
decompI = np.reshape(decompI,(20,6))

#print("\nfvects after DEcompression\n",decompf)
#print("\nIvects after DEcompression\n",decompI)

fb4c = (-0.3  ,  0.004 , 6.868, -5.454 , 8.288)
#[print(d/fb4c) for d in decompf]
#print(decompI)
