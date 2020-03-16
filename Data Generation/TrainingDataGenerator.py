


## this is a barebones generator that does the important parts of GenerateTrainData.nb
## THIS IS MEANT TO FEED INTO TRAINING.PY

import matplotlib, ast, sys, time, os, socket, warnings
from Utilities3 import *
from Utilities2 import *
sys.path.append('/Users/Heal/Dropbox/Research/Experiments/Git/')
from Utilities1 import *
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)
import numpy as np
from numpy.matlib import repmat
np.set_printoptions(suppress=True)
from time import time
import random
import multiprocessing as mp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

prefix = "~/Dropbox/Research/Experiments/"

myuniquetime = ''.join([str(random.randint(0,9)) for i in range(6)])
print(myuniquetime)

nIvects = 40
npoints = 40

ab   = sampS2p(nIvects)[:,:2]
cde  = sampR3p(nIvects)
l    = sampS2p(nIvects)
Ivec = calcIFromABCDE(l,ab,cde,(0,0))

abguess = sampS2p(npoints)[:,:2]
pool = mp.Pool(mp.cpu_count())

mycde = np.zeros((nIvects,npoints,3),dtype=np.float64)
for i in range(nIvects):
    mycde[i] = [pool.apply(solveKZs, args=(a,Ivec[i])).x for a in abguess]
    print("Done sampling the variety of I vector ",i+1," of ",nIvects,".")
pool.close()

Fvec = np.concatenate((np.asarray([abguess for i in range(nIvects)]),mycde),axis=2)
Fvec = np.asarray([np.concatenate(f) for f in Fvec])

Cvec = np.asarray([[evalKZs(np.concatenate([Fvec[i,(5*j):(5*j+5)],Ivec[i]]),[]) for j in range(npoints)] for i in range(nIvects)]) #confidence scores

print("Fvec shape: ",Fvec.shape) # nIvects x npoints*5,    matrix
print("Ivec shape: ",Ivec.shape) # nIvects x 6,            matrix
print("Cvec shape: ",Cvec.shape) # nIvects x npoints,      matrix

np.savetxt("Data/F_"+myuniquetime+".csv", Fvec, delimiter=",")
np.savetxt("Data/I_"+myuniquetime+".csv", Ivec, delimiter=",")
np.savetxt("Data/C_"+myuniquetime+".csv", Cvec, delimiter=",")




#print("\nMY CDE:    \n",mycde)
#print("\nTRUE CDE:  \n",cde)
#
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#[ax.scatter(mycde[i,:,0],mycde[i,:,1],mycde[i,:,2]) for i in range(nIvects)]
#[ax.scatter(cde[i,0],cde[i,1],cde[i,2],c='k') for i in range(nIvects)]
#plt.show()
