
"""This is a barebones generator that does the important parts of GenerateTrainData.nb. THIS IS MEANT TO FEED INTO TRAINING.PY"""

import matplotlib, sys, os, warnings, random

sys.path.append('/Users/Heal/Dropbox/Research/Experiments/Fresh-Git/')
sys.path.append('/Users/Heal/Dropbox/Research/Experiments/Fresh-Git/Utilities')
sys.path.insert(1, os.path.join(sys.path[0], 'Utilities'))
from Utilities_TDG import *
from DataGen_Parameters import *

import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
np.set_printoptions(suppress=True)
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

myuniquetime = ''.join([str(random.randint(0,9)) for i in range(6)])
print(myuniquetime)

ab   = sampS2p(nIvects, S2_scale)[:,:2]
cde  = sampR3p(nIvects, R3_scale)
l    = sampS2p(nIvects, S2_scale)
Ivec = calcIFromABCDE(l,ab,cde,(0,0))

abguess = sampS2p(npoints, S2_scale)[:,:2]
pool = mp.Pool(mp.cpu_count())

mycde = np.zeros((nIvects,npoints,3),dtype=np.float64)
for i in range(nIvects):
    mycde[i] = [pool.apply(solveKZs, args=(a,Ivec[i])).x for a in abguess]
    print("Done sampling the variety of I vector ",i+1," of ",nIvects,".")
pool.close()

Fvec = np.concatenate((np.asarray([abguess for i in range(nIvects)]),mycde),axis=2)
Fvec = np.asarray([np.concatenate(f) for f in Fvec])

Cvec = np.asarray([[evalKZs(np.concatenate([Fvec[i, (5*j):(5*j+5)], Ivec[i]]), []) for j in range(npoints)] for i in range(nIvects)]) #confidence scores

print("Fvec shape: ",Fvec.shape) # nIvects x npoints*5,    matrix
print("Ivec shape: ",Ivec.shape) # nIvects x 6,            matrix
print("Cvec shape: ",Cvec.shape) # nIvects x npoints,      matrix

np.savetxt(os.path.join(os.getcwd(), "Data/F_"+myuniquetime+".csv"), Fvec, delimiter=",")
np.savetxt(os.path.join(os.getcwd(), "Data/I_"+myuniquetime+".csv"), Ivec, delimiter=",")
np.savetxt(os.path.join(os.getcwd(), "Data/C_"+myuniquetime+".csv"), Cvec, delimiter=",")

#print("\nMY CDE:    \n",mycde)
#print("\nTRUE CDE:  \n",cde)
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#[ax.scatter(mycde[i,:,0],mycde[i,:,1],mycde[i,:,2]) for i in range(nIvects)]
#[ax.scatter(cde[i,0],cde[i,1],cde[i,2],c='k') for i in range(nIvects)]
#plt.show()
