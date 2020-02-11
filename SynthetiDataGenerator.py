


## this is a barebones generator that does the important parts of GenerateTrainData.nb
## THIS IS MEANT TO FEED INTO CVPR_FIGURES.PY

import matplotlib, ast, sys, time, os, socket, warnings
from Utilities  import *
from Utilities2 import *
sys.path.append('/Users/Heal/Dropbox/Research/Experiments/NN/')
from HelperFunctions import *
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

prefix  = "/Users/Heal/Dropbox/Research/Experiments/"
lightf  = "Data/light_directions.txt"
surfcf  = "Data/F_029027.csv"
npoints = 40
plotson = False
printon = True
#########################################
##### BEGIN OBTAINING THE I VECTORS #####

L = np.transpose(np.loadtxt(prefix+lightf))
F = np.asarray([[float(g) for g in h.split(',')] for h in np.loadtxt(prefix+surfcf,dtype='str')])
F = np.reshape(F,(F.shape[0],int(F.shape[1]/5),5))
if printon:
    print("all l shape ",L.shape)
    print("all f shape ",F.shape)

##customizable
imsize = 101 #needs to be odd so there can be a "center" pixel for (0,0)
imbnds = .1
px = (0,0)
f = repmat(F[40,30],20,1)
l = L
if printon:
    print("\nthese l \n",l)
    print("these f \n",f)

Ivec = np.asarray([calcIFromABCDE([l[i]],[f[i,:2]],[f[i,2:]],px) for i in range(20)])[:,0]

imf = makeQP(f[0],imsize,imbnds)
if plotson:
    fig = plt.figure()
    im = plt.imshow(imf)
    fig.colorbar(im, shrink=0.5, aspect=5)
    plt.title("surface patch")

##check gaussian derivatives match with true derivatives.
fimgGD = gaussD(imf,5,imbnds)
fvecGD = fimgGD[:,int((imsize-1)/2),int((imsize-1)/2)]
if printon:
    print("\n\ngd result:   ", fvecGD)
    print("real vector: ",     f[0])
    print("\n\nratio:       ", fvecGD/f[0])
    
#print("this should be constant...\n")
#print(fimgGD[4])
#fig = plt.figure()
#im = plt.imshow(np.log(fimgGD[4]))
#fig.colorbar(im, shrink=0.5, aspect=5)
#plt.title("f_xx")
#plt.show()


Iholder = np.zeros((20,6),dtype=np.float)
for thisl in range(20):

    imI = makeQP(Ivec[thisl],imsize,imbnds)
    if printon:
        print("\n\nimI(0,0) is:  ", imI[int((imsize-1)/2),int((imsize-1)/2)])
    imI[imI<0]=0
    if plotson:
        fig = plt.figure()
        im = plt.imshow(imI)
        fig.colorbar(im, shrink=0.5, aspect=5)
        plt.title("image patch")

    Inum = getINumer(fimgGD,l[thisl])
    if printon:
        print("Inum(0,0) is: ", Inum[int((imsize-1)/2),int((imsize-1)/2)])
    Inum[Inum<0]=0
    if plotson:
        fig = plt.figure()
        im = plt.imshow(Inum)
        fig.colorbar(im, shrink=0.5, aspect=5)
        plt.title("image patch -- numerically")

    IimgGD = gaussD(Inum,6,imbnds)
    IvecGD = IimgGD[:,int((imsize-1)/2),int((imsize-1)/2)]
    if printon:
        print("\n\nIvec(0,0) is: ",IvecGD)
        print("Iana(0,0) is:  ", Ivec[thisl])
        
    if plotson:
        plt.show()

    Iholder[thisl] = IvecGD

if printon:
    print("\n\nratio:       \n", Ivec/Iholder)


print("\n\n")
#print("\n\n",Iholder)
#print("\n\n",f)

#### END OBTAINING THE I VECTORS #####
######################################

abguess = sampS2p(npoints)[:,:2]
pool = mp.Pool(mp.cpu_count())
mycde = np.zeros((20,npoints,3),dtype=np.float64)
for i in range(20):
    mycde[i] = [pool.apply(solveKZs, args=(a,Iholder[i])).x for a in abguess]
    print("Done sampling the variety of I vector ",i+1," of ",20,".")
pool.close()

np.savetxt(prefix+"Data/Synth/L.csv", l,       delimiter=",")
np.savetxt(prefix+"Data/Synth/F.csv", f,       delimiter=",")
np.savetxt(prefix+"Data/Synth/I.csv", Iholder, delimiter=",")

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
[ax.scatter(mycde[i,:,0],mycde[i,:,1],mycde[i,:,2]) for i in range(20)]
[ax.scatter(f[i,2],f[i,3],f[i,4],c='k') for i in range(20)]
pltbnd=2
ax.set_xlim(f[i,2]-pltbnd,f[i,2]+pltbnd)
ax.set_ylim(f[i,3]-pltbnd,f[i,3]+pltbnd)
ax.set_zlim(f[i,4]-pltbnd,f[i,4]+pltbnd)
plt.show()

