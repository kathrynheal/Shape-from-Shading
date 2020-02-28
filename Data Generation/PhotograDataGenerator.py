



## THIS IS MEANT TO FEED INTO EVALUATE.PY


import matplotlib, ast, sys, time, os, socket, warnings
from Utilities3 import *
from Utilities2 import *
sys.path.append('/Users/Heal/Dropbox/Research/Experiments/Git/')
from Utilities1 import *
from PrintFuncs import *

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)
import numpy as np
from numpy.linalg import norm
from numpy.matlib import repmat
np.set_printoptions(suppress=True)
from time import time
import random
import multiprocessing as mp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.io import loadmat
from mpl_toolkits.axes_grid1.colorbar import colorbar
from matplotlib import gridspec
from scipy.ndimage import gaussian_filter






#### PARAMETERS FOR RUN
prefix = "/Users/Heal/Dropbox/Research/Experiments/"
dataset = "029027"
animal = "pig"
#datafname = "/Users/Heal/Dropbox/Research/XiongZickler2014_data/" + animal + "/result.mat"
datafname = "/Users/Heal/Dropbox/Research/Experiments/Git/Data Generation/"+animal+".mat"

UnitTest =  True

PlotsOn  =  False
PlotsOn0 =  False
PlotsOn2 =  True # the 6x6 grids
PlotsOn3 =  False
PlotsOn4 =  False
PlotsAny =  np.any([PlotsOn,PlotsOn0,PlotsOn2,PlotsOn3,PlotsOn4])

def valid(f): #mask out the NaNs -- set them to zero.
    return np.nan_to_num(np.multiply(M, f))





#### SET PATCH SPECS
patwid   = 1       #length of patch, w.r.t. some ground-truth units
sig      = 1       #variance for gaussian derivatives
patchsz  = 150     #num pixels of patch (a function of patwid & resolution) #351
subset = np.linspace(150,150+patchsz-1,num=patchsz,dtype=np.int)
patchy,patchx = np.meshgrid(subset,subset) #each is patchsz x patchsz
lc       = '04'





#### LOAD PHYSICAL MEASUREMENTS
alldata = loadmat(datafname)

I =alldata['I'][:,:,int(lc)][patchx,patchy]
Z =alldata['Z'][patchx,patchy]
N =alldata['n'][patchx,patchy]
r =alldata['rho'][patchx,patchy]
L =alldata['L'][:,int(lc)]
M =alldata['shadow_mask'][:,:,int(lc)][patchx,patchy]

myI = valid(np.nan_to_num(np.multiply(r,np.dot(N,L))))

print1(PlotsOn0,I,myI)

if UnitTest:
    X,Y = np.linspace(-patchsz/2,patchsz/2,patchsz),np.linspace(-patchsz/2,patchsz/2,patchsz)
    XX,YY = np.meshgrid(X,Y)

    Z   = (XX**3+YY**3)/1000
    dZx = 3*XX**2/1000
    dZy = 3*YY**2/1000
    dZxx = 6*XX/1000
    dZxy = 0*np.ones(XX.shape)
    dZyy = 6*YY/1000

#    Z   = (XX**2+YY**2)/100
#    dZx = (2*XX)/100
#    dZy = (2*YY)/100
#    dZxx = 2*np.ones(XX.shape)/100
#    dZxy = 0*np.ones(XX.shape)
#    dZyy = 2*np.ones(XX.shape)/100
    
#    Z   = (XX+YY**2)/100
#    dZx = np.ones(XX.shape)/100
#    dZy = (2*YY)/100
#    dZxx = 0*np.ones(XX.shape)/100
#    dZxy = 0*np.ones(XX.shape)
#    dZyy = 2*np.ones(XX.shape)/100


    normn = np.sqrt(1+dZx**2+dZy**2)
    N = np.transpose(np.asarray([dZy,dZx,np.ones(Z.shape)])/normn,(1,2,0))





#### GET GAUSSIAN DERIVATIVES FOR SURFACE
scl = 1  #(len(Z)-1)/(2*patwid)
gdZ = gaussD(Z,5,patwid,sig,scl) ##get gaussian surface derivatives

normz = np.sqrt(gdZ[0]**2+gdZ[1]**2+1)
gdNab1 = np.transpose(np.asarray([-gdZ[1]/normz,-gdZ[0]/normz,np.ones(gdZ[0].shape)/normz]),(1,2,0))


if UnitTest: print2(PlotsOn,dZx,dZy,dZxx,dZxy,dZyy,gdZ,UnitTest)
#unit test cheating: using exact derivs instead of gaussian derivs.
if UnitTest: gdZ = np.asarray([dZx,dZy,dZxx,dZxy,dZyy])





#### SANITY CHECK: do our fx,fy in gDZ match Ying's N?
#print("Keys in file: ",alldata.keys())
N[:,:,(1,0)] = N[:,:,(0,1)] if UnitTest else -N[:,:,(0,1)] #Ying's convention is y,x and negative
Na,  Nb      =  scl*N[:,:,0]/N[:,:,2],  scl*N[:,:,1]/N[:,:,2]
normn = np.sqrt(Na**2+Nb**2+1)
Nab1 = np.transpose(np.asarray([Na/normn,Nb/normn,np.ones(Na.shape)/normn]),(1,2,0))

#print3(PlotsOn,gdNab1,Nab1)
#print4(PlotsOn,gdZ,scl,Na,Nb,normz)





#### SANITY CHECK: does our image NtL match Ying's I?
NtL = np.dot(gdNab1,L) ## n IS normalized

#print5(PlotsOn,gdNab1,N,L,NtL,UnitTest)
print6(PlotsOn,gdZ) # do our surface derivatives look reasonable?





#### GET GAUSSIAN DERIVATIVES FOR IMAGE (GIVEN N MEASUREMENTS)
gdNtL = gaussD(NtL,6,patwid,sig,scl)
gdNtL = np.asarray([valid(s) for s in gdNtL])

print7(PlotsOn4,gdNtL)





#### GET ANALYTIC DERIVATIVES FOR IMAGE (GIVEN gdZ MEASUREMENTS)
sImat = [[symbI( L ,gdZ[:2,i,j],gdZ[2:,i,j]) for j in range(patchsz)] for i in range(patchsz)]
sImat = np.transpose(np.asarray(sImat),(2,0,1))
sImat = np.asarray([valid(s) for s in sImat])

print8(PlotsOn,sImat,gdNtL)
print9(PlotsOn,gdZ,sImat)






#### EXPERIMENTING WITH MULTIPLE GAUSSIAN DERIVATIVES
sigs = [0,1]
gdNtLmult = [gaussD(NtL,6,patwid,s,scl) for s in sigs] #if diff sigmas on NtL

viewsz0,vert = 0.0025,40
print10(PlotsOn2,patchsz,sigs,sImat,gdNtLmult,viewsz0)
print11(PlotsOn2,sImat,sigs,vert,viewsz0,gdNtLmult)
print14(PlotsOn2,sImat)
if UnitTest: print12(PlotsOn,XX,YY,sImat,gdNtL)






##### SANITY CHECK: Look at a little sub-sample.
            ##in the following, make sure you're not picking any edge pixels
            ##since GDs will give them zeroes and give you an error in explot_symm
gdNtL2 = np.concatenate(np.transpose(valid(gdNtL),(1,2,0)))
gdZ2 = np.concatenate(np.transpose(valid(gdZ),(1,2,0)))
gdZ2 = np.asarray([pponly(g) for g in gdZ2]) ##get Ying's pos-pos equiv of f



#######
# TODO: handle when gdZ2 has 00000 values from pponly...
#######



sImat2 = np.concatenate(np.transpose(valid(sImat),(1,2,0)))


#plt.figure()
#plt.imshow(sImat[5,10:-10,10:-10],origin='lower')
#plt.colorbar()
#plt.figure()
#plt.imshow(gdNtL[5,10:-10,10:-10],origin='lower')
#plt.colorbar()
#plt.show()



##### COMPRESS I VECTORS
IFT = np.asarray([exploit_symmetries(i,j,[],False) for i,j in zip(gdNtL2,gdZ2)]) #doesn't work
#IFT = np.asarray([exploit_symmetries(i,j,[],False) for i,j in zip(sImat2,gdZ2)]) #works
I21,F21,T21 = IFT[:,0],IFT[:,1],IFT[:,2]
I20,F20,T20 = np.stack(I21),np.stack(F21),np.stack(T21)
keepsmallI2 = np.where(np.amax(np.abs(I20[:,3:5]),axis=1)<40)
I2,F2,T2 = I20[keepsmallI2],F20[keepsmallI2],T20[keepsmallI2]
#with np.printoptions(precision=3, suppress=True):
#    print("\n\n I2 is:\n",np.stack(I21),"\n\n")






##### SANITY CHECK: IS THE TOTALLY-ANALYTIC VERSION OF TILDE-IYY ALWAYS NEGATIVE?
TIyymat = np.asarray([[ItoTIyy(sImat[:,i,j]) for j in range(patchsz)] for i in range(patchsz)])
print("IF THIS IS NOT POSITIVE Tilde IYY IS NEVER POSITIVE: ",np.nanmax(TIyymat.ravel()))

print13(PlotsOn4,TIyymat,gdZ)

TIyygdN = np.asarray([[ItoTIyy(gdNtL[:,i,j]) for j in range(patchsz)] for i in range(patchsz)])


##troubleshooting statistics
cubicnessf = FindThirdOrder(Z,gdZ,sig)
deviationI = gdNtL2[:,5]-sImat2[:,5]
#colr = I2[:,5]
#colr = cubicnessf.ravel()[keepsmallI2]
colr = deviationI[keepsmallI2]

if PlotsOn2:

    plt.figure()
    plt.imshow(cubicnessf,origin='lower')#,vmin=0,vmax=.00001)
    plt.title("'Cubicness' of surface f.")
    plt.colorbar()

#    plt.figure()
#    plt.scatter(I2[:,3],I2[:,4],c=colr)
#    plt.title("tIxx vs tIxy values, colored by aI,gdI deviation")
#    plt.colorbar()
#
#    plt.figure()
#    plt.scatter(I2[:,5],colr,c=colr)
#    plt.colorbar()
#
#    plt.figure()
#    plt.imshow(np.reshape(I20[:,5],TIyymat.shape),vmin=-40,vmax=40)
#    plt.title("Exploit-Symm-tIyy using Gaussian I")
#    plt.colorbar()
#
#    plt.figure()
#    plt.imshow(TIyymat,vmin=-.0000001,vmax=.0000001)
#    plt.title("Analytic-tIyy using Analytic I: see, always negative.")
#    plt.colorbar()
#
#    plt.figure()
#    plt.imshow(TIyygdN,vmin=-.000002,vmax=.000002)
#    plt.title("Analytic-tIyy using Gaussian I")
#    plt.colorbar()



if PlotsAny:
    plt.show()
