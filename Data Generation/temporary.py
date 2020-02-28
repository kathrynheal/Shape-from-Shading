## THIS IS MEANT TO FEED INTO EVALUATE.PY


import matplotlib, ast, sys, time, os, socket, warnings
from Utilities3 import *
from Utilities2 import *
sys.path.append('/Users/Heal/Dropbox/Research/Experiments/Git/')
from Utilities1 import *

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
from scipy import ndimage
import imageio
from sklearn.preprocessing import minmax_scale


#### PARAMETERS FOR RUN
prefix = "/Users/Heal/Dropbox/Research/Experiments/"
dataset = "029027"
animal = "cat"
#datafname = "/Users/Heal/Dropbox/Research/XiongZickler2014_data/" + animal + "/result.mat"

##I GOT THE CAT.MAT FILE FROM JIALIANG RUNNING YING'S CODE IN MATLAB
datafname = "/Users/Heal/Dropbox/Research/Experiments/Git/Data Generation/cat.mat"
#datalname = "/Users/Heal/Dropbox/Research/XiongZickler2014_data/" + animal + "/refined_light.txt"


def to01(NtL):
    NtL = np.nan_to_num(NtL - np.nanmin(NtL))
    return NtL/np.max(NtL)

def valid(f): #mask out the NaNs -- set them to zero.
    return np.nan_to_num(np.multiply(m, f))
    
def print1(NtL,NtLY,I):
    fig, axs = plt.subplots(1,2, sharey=True, tight_layout=True)
    imi = axs[0].imshow(NtL,origin='lower',vmin=0,vmax=1)
    axs[0].title.set_text('GD-Normal Image')
    fig.colorbar(imi, ax=axs[0], shrink=0.5)
    imi = axs[1].imshow(  I,origin='lower',vmin=0,vmax=1)
    axs[1].title.set_text('Photo')
    fig.colorbar(imi, ax=axs[1], shrink=0.5)

    fig, axs = plt.subplots(1,2, sharey=True, tight_layout=True)
    imi = axs[0].imshow(NtLY,origin='lower',vmin=0,vmax=1)
    axs[0].title.set_text("Ying-Normal Image")
    fig.colorbar(imi, ax=axs[0], shrink=0.5)
    imi = axs[1].imshow(   I,origin='lower',vmin=0,vmax=1)
    axs[1].title.set_text("Photo.")
    fig.colorbar(imi, ax=axs[1], shrink=0.5)

def print2(myI,I):
    fig, axs = plt.subplots(1,2, sharey=True, tight_layout=True)
    imi = axs[0].imshow(I,  origin='lower',cmap='gray',vmin=0,vmax=1.2)
    axs[0].title.set_text('photog from JW')
    fig.colorbar(imi, ax=axs[0], shrink=0.5)
    imi = axs[1].imshow(myI,origin='lower',cmap='gray',vmin=0,vmax=1.2)
    axs[1].title.set_text('rho*N*L from JW')
    fig.colorbar(imi, ax=axs[1], shrink=0.5)
    
def print3(gdNtL,sImat):
    fig,axs = plt.subplots(2,6, sharey=True, tight_layout=True,figsize=(20,8))
    fig.suptitle("GAUSSIAN (TOP) vs ANALYTIC (BOTTOM) DERIVATIVES", fontsize=16)
    labels = ["I","Ix","Iy","Ixx","Ixy","Iyy"]
    for i in [0,1,2,3,4,5]:

        imi = axs[0,i].imshow(gdNtL[i],origin='lower',vmin=-.0005,vmax=.0005)
        axs[0,i].title.set_text(labels[i]+' values')
        if i==0:
            fig.colorbar(imi, ax=axs[0,i], shrink=0.5)

        imi = axs[1,i].imshow(sImat[i],origin='lower',vmin=-.0005,vmax=.0005)
        axs[1,i].title.set_text(labels[i]+' values')
        if i==0:
            fig.colorbar(imi, ax=axs[1,i], shrink=0.5)

def print4(gdNtL,sImat):
    fig = plt.figure()
    plt.imshow(gdNtL[3]/sImat[3],origin='lower',vmin=-10,vmax=10)
    fig = plt.figure()
    plt.imshow(gdNtL[5]/sImat[5],origin='lower',vmin=-10,vmax=10)
    plt.show()


#### SET PATCH SPECS
patwid   = 1       #length of patch, w.r.t. some ground-truth units
sig      = 1          #variance for gaussian derivatives
patchsz  = 50     #num pixels of patch (a function of patwid & resolution) #351
subset = np.linspace(250,250+patchsz-1,num=patchsz,dtype=np.int)
patchy,patchx = np.meshgrid(subset,subset) #each is patchsz x patchsz
lc = '04'


#### LOAD PHYSICAL MEASUREMENTS
alldata = loadmat(datafname)
#print("Keys in file: ",alldata.keys())

I =alldata['I'][:,:,int(lc)][patchx,patchy]
Z =alldata['Z'][patchx,patchy]
N =alldata['n'][patchx,patchy]
r =alldata['rho'][patchx,patchy]
L =alldata['L'][:,int(lc)]
m =alldata['shadow_mask'][:,:,int(lc)][patchx,patchy]

myI = np.nan_to_num(np.multiply(r,np.dot(N,L)))
myI = valid(myI)

print2(myI,I)

#### GET GAUSSIAN DERIVATIVES FOR SURFACE
scl = 1#(len(Z)-1)/(2*patwid)
gdZ = gaussD(Z,5,patwid,sig,scl) ##get gaussian surface derivatives
normz = np.sqrt(gdZ[0]**2+gdZ[1]**2+1)
gdNab1 = np.transpose(np.asarray([-gdZ[1]/normz,-gdZ[0]/normz,np.ones(gdZ[0].shape)/normz]),(1,2,0))

#### SANITY CHECK: does our image NtL match Ying's I?
NtL  = valid(np.multiply(r,np.dot(gdNab1,L))) ## n IS normalized
NtLY = valid(np.multiply(r,np.dot(N,L))) ## n IS normalized

print1(NtL,NtLY,I)

sImat = [[symbI( L ,gdZ[:2,i,j],gdZ[2:,i,j]) for j in range(patchsz)] for i in range(patchsz)]
sImat = np.transpose(np.asarray(sImat),(2,0,1))
sImat = np.asarray([valid(s) for s in sImat])

gdNtL = gaussD(NtL,6,patwid,sig,scl)
gdNtL = np.asarray([valid(s) for s in gdNtL])

print3(gdNtL,sImat)

