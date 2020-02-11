
## Called by PhotograDataGenerator.py and SynthDataGenerator.py


import matplotlib, ast, sys, time, os, socket, warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)
import numpy as np
np.set_printoptions(suppress=True)
from time import time
from scipy.optimize import minimize
import random
from numpy.linalg import norm
from scipy.optimize import NonlinearConstraint
import multiprocessing as mp
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

#import os
#path = os.getcwd()
#print(path)

def makeQP(a,imsz,xybound):
    xyrange = np.linspace(-xybound,xybound,num=imsz) #num should be odd so there's a "center" pixel for 0,0
    X,Y = np.meshgrid(xyrange,xyrange)
    if len(a)==5:
        qp = a[0]*X + a[1]*Y + a[2]*X**2/2 + a[3]*X*Y + a[4]*Y**2/2
    else:
        qp = a[0] + a[1]*X + a[2]*Y + a[3]*X**2/2 + a[4]*X*Y + a[5]*Y**2/2
    return np.asarray(qp)

def getINumer(imf,l): #imf should be 5 x imsize x imsize
    n = np.stack((-imf[0],-imf[1],np.ones(imf[1].shape)))
    n = n/norm(n,axis=0)
    l = l/norm(l)
    Ivec = [[np.dot(n[:,i,j],l) for j in range(n.shape[2])] for i in range(n.shape[1])]
    return np.asarray(Ivec)

def gaussD(im,ord,bnd): #im is a SQUARE matrix. ord is 5 or 6.
    
    
    
    
    sig = 4
    
    
    
    
    m = 'mirror'
    scl = (len(im)-1)/(2*bnd)
    print("scl is: ",scl)
    #first axis here is the matrix row#, so it's "y"
    #second axis here is the matrix col#, so it's "x"
    dx  = scl*gaussian_filter(im,sigma=sig,order=(0,1),mode=m)
    dy  = scl*gaussian_filter(im,sigma=sig,order=(1,0),mode=m)
    dxx = scl*gaussian_filter(dx,sigma=sig,order=(0,1),mode=m)
    dxy = scl*gaussian_filter(dx,sigma=sig,order=(1,0),mode=m)
    dyy = scl*gaussian_filter(dy,sigma=sig,order=(1,0),mode=m)
    if ord==6:
        return np.asarray((im,dx,dy,dxx,dxy,dyy)) 
    return np.asarray((dx,dy,dxx,dxy,dyy))
    
