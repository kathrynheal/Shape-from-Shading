



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



#### PARAMETERS FOR RUN
prefix = "/Users/Heal/Dropbox/Research/Experiments/"
dataset = "029027"
animal = "pig"
datafname = "/Users/Heal/Dropbox/Research/XiongZickler2014_data/" + animal + "/result.mat"
focus =     True
PlotsOn =   False
PlotsOn2 =  True



#### SET PATCH SPECS
patwid = 1       #length of patch, w.r.t. some ground-truth units
sig = 1          #variance for gaussian derivatives
patchsz = 351    #num pixels of patch (a function of patwid & resolution)
subset = np.linspace(150,150+patchsz-1,num=patchsz,dtype=np.int)
patchx,patchy = np.meshgrid(subset,subset) #each is patchsz x patchsz




#### LOAD PHYSICAL MEASUREMENTS
alldata = loadmat(datafname)
Z =  alldata['Z'][patchx,patchy]
N =  alldata['n'][patchx,patchy]        #unit-length vectors
L =  alldata['L'][:,4]                  #to negate or not to negate???




#### GET GAUSSIAN DERIVATIVES FOR SURFACE
scl = 1 #(len(Z)-1)/(2*patwid)
gdZ = gaussD(Z,5,patwid,sig,scl) ##get gaussian surface derivatives




#### SANITY CHECK: do our fx,fy in gDZ match Ying's N?
#print("Keys in file: ",alldata.keys())
N[:,:,(1,0)] = -N[:,:,(0,1)] #Ying's convention is y,x and negative
Na,  Nb      =  scl*N[:,:,0]/N[:,:,2],  scl*N[:,:,1]/N[:,:,2]
normz = np.sqrt(Na**2+Nb**2+1)
Nab1 = np.transpose(np.asarray([Na/normz,Nb/normz,np.ones(Na.shape)/normz]),(1,2,0))
gdZab1 = np.transpose(np.asarray([gdZ[0],gdZ[1]/normz,np.ones(Na.shape)]),(1,2,0))

if PlotsOn:
    fig, axs = plt.subplots(2,2, sharey=True, tight_layout=True)
    imi = axs[0,0].imshow(gdZ[0],origin='lower',vmin=-3*scl,vmax=3*scl)
    axs[0,0].title.set_text('my dfx')
    fig.colorbar(imi, ax=axs[0,0], shrink=0.5)
    imi = axs[0,1].imshow(Na/normz,origin='lower',vmin=-3*scl,vmax=3*scl)
    axs[0,1].title.set_text('Ying dfx * scale')
    fig.colorbar(imi, ax=axs[0,1], shrink=0.5)
    imi = axs[1,0].imshow(gdZ[1],origin='lower',vmin=-3*scl,vmax=3*scl)
    axs[1,0].title.set_text('my dfy')
    fig.colorbar(imi, ax=axs[1,0], shrink=0.5)
    imi = axs[1,1].imshow(Nb/normz,origin='lower',vmin=-3*scl,vmax=3*scl)
    axs[1,1].title.set_text('Ying dfy * scale')
    fig.colorbar(imi, ax=axs[1,1], shrink=0.5)



#### SANITY CHECK: does our image NtL match Ying's I?
NtL = np.dot(Nab1,L) ## n is not normalized
if PlotsOn:
    fig, axs = plt.subplots(1,3, sharey=True, tight_layout=True, figsize=(15,5))
    
    nim = axs[0].imshow(N,origin='lower')
    axs[0].title.set_text('Normal Vector Field')
    
    yntlim =axs[1].imshow(np.dot(N,L),origin='lower',cmap='gray',vmax=1)
    axs[1].title.set_text("Ying's N, dot Ying's L")
    fig.colorbar(yntlim, ax=axs[1], shrink=0.5)

    ## TODO: WHY IS THIS NEGATIVE NtL???
    ntlim=axs[2].imshow( NtL ,origin='lower',cmap='gray')
    axs[2].title.set_text('NtL (Image w/o albedo)')
    fig.colorbar(ntlim, ax=axs[2], shrink=0.5)
    
    

#### SANITY CHECK: do our surface derivatives look reasonable?
if PlotsOn:
    fig,axs = plt.subplots(1,5, sharey=True, tight_layout=True,figsize=(15,5))
    labels = ["fx","fy","fxx","fxy","fyy"]
    for i in [0,1,2,3,4]:
        imi = axs[i].imshow(gdZ[i],origin='lower')
        axs[i].title.set_text(labels[i]+' values')
        fig.colorbar(imi, ax=axs[i], shrink=0.5)



#### GET GAUSSIAN DERIVATIVES FOR IMAGE (GIVEN N MEASUREMENTS)
gdNtL = gaussD(NtL,6,patwid,sig,scl)
if PlotsOn:#why the heck are we seeing negative tilde Iyy???
    fig = plt.figure()
    tIyy = ItoTIyy(gdNtL)
    imIyy = plt.imshow(tIyy,origin='lower',vmin=-.00001,vmax=.00001)#-.000001,vmax=.000001)
    plt.title('Sign(tilde Iyy) for GD I')
    fig.colorbar(imIyy, shrink=0.5)

    fig = plt.figure()
    tIyy = gdNtL[0]
    imIyy = plt.imshow(np.sign(tIyy),origin='lower')
    plt.title('Sign(I): Yellow is +')
    fig.colorbar(imIyy, shrink=0.5)

    fig = plt.figure()
    tIyy = gdNtL[1]**2*gdNtL[5] + gdNtL[2]**2*gdNtL[3]
    imIyy = plt.imshow(tIyy,origin='lower')
    plt.title('pos component: Yellow is +')
    fig.colorbar(imIyy, shrink=0.5)

    fig = plt.figure()
    tIyy = - 2*gdNtL[4]*gdNtL[1]*gdNtL[2]
    imIyy = plt.imshow(tIyy,origin='lower')
    plt.title('neg component: Yellow is +')
    fig.colorbar(imIyy, shrink=0.5)
    


#### GET ANALYTIC DERIVATIVES FOR IMAGE (GIVEN gdZ MEASUREMENTS)
sImat = [[symbI( L ,gdZ[:2,i,j],gdZ[2:,i,j]) for j in range(patchsz)] for i in range(patchsz)]
sImat = np.transpose(np.asarray(sImat),(2,0,1))
#with np.printoptions(precision=3, suppress=True):
#    sI = np.transpose(np.asarray(symbI( L ,gdZ[:2,:,100],gdZ[2:,:,100]))) #NEGATIVE L???
#    print(np.sort(np.asarray([ItoTIyy(s) for s in sI])[sI[:,0]>0]))
#    print("")
#print("\n\nMy analytic N dot Ying's L:\n",sImat[0])
if PlotsOn:
    fig, axs = plt.subplots(1,2, sharey=True, tight_layout=True)
    nim = axs[0].imshow(sImat[0],origin='lower',cmap='gray',vmin= -0.1,vmax= 1)
    axs[0].title.set_text('Compare analytic I')
    fig.colorbar(nim, ax=axs[0], shrink=0.5)
    ntlim=axs[1].imshow(gdNtL[0],origin='lower',cmap='gray',vmin= -0.1,vmax= 1)
    axs[1].title.set_text('to Gaussian derivative I')
    fig.colorbar(ntlim, ax=axs[1], shrink=0.5)

    fig, axs = plt.subplots(5,2, sharey=True, tight_layout=True,figsize=(8,8))
    for i in [1,2,3,4,5]:
        nim = axs[i-1,0].imshow(sImat[i],origin='lower',cmap='gray',vmin= -.0035,vmax= .0035)
        axs[i-1,0].title.set_text('Compare analytic I'+str(i))
        fig.colorbar(nim, ax=axs[i-1,0], shrink=0.5)
        ntlim=axs[i-1,1].imshow(gdNtL[i],origin='lower',cmap='gray',vmin= -.0035,vmax= .0035)
        axs[i-1,1].title.set_text('to Gaussian derivative I'+str(i))
        fig.colorbar(ntlim, ax=axs[i-1,1], shrink=0.5)
if PlotsOn2:
    sigs = [1,3,5,7]
    gdNtLmult = [gaussD(NtL,6,patwid,s,scl) for s in sigs] #if diff sigmas on NtL
    fig, axs = plt.subplots(6,1+len(sigs), sharey=True, tight_layout=True,figsize=(7,7))
    viewsz = .00035
    for i in range(6): #for the derivative order
        if i==0:
            axs[i,0].title.set_text('analytic')
            nim = axs[i,0].imshow(sImat[i],origin='lower',cmap='gray',vmin= 0,vmax=1)
        else:
            nim = axs[i,0].imshow(sImat[i],origin='lower',cmap='gray',vmin= -viewsz,vmax=viewsz)
        for j in range(len(sigs)): #for the sigma value
            if i==0:
                axs[i,j+1].title.set_text('sigma='+str(j))
                ntlim=axs[i,j+1].imshow(gdNtLmult[j][i],origin='lower',cmap='gray',vmin= 0,vmax= 1)
            else:
                ntlim=axs[i,j+1].imshow(gdNtLmult[j][i],origin='lower',cmap='gray',vmin= -viewsz,vmax= viewsz)
            #if j==len(sigs)-1:
            #    fig.colorbar(ntlim, ax=axs[i,j+1], shrink=0.5)






##### SANITY CHECK: IS THE TOTALLY-ANALYTIC VERSION OF TILDE-IYY ALWAYS NEGATIVE?
#sIyymat = np.asarray([[ItoTIyy(sImat[:,i,j]) for j in range(patchsz)] for i in range(patchsz)])
#print("IF THIS IS ZERO Tilde IYY IS NEVER POSITIVE: ",np.nanmax(sIyymat.ravel()))
#
#if PlotsOn:
#    fig = plt.figure()
#    imIyy = plt.imshow(sIyymat,origin='lower',vmin = -.00000002,vmax=.00000002)
#    plt.title('tilde I_yy for analytic I')
#    fig.colorbar(imIyy, shrink=0.5)
#
#    fig, axs = plt.subplots(1,2, sharey=True, tight_layout=True)
#    gc=axs[0].imshow(gdZ[2]*gdZ[4]-gdZ[3]**2,origin='lower',vmin=-.00005,vmax=.00005)
#    axs[0].title.set_text('Gaussian surf curv: Yellow is +')
#    fig.colorbar(gc,ax=axs[0], shrink=0.5)
#    mc=axs[1].imshow(gdZ[2]+gdZ[4],origin='lower',vmin=-.00005,vmax=5)
#    axs[1].title.set_text('Mean surf curv: Yellow is +')
#    fig.colorbar(mc,ax=axs[1], shrink=0.5)
#
#
##### SANITY CHECK: Look at a little sub-sample.
#            ##in the following, make sure you're not picking any edge pixels
#            ##since GDs will give them zeroes and give you an error in explot_symm
#gdNtL2 = np.concatenate(np.transpose(gdNtL,(1,2,0)))[4000:5000]
#gdNtL2 = gdNtL2[np.isfinite(norm(gdNtL2,axis=1))]
#gdZ2 = np.concatenate(np.transpose(gdZ,(1,2,0)))[4000:5000]
#gdZ2 = gdZ2[np.isfinite(norm(gdZ2,axis=1))]
#gdZ2 = np.asarray([pponly(g) for g in gdZ2]) ##get Ying's pos-pos equiv of f
#
#
#
#
##### COMPRESS I VECTORS
#IFT = np.asarray([exploit_symmetries(i,j,[],False) for i,j in zip(gdNtL2,gdZ2)])
#I2,F2,T2 = IFT[:,0],IFT[:,1],IFT[:,2]
#with np.printoptions(precision=3, suppress=True):
#    print("\n\n I2 is:\n",np.stack(I2),"\n\n")
#
#I2,F2,T2 = np.stack(I2)[:,3:5],np.stack(F2),np.stack(T2)
#
#keepsmallI2 = np.where(np.amax(np.abs(I2),axis=1)<100)
#I2,F2,T2 = I2[keepsmallI2],F2[keepsmallI2],T2[keepsmallI2]
#if PlotsOn:
#        plt.figure()
#        plt.scatter(I2[:,0],I2[:,1])
#        #color these dots by the norm(fxxx,fxxy,fyyy,...) "mean cubicness"


if PlotsOn or PlotsOn2:
    plt.show()
