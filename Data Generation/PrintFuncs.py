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


def print1(on,I,myI):
    if on:
        fig, axs = plt.subplots(1,2, sharey=True, tight_layout=True)
        imi = axs[0].imshow(I,   origin='lower',cmap='gray',vmin=0,vmax=1.2)
        axs[0].title.set_text('photog from JW')
        fig.colorbar(imi, ax=axs[0], shrink=0.5)
        imi = axs[1].imshow(myI, origin='lower',cmap='gray',vmin=0,vmax=1.2)
        axs[1].title.set_text('rho*N*L from JW')
        fig.colorbar(imi, ax=axs[1], shrink=0.5)

##this is the plot of gd-obtained fx,fy,fxx,fxy,fyy
def print2(on,dZx,dZy,dZxx,dZxy,dZyy,gdZ,UnitTest):
    if on:
        if UnitTest:
            tryem = [dZx,dZy,dZxx,dZxy,dZyy]
            for i in [0,1,2,3,4]:
                fig = plt.figure()
                plt.plot(tryem[i][:,40],label="true")
                plt.plot(gdZ[i][:,40],label="mine")
                plt.title("How well does our GD get our surface derivs?")
                plt.legend()
        else:
            fig = plt.figure()
            [plt.plot(gdZ[i][:,40],label=i) for i in range(5)]
            plt.title("Surface Derivatives")
            plt.legend()

def print3(on,gdNab1,Nab1):
    if on:
        fig, axs = plt.subplots(1,2, sharey=True, tight_layout=True)
        imi = axs[0].imshow(gdNab1,origin='lower')
        axs[0].title.set_text('gdNab1')
        fig.colorbar(imi, ax=axs[0], shrink=0.5)
        imi = axs[1].imshow(Nab1,origin='lower')
        axs[1].title.set_text('Nab1')
        fig.colorbar(imi, ax=axs[1], shrink=0.5)

def print4(on,gdZ,scl,Na,Nb,normz):
    if on:
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

def print5(on,gdNab1,N,L,NtL,UnitTest):
    if on:
        fig, axs = plt.subplots(1,3, sharey=True, tight_layout=True, figsize=(15,5))

        nim = axs[0].imshow(gdNab1,origin='lower')
        axs[0].title.set_text('Normal Vector Field')

        if not UnitTest:
            yntlim =axs[1].imshow(np.dot(N,L),origin='lower',cmap='gray',vmax=1)
            axs[1].title.set_text("Ying's N, dot Ying's L. maybe ignore.")
            fig.colorbar(yntlim, ax=axs[1], shrink=0.5)

        ntlim=axs[2].imshow( NtL ,origin='lower',cmap='gray')
        axs[2].title.set_text("Gauss-deriv N, dot Ying's L (Image w/o albedo)")
        fig.colorbar(ntlim, ax=axs[2], shrink=0.5)

##this is the plot of gd-obtained fx,fy,fxx,fxy,fyy
def print6(on,gdZ):
    if on:
        fig,axs = plt.subplots(1,5, sharey=True, tight_layout=True,figsize=(15,5))
        labels = ["fx","fy","fxx","fxy","fyy"]
        for i in [0,1,2,3,4]:
            imi = axs[i].imshow(gdZ[i],origin='lower')
            axs[i].title.set_text(labels[i]+' values')
            fig.colorbar(imi, ax=axs[i], shrink=0.5)

def print7(on,gdNtL):
    if on:#why the heck are we seeing negative tilde Iyy???
        v=.000001
        fig = plt.figure()
        tIyy = ItoTIyy(gdNtL)
        imIyy = plt.imshow(np.sign(tIyy),origin='lower')
        plt.title('Sign(tilde Iyy) for GD I')
        fig.colorbar(imIyy, shrink=0.5)

        fig = plt.figure()
        tIyy = gdNtL[0]
        imIyy = plt.imshow(np.sign(tIyy),origin='lower',vmin=-v,vmax=v)
        plt.title('Sign(I): Yellow is +')
        fig.colorbar(imIyy, shrink=0.5)

        fig = plt.figure()
        tIyy = gdNtL[1]**2*gdNtL[5] + gdNtL[2]**2*gdNtL[3]
        imIyy = plt.imshow(tIyy,origin='lower',vmin=-v,vmax=v)
        plt.title('pos component: Yellow is +')
        fig.colorbar(imIyy, shrink=0.5)

        fig = plt.figure()
        tIyy = - 2*gdNtL[4]*gdNtL[1]*gdNtL[2]
        imIyy = plt.imshow(tIyy,origin='lower',vmin=-v,vmax=v)
        plt.title('neg component: Yellow is +')
        fig.colorbar(imIyy, shrink=0.5)

##the OLD 6x6 image plot.
def print8(on,sImat,gdNtL):
    if on:
        fig, axs = plt.subplots(1,2, sharey=True, sharex=True, tight_layout=True)
        nim = axs[0].imshow(sImat[0],origin='lower',cmap='gray',vmin= -0.1,vmax= 1)
        axs[0].title.set_text('Compare analytic I')
        fig.colorbar(nim, ax=axs[0], shrink=0.5)
        ntlim=axs[1].imshow(gdNtL[0],origin='lower',cmap='gray',vmin= -0.1,vmax= 1)
        axs[1].title.set_text('to Gaussian derivative I')
        fig.colorbar(ntlim, ax=axs[1], shrink=0.5)

        fig, axs = plt.subplots(5,2, sharey=True, sharex=True, tight_layout=True,figsize=(8,8))
        viewsz = 0.0035
        for i in [1,2,3,4,5]:
        
            nim = axs[i-1,0].imshow(sImat[i],origin='lower',cmap='gray',vmin= -viewsz,vmax= viewsz)
            axs[i-1,0].title.set_text('Compare analytic I'+str(i))
            fig.colorbar(nim, ax=axs[i-1,0], shrink=0.5)

            ntlim=axs[i-1,1].imshow(gdNtL[i],origin='lower',cmap='gray',vmin= -viewsz,vmax= viewsz)
            axs[i-1,1].title.set_text('to Gaussian derivative I'+str(i))
            fig.colorbar(ntlim, ax=axs[i-1,1], shrink=0.5)

#evaluate the KZ equations -- hopefully close to zero
def print9(on,gdZ,sImat):
    if on:
        fI = np.concatenate([gdZ,sImat])
        fIvec = np.reshape(fI,(fI.shape[0],fI.shape[1]*fI.shape[2]))
        KZsvec = np.apply_along_axis(evalKZs, 0, fIvec)
        KZsimg = np.reshape(KZsvec,(fI.shape[1],fI.shape[2]))
        fig = plt.figure()
        plt.imshow(KZsimg)
        plt.title('KZ values')
        plt.colorbar()

## the REAL 6x6 image plot.
def print10(on,patchsz,sigs,sImat,gdNtLmult,viewsz):
    cf = 1#int((patchsz-1)/2 + 1 - 75)
    ct = -1#int((patchsz-1)/2 + 1 + 75)
    if on:
        fig, axs = plt.subplots(6,1+len(sigs), sharey=True, tight_layout=True,figsize=(7,7))
        for i in range(6): #for the derivative order
            if i==0:
                axs[i,0].title.set_text('analytic')
                nim = axs[i,0].imshow(sImat[i][cf:ct,cf:ct],
                                      origin='lower',cmap='gray',vmin= 0,vmax=1) #plot analytic
            else:
                nim = axs[i,0].imshow(sImat[i][cf:ct,cf:ct],
                                      origin='lower',cmap='gray',vmin= -viewsz,vmax=viewsz) #plot analytic
            for j in range(len(sigs)): #for the sigma value
                if i==0:
                    axs[i,j+1].title.set_text('sigma='+str(sigs[j]))
                    ntlim=axs[i,j+1].imshow(gdNtLmult[j][i][cf:ct,cf:ct],
                                            origin='lower',cmap='gray',vmin= 0,vmax= 1)
                else:
                    ntlim=axs[i,j+1].imshow(gdNtLmult[j][i][cf:ct,cf:ct],
                                            origin='lower',cmap='gray',vmin= -viewsz,vmax= viewsz)

## the 5x1 image derivative slice plots
def print11(on,sImat,sigs,vert,viewsz0,gdNtLmult):
    if on:
        plotthese = [1,2,3,4,5]
        rng = range(len(sImat[0][20]))
        labels = ["Ix","Iy","Ixx","Ixy","Iyy"]

        
        ## as a single figure
        fig, axs = plt.subplots(len(plotthese),1,sharex=True, sharey=True, tight_layout=True,figsize=(10,7))
        for i in range(len(plotthese)): #for the derivative order
            if i<2:
                viewsz=viewsz0*10
            else:
                viewsz=viewsz0
            if i==0:
                nim = axs[i].plot(sImat[plotthese[i]][vert,:]) #plot analytic
            else:
                nim = axs[i].plot(sImat[plotthese[i]][vert,:]) #plot analytic
                plt.ylim(-viewsz,viewsz)
            for j in range(len(sigs)): #for the sigma value
                if i==0:
                    ntlim=axs[i].plot(gdNtLmult[j][plotthese[i]][vert,:], label=str(sigs[j]))
                    axs[i].legend()
                else:
                    ntlim=axs[i].plot(gdNtLmult[j][plotthese[i]][vert,:])
                    
        ## as separate figures
        
        ###LET'S TRY A THING.
#        dIxy1,dIxx = np.gradient(sImat[1])
#        dIyy,dIxy2 = np.gradient(sImat[2])
#        print(dIxy1-dIxy2)
#        compareto = dIxx[:,vert], dIxy1[:,vert], dIyy[:,vert]
#
#        for i in range(len(plotthese)): #for the derivative order
#            plt.figure()
#            plt.title(labels[i])
#            toplot = sImat[plotthese[i]][:,vert]
#            if i>1:
#                toplot = toplot - compareto[i-2]
#            nim = plt.plot(toplot) #plot analytic

def print12(on,XX,YY,sImat,gdNtL):
    if on:
        fig = plt.figure()
        ax = fig.add_subplot(121, projection='3d')
        ax.scatter(XX.ravel(), YY.ravel(), sImat[5].ravel(), c='k', marker='o',label='analytic')
        ax.scatter(XX.ravel(), YY.ravel(), gdNtL[5].ravel(), c='b', marker='o',label='numerics')
        ax = fig.add_subplot(122)#, projection='3d')
        plt.imshow(sImat[5]/gdNtL[5],vmin=0.5,vmax=1.5)
        #ax.scatter(XX.ravel(), YY.ravel(),(sImat[5]/gdNtL[5]).ravel())
    
        fig = plt.figure()
        ax = fig.add_subplot(121, projection='3d')
        ax.scatter(XX.ravel(), YY.ravel(), sImat[3].ravel(), c='k', marker='o',label='analytic')
        ax.scatter(XX.ravel(), YY.ravel(), gdNtL[3].ravel(), c='b', marker='o',label='numerics')
        ax = fig.add_subplot(122)#, projection='3d')
        plt.imshow(sImat[3]/gdNtL[3],vmin=0.5,vmax=1.5)
        #ax.scatter(XX.ravel(), YY.ravel(), (sImat[3]/gdNtL[3]).ravel())

def print13(on,sIyymat,gdZ):
    if on:
        fig = plt.figure()
        imIyy = plt.imshow(sIyymat,origin='lower',vmin = -.00000001,vmax=.00000001)
        plt.title('tilde I_yy for analytic I')
        fig.colorbar(imIyy, shrink=0.5)
    
#        fig, axs = plt.subplots(1,2, sharey=True, tight_layout=True)
#        gc=axs[0].imshow(gdZ[2]*gdZ[4]-gdZ[3]**2,origin='lower',vmin=-.00005,vmax=.00005)
#        axs[0].title.set_text('Gaussian surf curv: Yellow is +')
#        fig.colorbar(gc,ax=axs[0], shrink=0.5)
#        mc=axs[1].imshow(gdZ[2]+gdZ[4],origin='lower',vmin=-.00005,vmax=5)
#        axs[1].title.set_text('Mean surf curv: Yellow is +')
#        fig.colorbar(mc,ax=axs[1], shrink=0.5)
    
    
##helpers to 6x6 plot
def print14(on,sImat,Ierr):
    if on:
        fig, axs = plt.subplots(6,1, sharey=True, sharex=True, tight_layout=True,figsize=(4,8))
        viewsz = 0.0035
       
        dIxy1,dIxx = np.gradient(sImat[1])
        dIyy,dIxy2 = np.gradient(sImat[2])
        
#        compareto = sImat[1],sImat[2], dIxx, dIxy1, dIyy
        compareto = sImat + Ierr
        
        for i in range(6):
            nim = axs[i].imshow(compareto[i],origin='lower',cmap='gray',vmin= -viewsz,vmax= viewsz)
            axs[i].title.set_text('analytic I'+str(i)+' + err')
            fig.colorbar(nim, ax=axs[i], shrink=0.5)

## show confidence plots (e.g. high cubicness areas)
def print15(on,Ierr,viewsz):
    viewsz /= 10
    if on:
      fig, axs = plt.subplots(1,3, sharey=True, tight_layout=True,figsize=(7,3))
      fig.suptitle('Large Magnitudes denote Low Confidence', fontsize=16)
      axs[0].imshow(Ierr[3], origin='lower',cmap='gray')
      axs[0].title.set_text("Ixx err")
      axs[1].imshow(Ierr[4], origin='lower',cmap='gray')
      axs[1].title.set_text("Ixy err")
      axs[2].imshow(Ierr[5], origin='lower',cmap='gray')
      axs[2].title.set_text("Iyy err")
