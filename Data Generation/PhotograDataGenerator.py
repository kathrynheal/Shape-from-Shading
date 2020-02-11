



## THIS IS MEANT TO FEED INTO EVALUATE.PY


import matplotlib, ast, sys, time, os, socket, warnings
from Utilities3 import *
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
from scipy.io import loadmat
from mpl_toolkits.axes_grid1.colorbar import colorbar


prefix = "/Users/Heal/Dropbox/Research/Experiments/"
dataset = "029027"
animal = "turtle"
unitTest = True

patchsz = 151
subset = np.linspace(100,100+patchsz-1,num=patchsz,dtype=np.int)
patchx,patchy = np.meshgrid(subset,subset) #each is patchsz x patchsz

alldata = loadmat("/Users/Heal/Dropbox/Research/XiongZickler2014_data/" + animal + "/result.mat")
#print("Keys in file: ",alldata.keys())
N = alldata['n'][patchx,patchy] #unit-length vectors
N[:,:,(1,0)] = -N[:,:,(0,1)] #Ying's convention is y,x and negative
Na = N[:,:,0]/N[:,:,2]
Nb = N[:,:,1]/N[:,:,2]

Z = alldata['Z'][patchx,patchy]
patwid = 1
gdZ = gaussD(Z,5,patwid) ##gaussian surface derivatives
scl = (len(Z)-1)/(2*patwid)
L = alldata['L']
#NtL=-(N[:,:,0]*L[0,0] + N[:,:,1]*L[1,0] - N[:,:,2]*L[2,0])/N[:,:,2]
NtL=np.dot(N,L[:,0])

#####plotzzz
fig, axs = plt.subplots(1,2, sharey=True, tight_layout=True)
nim = axs[0].imshow(N,origin='lower')
axs[0].title.set_text('Normal Vector Field')

ntlim=axs[1].imshow(NtL,origin='lower',cmap='gray')
axs[1].title.set_text('NtL (Image w/o albedo)')
fig.colorbar(ntlim, ax=axs[1], shrink=0.5)

fig,axs = plt.subplots(1,5, sharey=True, tight_layout=True,figsize=(15,5))
labels = ["a","b","c","d","e"]
for i in [0,1,2,3,4]:
    imi = axs[i].imshow(gdZ[i],origin='lower')
    axs[i].title.set_text(labels[i]+' values')
    fig.colorbar(imi, ax=axs[i], shrink=0.5)


fig, axs = plt.subplots(2,2, sharey=True, tight_layout=True)
imi = axs[0,0].imshow(gdZ[0],origin='lower',vmin=-3*scl,vmax=3*scl)
axs[0,0].title.set_text('my dfx')
fig.colorbar(imi, ax=axs[0,0], shrink=0.5)
imi = axs[0,1].imshow(scl*Na,origin='lower',vmin=-3*scl,vmax=3*scl)
axs[0,1].title.set_text('Ying dfx * scale')
fig.colorbar(imi, ax=axs[0,1], shrink=0.5)
imi = axs[1,0].imshow(gdZ[1],origin='lower',vmin=-3*scl,vmax=3*scl)
axs[1,0].title.set_text('my dfy')
fig.colorbar(imi, ax=axs[1,0], shrink=0.5)
imi = axs[1,1].imshow(scl*Nb,origin='lower',vmin=-3*scl,vmax=3*scl)
axs[1,1].title.set_text('Ying dfy * scale')
fig.colorbar(imi, ax=axs[1,1], shrink=0.5)
#plt.show()
###


gdNtL = gaussD(NtL,6,patwid)

fig = plt.figure()
tIyy = gdNtL[1]**2*gdNtL[5] - 2*gdNtL[4]*gdNtL[1]*gdNtL[2] + gdNtL[2]**2*gdNtL[3]
imIyy = plt.imshow(tIyy,origin='lower',vmin=-10,vmax=0)#-.000001,vmax=.000001)
plt.title('Sign(tilde Iyy): Yellow is +')
fig.colorbar(imIyy, shrink=0.5)
plt.show()

#fig = plt.figure()
#tIyy = gdNtL[0]
#imIyy = plt.imshow(np.sign(tIyy),origin='lower')
#plt.title('Sign(I): Yellow is +')
#fig.colorbar(imIyy, shrink=0.5)
#
#
#fig = plt.figure()
#tIyy = gdNtL[1]**2*gdNtL[5] + gdNtL[2]**2*gdNtL[3]
#imIyy = plt.imshow(tIyy,origin='lower')
#plt.title('pos component: Yellow is +')
#fig.colorbar(imIyy, shrink=0.5)
#
#
#fig = plt.figure()
#tIyy = - 2*gdNtL[4]*gdNtL[1]*gdNtL[2]
#imIyy = plt.imshow(tIyy,origin='lower')
#plt.title('neg component: Yellow is +')
#fig.colorbar(imIyy, shrink=0.5)
#plt.show()




#fig, axs = plt.subplots(1,2, sharey=True, tight_layout=True)
#gc=axs[0].imshow(gdZ[2]*gdZ[4]-gdZ[3]**2,origin='lower',vmin=-5,vmax=5)
#axs[0].title.set_text('Gaussian surf curv: Yellow is +')
#fig.colorbar(gc,ax=axs[0], shrink=0.5)
#mc=axs[1].imshow(gdZ[2]+gdZ[4],origin='lower',vmin=-5,vmax=5)
#axs[1].title.set_text('Mean surf curv: Yellow is +')
#fig.colorbar(mc,ax=axs[1], shrink=0.5)
#plt.show()



##in the following, make sure you're not picking any edge pixels, since GDs will give them zeroes and give you an error in explot_symm
#gdNtL2 = np.transpose(np.asarray([np.concatenate(i) for i in gdNtL]))[3051:6093]
#gdZ2 = np.transpose(np.asarray([np.concatenate(i) for i in gdZ]))[3051:6093]
#gdZ2 = np.asarray([pponly(g) for g in gdZ2]) ##get Ying's pos-pos equiv of f
#
#IFT = np.asarray([exploit_symmetries(i,j,[],False) for i,j in zip(gdNtL2,gdZ2)])
#I2 = IFT[:,0]
#F2 = IFT[:,1]
#T2 = IFT[:,2]
#
#I2 = np.stack(I2)[:,3:5]
#F2 = np.stack(F2)
#T2 = np.stack(T2)
#
#
#print(F2.shape)
###WHY are some of the I2 elements the wrong sign? like Iyy>0?
#
#print("\n\n",gdNtL2[:,(3,5)],"\n\n")
##print(NtL)
##print(gdZ2)
##print(np.round(np.stack(IFT[:,0]),decimals=9))
#print(I2)
#
#keepsmallI2 = np.where(np.amax(np.abs(I2),axis=1)<10)
#I2 = I2[keepsmallI2]
#F2 = F2[keepsmallI2]
#T2 = T2[keepsmallI2]
#
#plt.figure()
#plt.scatter(I2[:,0],I2[:,1])
#plt.show()
