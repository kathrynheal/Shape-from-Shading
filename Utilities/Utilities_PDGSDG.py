"""Called by PhotograDataGenerator.py and SynthDataGenerator.py"""


import matplotlib, ast, sys, os, socket, warnings
import numpy as np
from scipy.optimize import minimize
from numpy.linalg import norm
from scipy.optimize import NonlinearConstraint
import multiprocessing as mp
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.signal import convolve2d
import scipy.ndimage as ndi
np.set_printoptions(suppress=True)
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)


def makeQP(a,imsz,xybound):
    xyrange = np.linspace(-xybound,xybound,num=imsz) # num should be odd so there's a "center" pixel for 0,0
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


def gragauss(sigma):
    """Create Hessian-of-Gaussian kernels with standard deviation sigma.
    Use equation 4.21 in the Szelsiki book. Or in Mathematica by differentiating kernel G2.
    """
    # Use an odd-size square window with length greater than 5 times sigma
    w = 2 * np.floor(7 * sigma / 2) + 1
    x, y = np.meshgrid(np.arange(-(w-1)/2, (w-1)/2 + 1),
                       np.arange(-(w-1)/2, (w-1)/2 + 1))

    sc = 1/(2*np.pi)
    r2 = x**2 + y**2
    G2 = np.exp(-r2/(2*sigma**2))
    
    dx = -x*(1/sigma**3)*G2*sc
    dy = -y*(1/sigma**3)*G2*sc
    
    return dx,dy
    

def hesgauss(sigma):
    """Create Hessian-of-Gaussian kernels with standard deviation sigma.
    Use equation 4.23 in the Szelsiki book.
    """
    # Use an odd-size square window with length greater than 5 times sigma
    w = 2 * np.floor(7 * sigma / 2) + 1
    x, y = np.meshgrid(np.arange(-(w-1)/2, (w-1)/2 + 1),
                       np.arange(-(w-1)/2, (w-1)/2 + 1))
    
    # TODO: write first derivative version of this
    # c2d( c2d(im, | ), _ )
    
    r2 = x**2 + y**2
    G2 = 1/(2*np.pi*sigma**2)*np.exp(-r2/(2*sigma**2))
    
#    dxx = ((x**2 - sigma**2) / sigma**4) * G2
#    dxy = ( x * y            / sigma**4) * G2
#    dyy = ((y**2 - sigma**2) / sigma**4) * G2
    dxx = (2*np.pi/sigma)*G2*(1-x**2/(2*sigma**2))
    dxy = 0*x # CHANGE THIS
    dyy = (2*np.pi/sigma)*G2*(1-y**2/(2*sigma**2))

    
#    #divide by filter sum so entire filters each sum to 1
    dxx /= np.sum(dxx.ravel())
    dxy /= np.sum(dxy.ravel())
    dyy /= np.sum(dyy.ravel())

    return dxx, dxy, dyy


def gaussD(im,ord,bnd,sig,scl): #im is a SQUARE matrix. ord is 5 or 6.
    
    # older tries.
    #print("scl is: ",scl)
    #first axis here is the matrix row#, so it's "y"
    #second axis here is the matrix col#, so it's "x"
    #    m = 'mirror'
    #    dx  = scl*gaussian_filter(im,sigma=sig,order=(0,1),mode=m)
    #    dy  = scl*gaussian_filter(im,sigma=sig,order=(1,0),mode=m)
    #    dxx = scl*gaussian_filter(im,sigma=sig,order=(0,2),mode=m)
    #    dxy = scl*gaussian_filter(im,sigma=sig,order=(1,1),mode=m)
    #    dyy = scl*gaussian_filter(im,sigma=sig,order=(2,0),mode=m)
    
    if sig==0:
        dy, dx  = np.gradient(im)
        dxy,dxx = np.gradient(dx)
        dyy,dxy = np.gradient(dy)
        dx,dy,dxx,dxy,dyy = scl*dx,scl*dy,scl**2*dxx,scl**2*dxy,scl**2*dyy
    else:
        Gx, Gy = gragauss(sig)
        Gxx, Gxy, Gyy = hesgauss(sig)
        dx  = ndi.convolve(im, Gx)  # x  derivative
        dy  = ndi.convolve(im, Gy)  # y  derivative
        #        dxx = ndi.convolve(im, Gxx)  # xx derivative
        #        dxy = ndi.convolve(im, Gxy)  # xy derivative
        #        dyy = ndi.convolve(im, Gyy)  # yy derivative
        dxx = ndi.convolve(dx, Gx)  # xx derivative
        dxy = ndi.convolve(dy, Gx)  # xy derivative
        dyy = ndi.convolve(dy, Gy)  # yy derivative
        
    if ord==6:
        return np.asarray((im,dx,dy,dxx,dxy,dyy))
    return np.asarray((dx,dy,dxx,dxy,dyy))


def FindThirdOrder(im,dim,sig):
    fx,fy,fxx,fxy,fyy = dim
    if sig>0:
        Gx, Gy = gragauss(sig+1)
        dxxx, dxxy = ndi.convolve(fxx, Gx),  ndi.convolve(fxx, Gy)
        dxyx, dxyy = ndi.convolve(fxy, Gx),  ndi.convolve(fxy, Gy)
        dyyx, dyyy = ndi.convolve(fyy, Gx),  ndi.convolve(fyy, Gy)
    else:
        dxxy, dxxx  = np.gradient(fxx)
        dxyy, dxyx  = np.gradient(fxy)
        dyyy, dyyx  = np.gradient(fyy)
    stat = dxxx**2+dxxy**2+dxyy**2+dyyy**2
    return stat, np.asarray([dxxx,dxxy,dxyy,dyyy])


def GetCubicIError(L,surf):
    """returns: Iquartic - Iquad, derived in 'et cetera.nb'"""
    l1,l2,l3 = L
    if len(surf)==5:
        a,b,c,d,e = surf
        f,g,h,i = 0,0,0,0
    elif len(surf)==9:
        a,b,c,d,e,f,g,h,i = surf
    else:
        print("GetCubicIError: f is not of the right shape.")
    var1 = l1 + b**2*l1 - a*b*l2 + a*l3
    var2 = -(a*b*l1) + l2 + a**2*l2 + b*l3
    var3 = -(1 + a**2 + b**2)**(3/2) * np.sqrt(l1**2 + l2**2 + l3**2)
    z = np.zeros(var3.shape)
    Ierr = np.asarray([ z, z, z, (f*var1+g*var2)/var3, (g*var1+h*var2)/var3, (h*var1+i*var2)/var3])
    return Ierr
