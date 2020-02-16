
## Called by TrainingDataGenerator.py


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


def evalKZs(cde,abIvec):
    c,d,e = cde
    a,b = abIvec[:2]
    I0, Ix0, Iy0, Ixx0, Ixy0, Iyy0 = abIvec[2:]
    
    ##KZs evaluated at the origin (x,y)=(0,0):
    p1 = c**2*I0 + b**2*c**2*I0 - 2*a*b*c*d*I0 + d**2*I0 + a**2*d**2*I0 + 2*a*c*Ix0 + 2*a**3*c*Ix0 + 2*a*b**2*c*Ix0 + 2*b*d*Ix0 + 2*a**2*b*d*Ix0 + 2*b**3*d*Ix0 + Ixx0 + 2*a**2*Ixx0 + a**4*Ixx0 + 2*b**2*Ixx0 + 2*a**2*b**2*Ixx0 + b**4*Ixx0
    p2 = d**2*I0 + b**2*d**2*I0 - 2*a*b*d*e*I0 + e**2*I0 + a**2*e**2*I0 + 2*a*d*Iy0 + 2*a**3*d*Iy0 + 2*a*b**2*d*Iy0 + 2*b*e*Iy0 + 2*a**2*b*e*Iy0 + 2*b**3*e*Iy0 + Iyy0 + 2*a**2*Iyy0 + a**4*Iyy0 + 2*b**2*Iyy0 + 2*a**2*b**2*Iyy0 + b**4*Iyy0
    p3 = c*d*I0 + b**2*c*d*I0 - a*b*d**2*I0 - a*b*c*e*I0 + d*e*I0 + a**2*d*e*I0 + a*d*Ix0 + a**3*d*Ix0 + a*b**2*d*Ix0 + b*e*Ix0 + a**2*b*e*Ix0 + b**3*e*Ix0 + Ixy0 + 2*a**2*Ixy0 + a**4*Ixy0 + 2*b**2*Ixy0 + 2*a**2*b**2*Ixy0 + b**4*Ixy0 + a*c*Iy0 + a**3*c*Iy0 + a*b**2*c*Iy0 + b*d*Iy0 + a**2*b*d*Iy0 + b**3*d*Iy0
    return norm((p1,p2,p3))

def solveKZs(ab,Ivec):
    cone = NonlinearConstraint(lambda cde: [cde[0]*cde[2]-cde[1]**2 , cde[0]+cde[2]], 0, 100)
    if len(ab.ravel())==2:
        input = np.concatenate([ab,Ivec]) # 1 x 8 matrix
        sols = minimize(evalKZs,(1,-1,1),input,method='trust-constr',constraints = cone)
    else:
        input = np.concatenate([ab,Ivec],axis=1) # n x 8 matrix
        sols = [minimize(evalKZs,(1,-1,1),i,method='trust-constr',constraints = cone) for i in input]
    return sols

def sampS2p(npoints):
    vec = np.random.randn(3, npoints)
    vec /= norm(vec, axis=0)
    vec = vec/vec[2]
    return np.transpose(vec)/10
    
def coneMember(cde):
    c,d,e = cde
    return c*e-d**2>0 and c+e>0

def sampR3p(npoints):
    vec = np.random.uniform(-10,10,(3,1))*0
    while len(vec[0])<npoints+1:
        new = np.random.uniform(-10,10,(3,1))
        if coneMember(new):
            vec = np.append(vec,new,axis=1)
    return np.transpose(vec[:,1:])
    
def symbI(L, ab, cde ):
    #this comes from the CalcIFromABCDE in Utilities2.nb

    L1,L2,L3 = L
    a,b = -ab
    c,d,e = -cde
    
    Iout = (
    (-(a*L1) - b*L2 + L3)/(np.sqrt(1 + a**2 + b**2)*np.sqrt(L1**2 + L2**2 + L3**2)), ##
    (-(c*(L1 + b**2*L1 - a*b*L2 + a*L3)) - d*(-(a*b*L1) + L2 + a**2*L2 + b*L3))/ ((1 + a**2 + b**2)**(3/2)*np.sqrt(L1**2 + L2**2 + L3**2)), ##
    (-(d*(L1 + b**2*L1 - a*b*L2 + a*L3)) - e*(-(a*b*L1) + L2 + a**2*L2 + b*L3))/ ((1 + a**2 + b**2)**(3/2)*np.sqrt(L1**2 + L2**2 + L3**2)), ##
    (2*(b + b**3)*c*d*L1 + b*((1 + b**2)*c**2 + 3*d**2)*L2 + a**3*d*(d*L1 + 2*c*L2) - ((1 + b**2)*c**2 + (1 - 2*b**2)*d**2)*L3 - a**2*(b*(4*c*d*L1 + 2*c**2*L2 - 3*d**2*L2) + (-2*c**2 + d**2)*L3) + a*(3*(1 + b**2)*c**2*L1 + (1 - 2*b**2)*d**2*L1 + 2*c*d*(L2 - 2*b**2*L2 + 3*b*L3)))/((1 + a**2 + b**2)**(5/2)*np.sqrt(L1**2 + L2**2 + L3**2)), ##
    ((b + b**3)*(d**2 + c*e)*L1 + b*d*(c + b**2*c + 3*e)*L2 + a**3*(d*e*L1 + d**2*L2 + c*e*L2) - d*(c + b**2*c + e - 2*b**2*e)*L3 - a**2*(2*b*(d**2 + c*e)*L1 + b*d*(2*c - 3*e)*L2 + d*(-2*c + e)*L3) + a*(3*(1 + b**2)*c*d*L1 - (-1 + 2*b**2)*d*(e*L1 + d*L2) + 3*b*d**2*L3 + c*e*(L2 - 2*b**2*L2 + 3*b*L3)))/((1 + a**2 + b**2)**(5/2)* np.sqrt(L1**2 + L2**2 + L3**2)), ##
    (2*(b + b**3)*d*e*L1 + b*((1 + b**2)*d**2 + 3*e**2)*L2 + a**3*e*(e*L1 + 2*d*L2) - ((1 + b**2)*d**2 + (1 - 2*b**2)*e**2)*L3 - a**2*(b*(4*d*e*L1 + 2*d**2*L2 - 3*e**2*L2) + (-2*d**2 + e**2)*L3) + a*(3*(1 + b**2)*d**2*L1 + (1 - 2*b**2)*e**2*L1 + 2*d*e*(L2 - 2*b**2*L2 + 3*b*L3)))/ ((1 + a**2 + b**2)**(5/2)*np.sqrt(L1**2 + L2**2 + L3**2))
    )
    return Iout


def calcIFromABCDE(L,ab,cde,px):
    if norm(px)!=0:
        print("not yet equipped to handle (x,y)!=(0,0).")
        return
    return np.asarray([symbI(l,a,c) for l,a,c in zip(L,ab,cde)])

    
    
def ItoTIyy(bI): #bI is (possibly multiple) 6-vectors
    if len(bI.ravel())==6:
        return bI[1]**2*bI[5] - 2*bI[4]*bI[1]*bI[2] + bI[2]**2*bI[3]     
    return bI[1]**2*bI[5] - 2*bI[4]*bI[1]*bI[2] + bI[2]**2*bI[3]
