    
"""Called by Training.py"""


import matplotlib, ast, sys, time, os, glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.pyplot import *
from mpl_toolkits.mplot3d import *
from scipy.linalg import block_diag
from numpy.linalg import inv
from numpy.matlib import repmat
np.set_printoptions(suppress=True)


#####################################################################################
#####################################################################################


def getVars(name, vars):
    """FOR GETTING L1 (etc.) WEIGHTS"""
    for i in range(len(vars)):
        if vars[i].name.startswith(name):
            return vars[i]
    return None


#####################################################################################
#####################################################################################


def saveAll(saveall, numIters, codeis, prefix,costvec, sess, saver, signature, runtype, alloutput):
    if saveall==1:
        np.save(prefix+"Outputs/allcosts" +codeis, costvec[1:len(costvec)-1])
        np.save(prefix+"Outputs/signatures" +codeis, signature)
        saver.save(sess,"./"+prefix+"NN/Models/KH_Model_"+runtype+"_"+codeis)
    return None


#####################################################################################
#####################################################################################


def WriteTrainingOutputs(uniquenum,paramsIn):
    fname = "TrainingOutput/"+uniquenum+"/Params.txt"
    with open(fname,"w+") as f:
        f.write("\n\n*****************\n\nHyper Params:\n")
        strg = ["\nDataset identifier:   ",
                "\nNumber of iterations:   ",
                "\nStep size:   ",
                "\nBatch size:   ",
                "\nDepth of h:   ",
                "\nWidth of h:   ",
                "\nDepth of g:   ",
                "\nWidth of g:   ",
                "\nTest/Train ratio:   ",
                "\ntoyopt:   ",
                "\nsubset:   ",
                "\nsize0:   ",
                "\nAdam B1:   "]
        B = [s+str(n) for s,n in list(zip(strg,paramsIn))]
        f.write(''.join(B))
    return fname
    
    
#####################################################################################
#####################################################################################


def PrintParams(uniquenum,paramsIn):
    print("\n\n*****************\n\nHyper Params:\n")
    strg = ["Dataset identifier:   ",
            "Number of iterations:   ",
            "Step size:   ",
            "Batch size:   ",
            "Depth of h:   ",
            "Width of h:   ",
            "Depth of g:   ",
            "Width of g:   ",
            "Test/Train ratio:   ",
            "toyopt:   ",
            "subset:   ",
            "size0:   ",
            "Adam B1:   "]
    B = [print(s+str(n)) for s,n in list(zip(strg,paramsIn))]


#####################################################################################
#####################################################################################


def prettyprint(data,args):
    print("")
    frmt = ("{:>35.6f}"+"{:>35.6f}")
    print(("{:>35}"*2).format(*args))
    st = "----------"
    print(("{:>35}"*2).format(*[st,st]))
    for i in data:
        print(frmt.format(*i))


#####################################################################################
#####################################################################################


def animate(i,argl):
    ax,Y_,yo_all = argl
    yplt = yo_all[i]
    ax.clear()
    sc2 = ax.scatter3D(yplt[:,0],yplt[:,1],yplt[:,2])
    sc3 = ax.scatter3D(  Y_[:,0],  Y_[:,1],  Y_[:,2])
    ax.set_title('NN Results.\nEPOCH: '+str(i)+' of '+str(len(yo_all))+'.')
    ax.set_xlabel('f_{xx}')
    ax.set_ylabel('f_{xy}')
    ax.set_zlabel('f_{yy}')
    ax.legend([sc2, sc3], ['Training Output from NN', 'Ground Truth'], loc='lower left', numpoints = 1)
    
    
#####################################################################################
#####################################################################################


def toysave(I,X,Y):
    np.save(os.getcwd()+"tempI_TOY.npy",  I)
    np.save(os.getcwd()+"tempAB_TOY.npy", X)
    np.save(os.getcwd()+"tempCDE_TOY.npy",Y)
    
    
#####################################################################################
#####################################################################################


def isdiverging(yo_): #this usually doesn't get flagged when using AdamOptimizer
    if np.isnan(yo_).any():
        print("\n\nERROR: Initialization failed. You're getting NaNs.\nRetry, and if problem persists, decrease step size.\n")
        sys.exit()


#####################################################################################
#####################################################################################


def loaddata(toyopt,size0,subset,uniquetime,startind):

    #if dataset is bigger than 10 this is a good idea.
    if toyopt==1:
        #toy1, random
        print("Data is Toy Problem 1")
        X = np.random.randint(10,size=(size0,2))
        I = np.random.randint(10,size=(size0,6))
        Y = np.random.randint(10,size=(size0,3))
        toysave(I,X,Y)
    elif toyopt==2: #stepsize=1e-3
        print("Data is Toy Problem 2")
        X = np.random.uniform(1,4,size=(size0,3))
        I = np.random.uniform(1,4,size=(size0,2))
        Y = np.array([[x,x+i,x*i] for x,i in zip(np.sum(I,axis=1),np.sum(I,axis=1))])
        toysave(I,X,Y)
    elif toyopt==3: #stepsize=1e-2
        print("Data is Toy Problem 3")
        rn = 1;
        X, I = np.mgrid[-rn:(rn):10j, -rn:(rn):10j]
        X = X.flatten().reshape((X.shape[0]*X.shape[1],1))
        I = I.flatten().reshape((I.shape[0]*I.shape[1],1))
        Y = 10*I*X+1
        toysave(I,X,Y)
    elif toyopt==0 or toyopt==5:#then use real data!!!!
        if toyopt==0:
            print("Data is /all*")
            ListI,Listf = parsedata()
            sn = int(Listf.shape[1]/5)
            ListI = np.concatenate(np.asarray([repmat(l,sn,1) for l in ListI]))
            Listf = np.concatenate(Listf.reshape(Listf.shape[0],sn,5))
        elif toyopt==5:
            print("Data is /Synth/")
            i = os.getcwd()+"/Data/Synth/I.csv"
            f = os.getcwd()+"/Data/Synth/F.csv"
            ListI = np.asarray([[float(g) for g in h.split(',')] for h in np.loadtxt(i,dtype='str')])
            Listf = np.asarray([[float(g) for g in h.split(',')] for h in np.loadtxt(f,dtype='str')])
        if subset:
            subnum=size0
            I = ListI[startind:(startind+subnum)]
            X = Listf[startind:(startind+subnum),:2]
            Y = Listf[startind:(startind+subnum),2:]
        else:
            I = ListI
            X = Listf[:,:2]
            Y = Listf[:,2:]
    return I,X,Y
    
    
#####################################################################################
#####################################################################################


def parsedata():

    #print("\n\nThe Current Working Directory:\n ",os.getcwd(),"\n\n")

    #scrape all files
    HoldI,Holdf = [],[]
    filepath = os.getcwd()
    os.chdir(filepath + "/Data")
    print("Reading Training Data:")
    GlobI = np.sort(glob.glob("I_*"))
    Globf = np.sort(glob.glob("F_*"))
    for f,i in zip(Globf, GlobI):
        print(f,i)
        ListI = np.asarray([[float(g) for g in h.split(',')] for h in np.loadtxt(i,dtype='str')])
        Listf = np.asarray([[float(g) for g in h.split(',')] for h in np.loadtxt(f,dtype='str')])
        print(Listf.shape)
        if HoldI == []:
            HoldI = [ListI]
            Holdf = [Listf]
        else:
            HoldI = np.append(HoldI,[ListI],axis=1)
            Holdf = np.append(Holdf,[Listf],axis=1)
        print(np.array(Holdf).shape)
    ListI = np.concatenate(HoldI)
    Listf = np.concatenate(Holdf)
    os.chdir(filepath)
    return ListI,Listf


#####################################################################################
#####################################################################################


def splittraintest(X,Y,I,ttfrac,toyopt,subset):

    #set train fraction & shuffle dataset
    #np.random.seed()#(707)
    idx=np.arange(X.shape[0])
    if ttfrac < 1:
        np.random.shuffle(idx) #modifies idx in-place, so it's now shuffled
    X = X[idx]
    Y = Y[idx]
    I = I[idx]

    train_stop = int(len(X) * ttfrac)
    test_stop  = len(X)-train_stop
    X_ = X[:train_stop]
    Y_ = Y[:train_stop]
    I_ = I[:train_stop]

    # have the same 10% holdout as the previous example
    X_t = X[len(X)-test_stop:]
    Y_t = Y[len(X)-test_stop:]
    I_t = I[len(I)-test_stop:]

    return X,Y,I,X_,Y_,I_,X_t,Y_t,I_t,idx,train_stop,test_stop


#####################################################################################
#####################################################################################
    
    
def getyings(f):
    a,b,c,d,e = f
    if not c == e:
        A = np.array([[-2*(c/2),-d,-a],[-d,-2*(e/2),-b],[0,0,1]])
        t = np.arctan(d/(c/2-e/2))
        B1 = np.eye(3)
        B2 = np.diag((-1,-1,1))
        B3 = np.array([[ np.cos(t),  np.sin(t), 0], [ np.sin(t), -np.cos(t), 0], [0, 0, 1]])
        B4 = np.array([[-np.cos(t), -np.sin(t), 0], [-np.sin(t),  np.cos(t), 0], [0, 0, 1]])
        Bs = [B1, B2, B3, B4]
        Ys = np.array([np.matmul(B,A) for B in Bs])
        Yabcdes = np.array([[-Y[0,2], -Y[1,2], -Y[0,0], -Y[0,1], -Y[1,1]] for Y in Ys])
        return Yabcdes
    else:
        print("Fourfold orbit is degenerate because c=e.")
        return np.matlib.repmat(0*f,4,1)


#####################################################################################
#####################################################################################
    
    
def ispp(y):  # can feed in only one y vector here, each of size 3
    return np.minimum(y[0]*y[2]-y[1]**2,y[0]+y[2])


#####################################################################################
#####################################################################################
    
    
def pponly(f): #feed in only a single f vector here, each of size 5
    all4 = getyings(f)
    msk1 = np.array([ispp(a4) for a4 in all4[:,2:]])
    ppo1 = all4[msk1 > 0]
    return np.array(ppo1[0]) if len(ppo1)>0 else 0*f


#####################################################################################
#####################################################################################
    
    
def importdata(dataname,pickthisIf):
    """load real data I-value and ground truth f-value. this should come from ... /Experiments/Other\ Notebooks/RealDataProcessing.nb"""

    directory = str(os.getcwd()+"/Data/"+dataname)
    StrgI  = open(directory+"_I.txt", "r").read()
    Strgf  = open(directory+"_f.txt", "r").read()
    ListI  = ast.literal_eval(StrgI)[pickthisIf]
    Listf  = ast.literal_eval(Strgf)
    real_I = np.delete(np.array(ListI),2)
    real_I = np.array(real_I/np.linalg.norm(real_I))          #I, Ix, Ixx, Ixy, Iyy
    real_x = np.array(Listf)[pickthisIf,:2]                   #fx,fy,
    real_y = np.array(Listf)[pickthisIf,2:]                   #       fxx, fxy, fyy
    real_I = np.reshape(real_I, (1,5)).astype(np.float32)
    real_x = np.reshape(real_x, (1,2)).astype(np.float32)
    real_y = np.reshape(real_y, (1,3)).astype(np.float32)
    
    return real_I,real_x,real_y


#####################################################################################
#####################################################################################
    
    
def plotquadf(a):
    x, y = np.array(np.meshgrid(range(50), range(50)))/200#/1050
    quadr = lambda x0,y0: np.dot(a,np.array([x0,y0,x0**2,x0*y0,y0**2]))
    quadrfun = np.vectorize(quadr)
    return x,y,quadrfun(x,y)


#####################################################################################
#####################################################################################
    
    
def plotquadI(a):
    x, y = np.array(np.meshgrid(range(50), range(50)))/200#/1050
    b = np.insert(a,2,0) if len(a.flatten())<6 else a
    quadr = lambda x0,y0: np.dot(b,np.array([1,x0,y0,x0**2,x0*y0,y0**2]))
    quadrfun = np.vectorize(quadr)
    return quadrfun(x,y)

#####################################################################################
#####################################################################################
    
def getT(bI):
    Ix,Iy = bI[1],bI[2]
    tsol = -1j * np.log((Ix - 1j*Iy) / np.sqrt(Ix**2 + Iy**2))
    return tsol


#####################################################################################
#####################################################################################


def rotateit(bI,bp,t):
    """got this from Utilities2.py's RotateIVecAndPoints function"""
    
    G11,G12,G21,G22 = np.cos(t), -np.sin(t), np.sin(t), np.cos(t)
    T1 = np.array([[G11,G12],[G21,G22]])
    T2 = np.array([[G11**2,  2*G11*G21,         G21**2],
                   [G11*G12, G12*G21 + G11*G22, G21*G22],
                   [G12**2,  2*G12*G22,         G22**2]])
    T = block_diag( [1] , T1 , inv(T2) )
    trainI = np.array([np.matmul(T,elem) for elem in bI])
    trainp = np.array([np.matmul(T[1:,1:], elem) for elem in bp])
    trainI = np.real(trainI)
    trainp = np.real(trainp)
    return trainI,trainp

#####################################################################################
#####################################################################################
    

#### approximate the intersection of your surfaces
#def intersectionness(x):
#    mysum = [np.amin(np.linalg.norm(all_cand[i,:,2:]-x,axis=1)) for i in [0,1,2,3]]
#    return np.sum(np.array(mysum))
#x0 = np.zeros(3)#5)
#optsol=optimize.minimize(intersectionness, x0, method="BFGS")
#print(optsol.x)

#####################################################################################
#####################################################################################
    
def get_x(real_x,nbhd_sz,nbhd_gr):
    a = np.linspace(real_x[0,0]-nbhd_sz,real_x[0,0]+nbhd_sz,num=nbhd_gr)
    b = np.linspace(real_x[0,1]-nbhd_sz,real_x[0,1]+nbhd_sz,num=nbhd_gr)
    a,b = np.array(np.meshgrid(a,b))
    nbhd_x = np.transpose(np.array((a.flatten(), b.flatten())))
    return nbhd_x


#####################################################################################
#####################################################################################
    
    
def exploit_symmetries(real_I,real_f,info=[],verbose=False):
    """exploit symmetries to reduce I,f dimensions.
    takes a single pair of vectors at a time.
    if verbose==True, print whether it satisfies the KZs at that step."""

    if len(info)>1: #you're using an I vector you've called this on previously, just with a different f
        initI,real_t,sc1,sc2 = info
        dummy = real_I.reshape(1,-1)
        dummy,real_f0 = rotateit(dummy, real_f.reshape(1,-1), real_t)
        real_f0 = real_f0[0]
        real_f0 = np.multiply((1,1,1/sc1,1/np.sqrt(np.abs(sc1*sc2)),1/sc2),real_f0)
        real_I0 = dummy[0]
        #print("satisfies KZs 0 during ES:       ", np.round(evalKZs(real_I0,real_f0),15))
        
    else: #you're computing this for a new I vector.
        ##divide by I to make I=1.
        initI = real_I[0]
        real_I0 = real_I/initI
        real_f0 = real_f
#        print("f after isotrop",real_f0)

        if verbose:
            print("satisfies KZs 1:     ", np.round(evalKZs(real_I0, real_f0),15))

        ##rotate to make Iy=0,Ix>0.
        real_t = getT(real_I0)
        real_I0,real_f0 = rotateit(real_I0.reshape(1,-1), real_f0.reshape(1,-1), real_t)
        real_I0,real_f0 = real_I0.ravel(),real_f0.ravel()
        real_I0rot = real_I0
#        print("f after rotatio",real_f0)

        if verbose:
            print("satisfies KZs 2:     ", np.round(evalKZs(real_I0, real_f0),15))

        ##anisotropically scale to make Ix=1, Iyy=-1.
        sc1,sc2 = real_I0[1],np.sqrt(np.abs(real_I0[5]))
        real_I0 = np.multiply((1,1/sc1,1/sc2,1/(sc1**2),1/(sc1*sc2),               1/(sc2**2)),real_I0)
        real_f0 = np.multiply((  1,    1,    1/sc1,     1/np.sqrt(np.abs(sc1*sc2)),1/sc2),     real_f0)
#        print("f after anisotr",real_f0)
#        print(" ")
        if verbose:
            print("satisfies KZs 3:     ", np.round(evalKZs(real_I0, real_f0),15))

        info = initI,real_t,sc1,sc2 #data that allows one to decompress later
        
    return np.array(real_I0),np.array(real_f0),info


#####################################################################################
#####################################################################################
        

def inv_exploit_symmetries(real_I0,real_f0,info,verbose=False):
    """takes a single pair of vectors (in R^6 and R^5) at a time.
    if verbose==True, print whether it satisfies the KZs at that step."""
    
    real_I0 = np.asarray(real_I0)
    real_f0 = np.asarray(real_f0)

    ##UNDO anisotropically scale to make Ix=1, Iyy=-1.
    sc1,sc2 = 1/info[2],1/info[3]
    real_I0 = np.multiply((1,1/sc1,1/sc2,1/sc1**2,1/(sc1*sc2),1/sc2**2),real_I0)
    real_f0 = np.multiply((1,1,1/sc1,1/np.sqrt(np.abs(sc1*sc2)),1/sc2),real_f0)
#    print("f undon anisotr",np.real(real_f0))

    if verbose:
        print("satisfies KZs 3:     ", np.round(evalKZs(real_I0, real_f0),15))

    ##UNDO rotate to make Iy=0,Ix>0.
    real_I0,real_f0 = rotateit(real_I0.reshape(1,-1), real_f0.reshape(1,-1), -info[1])
    real_I0,real_f0 = real_I0[0],real_f0[0]
#    print("f undon rotatio",real_f0)
    if verbose:
        print("satisfies KZs 2:     ", np.round(evalKZs(real_I0, real_f0),15))

    ##UNDO divide by I to make I=1.
    real_I0 = info[0]*real_I0
    real_f0 = real_f0
#    print("f undon isotrop",real_f0,"\n")

    if verbose:
        print("satisfies KZs 1:     ", np.round(evalKZs(real_I0, real_f0),15))

    return np.array(real_I0),np.array(real_f0)
    

#####################################################################################
#####################################################################################
        
def analytic_rot(I0,f0):
    
    I,Ix,Iy,Ixx,Ixy,Iyy = I0
    fx,fy,fxx,fxy,fyy   = f0
    
    In = Ix**2+Iy**2
    
    I1 = \
    ( I, np.sqrt(In), 0, \
    ( Ix**2*Ixx + 2*Ix*Iy*Ixy + Iy**2*Iyy )/In, \
    ( (Ix**2-Iy**2)*Ixy + Ix*Iy*(Iyy-Ixx) )/In, \
    ( Ix**2*Iyy - 2*Ix*Iy*Ixy + Iy**2*Ixx )/In \
    )
    
    f1 = ( \
    (fx*Ix+fy*Iy)/np.sqrt(In), \
    (fy*Ix-fx*Iy)/np.sqrt(In), \
    ( fxx*Ix**2 + 2*fxy*Ix*Iy + fyy*Iy**2 )/In, \
    ( fxy*(Ix**2-Iy**2) + (fyy-fxx)*Ix*Iy )/In, \
    ( fyy*Ix**2 - 2*fxy*Ix*Iy + fxx*Iy**2 )/In \
    )
    
    return I1,f1


#####################################################################################
#####################################################################################
    

def set_unit(opt):

    if opt==1:
        Z   = (XX**3+YY**3)/1000
        dZx = 3*XX**2/1000
        dZy = 3*YY**2/1000
        dZxx = 6*XX/1000
        dZxy = 0*np.ones(XX.shape)
        dZyy = 6*YY/1000
    elif opt==2:
        Z   = (XX**2+YY**2)/100
        dZx = (2*XX)/100
        dZy = (2*YY)/100
        dZxx = 2*np.ones(XX.shape)/100
        dZxy = 0*np.ones(XX.shape)
        dZyy = 2*np.ones(XX.shape)/100
    elif opt==3:
        Z   = (XX+YY**2)/100
        dZx = np.ones(XX.shape)/100
        dZy = (2*YY)/100
        dZxx = 0*np.ones(XX.shape)/100
        dZxy = 0*np.ones(XX.shape)
        dZyy = 2*np.ones(XX.shape)/100
    
    return Z,dZx,dZy,dZxx,dZxy,dZyy

#####################################################################################
#####################################################################################

def valid(f):
    """mask out the NaNs -- set them to zero."""
    return np.nan_to_num(np.multiply(M, f))

#####################################################################################
#####################################################################################

def normalize(A):
    return (A-np.mean(A,axis=0))/np.var(A,axis=0)
   
#####################################################################################
#####################################################################################

from Utilities_PDGSDG import *
from Utilities_TDG import *

def DataLoadingMain(starttime,ttfrac,toyopt,size0,subset,uniquetime):

    verbose  = False
    fastbeta = False  # this is buggy but might speed up computation when it works

    pruning1 = False
    pruning2 = False
    pruning3 = True

    startind = 0
    I0,X0,Y0 = loaddata(toyopt,size0,subset,uniquetime,startind) ##KH FUNCTION
    F0 = np.concatenate([X0,Y0],axis=1)
#    [print("satisfies KZs 0:       ", np.round(evalKZs(   np.concatenate([F0[t], I0[t]])    ),15)) for t in range(30)]

    plotcluster = np.real(Y0)  # np.real(Y_[np.abs(np.real(Y_[:,1])-2)>1])
#    print("\n",plotcluster)
    fig8 = plt.figure()
    axL9 = fig8.add_subplot("110", projection='3d')
    axL9.scatter3D(plotcluster[:,0],plotcluster[:,1],plotcluster[:,2],s=2)
    plt.title("Uncompressed variety samples.")
    axL9.set_xlabel("fxx")
    axL9.set_ylabel("fxy")
    axL9.set_zlabel("fyy")
    axL9.autoscale()
#    plt.show()

    if toyopt == 0 or toyopt == 5: #if NOT test data, do symmetry stuff
        if toyopt ==5:
            pruning1,pruning2,pruning3 = False,False,False
        if pruning1:
            ##optional: could trim F0,I0 to remove too-large c,d,e points and ease training
            if verbose:
                print("...pruning for small-ish Y before compressing...")
            prunethresh = 100000#10
            prunecde = np.arange(len(I0))[np.amax(np.abs(F0[:,2:]),axis=1)<prunethresh]
            I0 = I0[prunecde]
            F0 = F0[prunecde]
        if pruning2:
            ##optional: for some reason, some of these don't solve the KZs. Remove them.
            if verbose:
                print("...pruning for KZ=0...")
            KZgoodnessthreshold=5
            goodinds = np.asarray(
                np.where(
                    [np.max(
                        np.abs(
                            evalKZs(
                                np.concatenate([f,i])
                            )
                        )
                    )<=KZgoodnessthreshold for f,i in zip(F0,I0)]
                )
            )
            I0 = I0[goodinds][0]
            F0 = F0[goodinds][0]

        ############################################

        ############################################
        #COMPRESSION OF I-VECTORS

        if fastbeta:    #Compression of I vectors -- new as of 01/2020.
                        #Being smarter about computations.
            #group into distinct I's
            uniqueIs = np.unique([tuple(row) for row in I0],axis=0)
            indsuqIs = np.array([np.where((I0==elem).all(axis=1)) for elem in uniqueIs])
            justuniqueinds = [k[0][0] for k in indsuqIs]
            if verbose:
                print(I0.shape,F0.shape," I,F shapes")
                print("finished grouping data\n")

            IFT0 = np.stack([exploit_symmetries(I0[t],F0[t],[],False) for t in justuniqueinds])
            Iout = np.stack(IFT0[:,0])
            Fout = np.stack(IFT0[:,1])
            Ic = np.stack(IFT0[:,0])
            info0 = np.stack(IFT0[:,2])
            I = np.concatenate(np.array([repmat(Ic[t],len(indsuqIs[t][0]),1) for t in range(len(info0))]))

            #TODO: Fix this chunk.
            T = np.concatenate(np.array([repmat(info0[t],len(indsuqIs[t][0]),1) for t in range(len(info0))])) ##not being used yet. to test reconstr make sure (I,T) --> I0.
            newordering = np.concatenate(np.concatenate(indsuqIs))
            F = np.array([exploit_symmetries(I0[newordering[t]], F0[newordering[t]], T[t], False)[1] for t in range(len(T))])
            X,Y = F[:,:2],F[:,2:]
            print("F",F)
            
        else: #slower but safer way to compress I
            IFT0 = np.stack([exploit_symmetries(i,f,[],False) for i,f in zip(I0,F0)])
            I,F,info0 = np.stack(IFT0[:,0]),np.stack(IFT0[:,1]),np.stack(IFT0[:,2])
            X,Y = F[:,:2],F[:,2:]
        
        #Project onto the nontrivial dimensions of I
        I = I[:,3:5]

        if verbose:
            print(I.shape,F.shape," I,F shapes")
            print("finished compressing data\n")

        if pruning3:
            if verbose:
                print("...pruning for smallish Y after compressing...")
            prunethresh = 5
            prunecde = np.arange(len(I))[np.amax(np.abs(Y),axis=1)<prunethresh]
            I = I[prunecde]
            X = X[prunecde]
            Y = Y[prunecde]

        uniqueIs = np.unique([tuple(row) for row in I],axis=0)
        indsuqIs = np.array([np.where((I==elem).all(axis=1)) for elem in uniqueIs])
        justuniqueinds = [k[0][0] for k in indsuqIs]
        if verbose:
            print("Number of unique I vectors, after compression and pruning: ",len(indsuqIs))
        plotcluster = [np.real(Y[t[0].ravel()]) for t in indsuqIs]
        fig8 = plt.figure()
        axL9 = fig8.add_subplot("110", projection='3d')
        for p in plotcluster:
            axL9.scatter3D(p[:,0],p[:,1],p[:,2],s=2)
        plt.title("Compressed variety samples.")
        axL9.set_xlabel("fxx")
        axL9.set_ylabel("fxy")
        axL9.set_zlabel("fyy")
        axL9.autoscale()
#        plt.show()

        scaletodiscriminate=10
        Y = Y*scaletodiscriminate

        X = np.round(X,decimals=15)
        Y = np.round(Y,decimals=15)
        I = np.round(I,decimals=15)
    else: #if toy data
        scaletodiscriminate=1
        I = I0
        X = X0
        Y = Y0


    ############################################
    ############################################
    #DATA PROCESSING FOR NETWORK
#    X,I = normalize(X), normalize(I)
    X,Y,I, X_,Y_,I_, X_t,Y_t,I_t, idx0,train_stop,test_stop = splittraintest(X,Y,I,ttfrac,toyopt,subset)
    
    datatime = time.time()
    #print("\nData preprocessing time: ", datatime - starttime)

    dataout   = X ,  Y ,  I
    dataout_  = X_,  Y_,  I_
    dataout_t = X_t, Y_t, I_t
    etcout    = idx0,train_stop,test_stop,datatime,scaletodiscriminate,info0

    return dataout,dataout_,dataout_t,etcout

