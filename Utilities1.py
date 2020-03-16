
## Called by Training.py


import matplotlib, ast, sys, time, os
#matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.pyplot import *
from mpl_toolkits.mplot3d import *
np.set_printoptions(suppress=True)
from time import time
from scipy.linalg import block_diag
from numpy.linalg import inv
import glob, os
from numpy.matlib import repmat


def getVars(name,vars): ## FOR GETTING L1 (etc.) WEIGHTS ##
    for i in range(len(vars)):
        if vars[i].name.startswith(name):
            return vars[i]
    return None

def saveAll(saveall,numIters,codeis,prefix,costvec,sess,saver,signature,runtype,alloutput):
    if saveall==1:
        np.save(prefix+"Outputs/allcosts" +codeis, costvec[1:len(costvec)-1])
        np.save(prefix+"Outputs/signatures" +codeis, signature)
        saver.save(sess,"./"+prefix+"NN/Models/KH_Model_"+runtype+"_"+codeis)
    return None

def WriteTrainingOutputs(uniquenum,paramsIn):
    fname = "Outputs/ONESTEP/"+uniquenum+"/Params.txt"
    f = open(fname,"w+")
    f.write("\n\n*****************\n\nHyper Params:\n")
    strg = ["\nDataset identifier:   ","\nNumber of iterations:   ","\nStep size:   ","\nBatch size:   ","\nDepth of h:   ","\nWidth of h:   ","\nDepth of g:   ","\nWidth of g:   ","\nTest/Train ratio:   ","\ntoyopt:   ","\nsubset:   ","\nsize0:   ","\nAdam B1:   "]
    B = [s+str(n) for s,n in list(zip(strg,paramsIn))]
    f.write(''.join(B))
    return fname

def PrintParams(uniquenum,paramsIn):
    print("\n\n*****************\n\nHyper Params:\n")
    strg = ["Dataset identifier:   ","Number of iterations:   ","Step size:   ","Batch size:   ","Depth of h:   ","Width of h:   ","Depth of g:   ","Width of g:   ","Test/Train ratio:   ","toyopt:   ","subset:   ","size0:   ","Adam B1:   "]
    B=[print(s+str(n)) for s,n in list(zip(strg,paramsIn))]

def prettyprint(data,args):
    print("")
    frmt = ("{:>35.6f}"+"{:>35.6f}")
    print(("{:>35}"*2).format(*args))
    st = "----------"
    print(("{:>35}"*2).format(*[st,st]))
    for i in data:
        print(frmt.format(*i))

def animate(i,argl):
    ax     = argl[0]
    Y_     = argl[1]
    yo_all = argl[2]
    
    yplt = yo_all[i]
    ax.clear()
    sc2 = ax.scatter3D(yplt[:,0],yplt[:,1],yplt[:,2])
    sc3 = ax.scatter3D(  Y_[:,0],  Y_[:,1],  Y_[:,2])
    ax.set_title('NN Results.\nEPOCH: '+str(i)+' of '+str(len(yo_all))+'.')
    ax.set_xlabel('f_{xx}')
    ax.set_ylabel('f_{xy}')
    ax.set_zlabel('f_{yy}')
    ax.legend([sc2, sc3], ['Training Output from NN', 'Ground Truth'], loc='lower left', numpoints = 1)

def toysave(I,X,Y):
    np.save("tempI_TOY.npy",  I)
    np.save("tempAB_TOY.npy", X)
    np.save("tempCDE_TOY.npy",Y)

def isdiverging(yo_):
    if np.isnan(yo_).any():
        print("\n\n")
        print("ERROR: Initialization failed. You're getting NaNs.\nRetry, and if problem persists, decrease step size.\n")
        sys.exit()

def lintran(z,A,c):
    tmp = tf.reshape(z,(tf.shape(z)[0],1,tf.shape(z)[1]))
    return tf.reshape(tf.matmul(tmp,A),tf.shape(c)) + c

#####################################################################################
#####################################################################################
#params = [uniquetime, numIters, stepsize, batch_size, "(fixed) one hidden layer",
#          width_h, depth_g, width_g, ttfrac, toyopt, subset, size0]
#make sure this still matches ParallelOS.py file!

def network(hyperparams,nvals,datatime,i,x,y,varsW,varsb):
    
    uniquetime, numIters, stepsize, batch_size, depth_h, width_h, depth_g, width_g, ttfrac, toyopt, subset, size0, Adamb1 = hyperparams #unpacking params
    ngI,ngT,nhX,nhY = nvals
    
    ###########################################
    #############THIS IS THE G NETWORK###########
    ############################################

    if len(varsW)==0: #if creating a new graph
        print("Creating new random inits for weights and biases in HelperFunctions.")
        # TF Variables are our neural net parameter tensors, we initialize them to random (gaussian) values in
        # Layer1. Variables are allowed to be persistent across training epochs and updatable bt TF operations
        initW1 = tf.random.normal([ngI,     width_g],       mean=0, stddev=2)
        initb1 = tf.random.normal([width_g, 1],             mean=0, stddev=2)
        if depth_g>1:
            initWi = tf.random.normal([width_g, width_g],   mean=0, stddev=2)
            initbi = tf.random.normal([width_g  ],          mean=0, stddev=2)
        initWo = tf.random.normal([width_g, ngT],           mean=0, stddev=2)
        initbo = tf.random.normal([1,       ngT],           mean=0, stddev=2)

        W1 = tf.Variable(initW1, name='w1_g')
        b1 = tf.Variable(initb1, name='b1_g')
        t1 = tf.tensordot(i, W1, axes=1) + tf.transpose(b1) if width_g>1 else tf.matmul(i, W1) + b1
        t1 = tf.nn.relu(t1, name='lay1_g')

        # hidden layer weights and biases
        ti=t1
        for k in range(depth_g-2): #will not execute if depth_g==1
            Wi = tf.Variable(initWi, name='wi_g')
            bi = tf.Variable(initbi, name='bi_g')
            ti = tf.tensordot(ti, Wi,axes=1) + bi if width_g>1 else tf.multiply(ti, Wi) + bi
            ti = tf.nn.relu(ti, name='layi_g')

        # output layer weights and biases
        Wo = tf.Variable(initWo, name='wo_g')
        bo = tf.Variable(initbo, name='bo_g')
        to = tf.tensordot(ti, Wo, axes=1) + bo if width_g>1 else tf.multiply(ti, Wo) + bo
        to = tf.identity(to, name='layo_g')
    
    
    else: #if restoring an existing graph, whose variable values you're passing into this function as params
        
        print("Restoring existing graph in HelperFunctions\n")
        
        W1 = varsW[0]
        b1 = tf.transpose(varsb[0])
        t1 =tf.tensordot(i, W1,axes=1) + b1 if width_g>1 else tf.matmul(i, W1) + b1
        t1 = tf.nn.relu(t1, name='lay1_g')
        
        # hidden layer weights and biases
        ti=t1
        for k in range(depth_g-2): #will not execute if depth_g==1
            Wi = varsW[k+1]
            bi = varsb[k+1]
            ti = tf.tensordot(ti, Wi,axes=1) + bi if width_g>1 else tf.multiply(ti, Wi) + bi
            ti = tf.nn.relu(ti, name='layi_g')
        
        # output layer weights and biases
        Wo = varsW[len(varsW)-1]
        bo = varsb[len(varsb)-1]
        to = tf.tensordot(ti, Wo,axes=1) + bo if width_g>1 else tf.multiply(ti, Wo) + bo
        to = tf.nn.relu(to, name='layo_g')

    gtime = time()
    print("g-network setup time: ", gtime - datatime)

    ###########################################
    #############THIS IS THE H NETWORK###########
    ###########################################

    start = 0
    V1 = tf.gather(to,np.arange(start,start+nhX*width_h),axis=1)
    start = start + nhX*width_h
    Vo = tf.gather(to,np.arange(start,start+nhY*width_h),axis=1)
    start = start + nhY*width_h
    c1 = tf.gather(to,np.arange(start,start+width_h),axis=1)
    start = start + width_h
    co = tf.gather(to,np.arange(start,start+nhY),axis=1)

    V1 = tf.reshape(V1,(tf.shape(x)[0],nhX,width_h)) ##PROBLEM
    Vo = tf.reshape(Vo,(tf.shape(x)[0],width_h,nhY))

    ## this used to be the yucky ad hoc part at the end of this doc
    y1 = tf.nn.relu( lintran(x, V1,c1), name='lay1_h')
    yo = tf.identity(lintran(y1,Vo,co), name='layo_h')

    ###########################################
    #############SETUP OF LOSS & OPT###########
    ###########################################

    htime = time()
    print("h-network setup time: ",htime - gtime )

    return htime, yo,to,t1,W1, b1,Wo,bo,to



#####################################################################################
#####################################################################################
def loaddata(toyopt,size0,subset,uniquetime,startind):

    #if dataset is bigger than 10 this is a good idea.
    if toyopt==1:
        #toy1, random
        X = np.random.randint(10,size=(size0,2))
        I = np.random.randint(10,size=(size0,6))
        Y = np.random.randint(10,size=(size0,3))
        toysave(I,X,Y)
    elif toyopt==2: #stepsize=1e-3
        X = np.random.uniform(1,4,size=(size0,3))
        I = np.random.uniform(1,4,size=(size0,2))
        Y = np.array([[x,x+i,x*i] for x,i in zip(np.sum(I,axis=1),np.sum(I,axis=1))])
        #Y = np.array([[x,x+i,x*i] for x,i in zip(X[:,0],I[:,0])])#[:,:,0]
        #Y = np.array([[x[0]*x[2],x[1]*x[1]*x[2]+i[0],x[0]*i[1]] for x,i in zip(X,I)])
        #Y = np.array([[x[1],i[1],x[1]+i[1]] for x,i in zip(X,I)])
        toysave(I,X,Y)
    elif toyopt==3: #stepsize=1e-2
        rn = 1;
        X, I = np.mgrid[-rn:(rn):10j, -rn:(rn):10j]
        X = X.flatten().reshape((X.shape[0]*X.shape[1],1))
        I = I.flatten().reshape((I.shape[0]*I.shape[1],1))
        Y = I*X+10
        toysave(I,X,Y)
    elif toyopt==0 or toyopt==5:#then use real data!!!! imported from MainNNA.py
        #prefix = "~/Dropbox/Research/Experiments/"
        prefix = ""
        if toyopt==0:
#            FileI = prefix+"Data/into_NNA_Is_"+uniquetime+".txt"
#            Filef = prefix+"Data/into_NNA_fs_"+uniquetime+".txt"

            ListI,Listf = parsedata()
            
            sn = int(Listf.shape[1]/5)
            ListI = np.asarray([repmat(l,sn,1) for l in ListI])
            ListI = np.concatenate(ListI)
            print("I shape" ,ListI.shape)
            Listf = Listf.reshape(Listf.shape[0],sn,5)
            Listf = np.concatenate(Listf)
            print("f shape",Listf.shape)

            
        elif toyopt==5:
            FileI = prefix+"Data/Synth/I.csv"
            Filef = prefix+"Data/Synth/F.csv"
            StrgI = open(FileI, "r").read()
            Strgf = open(Filef, "r").read()
            ListI = np.array(ast.literal_eval(StrgI))
            Listf = np.array(ast.literal_eval(Strgf))
        selectcols=[0,1,2,3,4,5]#[0,1,3,4,5]
        if subset:
            subnum=size0
            I = ListI[startind:(startind+subnum),selectcols]
            X = Listf[startind:(startind+subnum),:2]
            Y = Listf[startind:(startind+subnum),2:]
        else:
            I = ListI[:,selectcols]
            X = Listf[:,:2]
            Y = Listf[:,2:]
    return I,X,Y
    
def parsedata():

    print("\n\nThe Current Working Directory:\n ",os.getcwd(),"\n\n")

    #scrape all files
    HoldI,Holdf = [],[]
    filepath = os.getcwd()#"/Users/Heal/Dropbox/Research/Experiments"
    os.chdir(filepath + "/Data")
    print("Reading Training Data:")
    for f,i in zip(np.sort(glob.glob("F_*")),np.sort(glob.glob("I_*"))):
        print(f,i)
        ListI = np.asarray([[float(g) for g in h.split(',')] for h in np.loadtxt(i,dtype='str')])
        Listf = np.asarray([[float(g) for g in h.split(',')] for h in np.loadtxt(f,dtype='str')])
        if HoldI == []:
            HoldI = [ListI]
            Holdf = [Listf]
        else:
            HoldI = np.append(HoldI,[ListI],axis=1)
            Holdf = np.append(Holdf,[Listf],axis=1)
    ListI = np.concatenate(HoldI)
    Listf = np.concatenate(Holdf)
    print("\n\n")
    os.chdir(filepath)
    return ListI,Listf


def splittraintest(X,Y,I,ttfrac,toyopt,subset):
    #set train fraction & shuffle dataset
    np.random.seed(707)
    idx=np.arange(X.shape[0])
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

def getyings(f):
    a,b,c,d,e = f
    if not f[2] == f[4]:
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
        print("Ying failed because c=e.")
        return np.matlib.repmat(0*f,4,1)

def ispp(y): #can feed in only one y vector here, each of size 3
    return np.minimum(y[0]*y[2]-y[1]**2,y[0]+y[2])

def pponly(f): #feed in only a single f vector here, each of size 5
    all4 = getyings(f)
    msk1 = np.array([ispp(a4) for a4 in all4[:,2:]])
    ppo1 = all4[msk1 > 0]
    return np.array(ppo1[0]) if len(ppo1)>0 else 0*f


##load real data I-value and ground truth f-value.
##this should come from ... /Experiments/Other\ Notebooks/RealDataProcessing.nb
def importdata(dataname,pickthisIf):
    directory = str(os.getcwd()+"/Data/"+dataname)
    StrgI  = open(directory+"_I.txt", "r").read()
    Strgf  = open(directory+"_f.txt", "r").read()
    ListI  = ast.literal_eval(StrgI)[pickthisIf]
    Listf  = ast.literal_eval(Strgf)
    real_I = np.delete(np.array(ListI),2)
    real_I = np.array(real_I/np.linalg.norm(real_I))                             #I, Ix, Ixx, Ixy, Iyy
    real_x = np.array(Listf)[pickthisIf,:2]                                      #fx,fy,
    real_y = np.array(Listf)[pickthisIf,2:]                                      #       fxx, fxy, fyy
    real_I = np.reshape(real_I, (1,5)).astype(np.float32)
    real_x = np.reshape(real_x, (1,2)).astype(np.float32)
    real_y = np.reshape(real_y, (1,3)).astype(np.float32)
    return real_I,real_x,real_y



##############################
##so the weights/biases don't get retrained.
#for i in range(4): #idk why I have to do this multiple times to clear all the variables out of the "trainable" set
#    all_trainable = tf.get_collection_ref(tf.GraphKeys.TRAINABLE_VARIABLES)
#    for rem in all_trainable:
#        all_trainable.remove(rem)
#
#imp_gdef = imported_graph.as_graph_def()
#
#y_out, = tf.import_graph_def(imp_gdef, input_map={"inp_h:0": x_var, "inp_g:0":I, "out_h:0":y}, return_elements=["layo_g:0"], name=None) #what I get out of my *fixed, saved* NN

#### Yucky ad hoc case-handling because I don't know how tensorflow implements multiplication when input is fed into the system as an entire batch
#if ngI==1:
#    #        print("in here 1")
#    tmp = tf.reduce_sum(x*V1,1) + tf.reshape(tf.transpose(c1), [-1])
#    if nhX==1 and ngI>1:
#        #        print("in here 2")
#        tmp = x*V1 + c1
#    if ngI>1 and nhX>1:
#        #        print("in here 3")
#        tmp = lintran(x,V1,c1) if width_h>1 else tf.reduce_sum(x*V1,1) + tf.reshape(tf.transpose(c1), [-1])
#    y1 = tf.nn.relu(tmp, name='lay1_h')
#
#if ngI>1 and nhX>1:
#    #        print("in here 4")
#    yo = lintran(y1,Vo,co) if width_h>1 else y1*Vo+co
#    else:
#        #        print("in here 5")
#        yo = lintran(y1,Vo,co)
#    yo = tf.identity(yo, name='layo_h')


def plotquadf(a):
    x, y = np.array(np.meshgrid(range(50), range(50)))/200#/1050
    quadr = lambda x0,y0: np.dot(a,np.array([x0,y0,x0**2,x0*y0,y0**2]))
    quadrfun = np.vectorize(quadr)
    return x,y,quadrfun(x,y)

def plotquadI(a):
    x, y = np.array(np.meshgrid(range(50), range(50)))/200#/1050
    b = np.insert(a,2,0) if len(a.flatten())<6 else a
    quadr = lambda x0,y0: np.dot(b,np.array([1,x0,y0,x0**2,x0*y0,y0**2]))
    quadrfun = np.vectorize(quadr)
    return quadrfun(x,y)

def getT(bI):
    Ix,Iy = bI[1],bI[2]
    tsol = -1j * np.log((Ix - 1j*Iy) / np.sqrt(Ix**2 + Iy**2))
    
#    with np.printoptions(precision=3, suppress=True):
#        print("\n",tsol,"\n",Ix,Iy)
#        print("sign tilde Iyy should be:", bI[1]**2*bI[5] - 2*bI[4]*bI[1]*bI[2] + bI[2]**2*bI[3])
        
    return tsol

def rotateit(bI,bp,t):    #got this from Utilities2.py's RotateIVecAndPoints function
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

#### approximate the intersection of your surfaces
#def intersectionness(x):
#    mysum = [np.amin(np.linalg.norm(all_cand[i,:,2:]-x,axis=1)) for i in [0,1,2,3]]
#    return np.sum(np.array(mysum))
#x0 = np.zeros(3)#5)
#optsol=optimize.minimize(intersectionness, x0, method="BFGS")
#print(optsol.x)

def get_x(real_x,nbhd_sz,nbhd_gr):
    a = np.linspace(real_x[0,0]-nbhd_sz,real_x[0,0]+nbhd_sz,num=nbhd_gr)
    b = np.linspace(real_x[0,1]-nbhd_sz,real_x[0,1]+nbhd_sz,num=nbhd_gr)
    a,b = np.array(np.meshgrid(a,b))
    nbhd_x = np.transpose(np.array((a.flatten(), b.flatten())))
    return nbhd_x

#def evalKZs(Ivec,fvec):
#    a,b,c,d,e = fvec
#    I0, Ix0, Iy0, Ixx0, Ixy0, Iyy0 = Ivec
#    p1 = c**2*I0 + b**2*c**2*I0 - 2*a*b*c*d*I0 + d**2*I0 + a**2*d**2*I0 + 2*a*c*Ix0 + 2*a**3*c*Ix0 + 2*a*b**2*c*Ix0 + 2*b*d*Ix0 + 2*a**2*b*d*Ix0 + 2*b**3*d*Ix0 + Ixx0 + 2*a**2*Ixx0 + a**4*Ixx0 + 2*b**2*Ixx0 + 2*a**2*b**2*Ixx0 + b**4*Ixx0
#    p2 = d**2*I0 + b**2*d**2*I0 - 2*a*b*d*e*I0 + e**2*I0 + a**2*e**2*I0 + 2*a*d*Iy0 + 2*a**3*d*Iy0 + 2*a*b**2*d*Iy0 + 2*b*e*Iy0 + 2*a**2*b*e*Iy0 + 2*b**3*e*Iy0 + Iyy0 + 2*a**2*Iyy0 + a**4*Iyy0 + 2*b**2*Iyy0 + 2*a**2*b**2*Iyy0 + b**4*Iyy0
#    p3 = c*d*I0 + b**2*c*d*I0 - a*b*d**2*I0 - a*b*c*e*I0 + d*e*I0 + a**2*d*e*I0 + a*d*Ix0 + a**3*d*Ix0 + a*b**2*d*Ix0 + b*e*Ix0 + a**2*b*e*Ix0 + b**3*e*Ix0 + Ixy0 + 2*a**2*Ixy0 + a**4*Ixy0 + 2*b**2*Ixy0 + 2*a**2*b**2*Ixy0 + b**4*Ixy0 + a*c*Iy0 + a**3*c*Iy0 + a*b**2*c*Iy0 + b*d*Iy0 + a**2*b*d*Iy0 + b**3*d*Iy0
#    return p1,p2,p3

#########################
#########################
##exploit symmetries to reduce I,f dimensions
#########################
#########################
def exploit_symmetries(real_I,real_f,info=[],verbose=False):
    #takes a single pair of vectors at a time.
    #if verbose==True, print whether it satisfies the KZs at that step.

#    print("satisfies KZs 0 before ES:       ", np.round(evalKZs(real_I,real_f),15))

    if len(info)>1: #you're using an I vector you've called this on previously, just with a different f
        initI,real_t,sc1,sc2 = info
        dummy = real_I.reshape(1,-1)
        dummy,real_f0 = rotateit(dummy, real_f.reshape(1,-1), real_t)
        real_f0 = real_f0[0]
        real_f0 = np.multiply((1,1,1/sc1,1/np.sqrt(np.abs(sc1*sc2)),1/sc2),real_f0)
        real_I0 = dummy[0]
#        print("satisfies KZs 0 during ES:       ", np.round(evalKZs(real_I0,real_f0),15))
    
    else: #you're computing this for a new I vector.
        ##divide by I to make I=1.
        initI = real_I[0]
        real_I0 = real_I/initI
        real_f0 = real_f
        if verbose:
            print("satisfies KZs 1:     ", np.round(evalKZs(real_I0, real_f0),15))

        ##rotate to make Iy=0,Ix>0.
        real_t = getT(real_I0)
        real_I0,real_f0 = rotateit(real_I0.reshape(1,-1), real_f0.reshape(1,-1), real_t)
        real_I0,real_f0 = real_I0.ravel(),real_f0.ravel()
        real_I0rot = real_I0
        if verbose:
            print("satisfies KZs 2:     ", np.round(evalKZs(real_I0, real_f0),15))
        
        ##anisotropically scale to make Ix=1, Iyy=-1.
        sc1,sc2 = real_I0[1],np.sqrt(np.abs(real_I0[5]))
        real_I0 = np.multiply((1,1/sc1,1/sc2,1/(sc1**2),1/(sc1*sc2),               1/(sc2**2)),real_I0)
        real_f0 = np.multiply((  1,    1,    1/sc1,     1/np.sqrt(np.abs(sc1*sc2)),1/sc2),     real_f0)
        if verbose:
            print("satisfies KZs 3:     ", np.round(evalKZs(real_I0, real_f0),15))
        
        info = initI,real_t,sc1,sc2 #data that allows one to decompress later
    
    return np.array(real_I0),np.array(real_f0),info


def inv_exploit_symmetries(real_I0,real_f0,info,verbose=False):
    #takes a single pair of vectors (in R^6 and R^5) at a time.
    #if verbose==True, print whether it satisfies the KZs at that step.

    ##UNDO anisotropically scale to make Ix=1, Iyy=-1.
    sc1,sc2 = 1/info[2],1/info[3]
    real_I0 = np.multiply((1,1/sc1,1/sc2,1/sc1**2,1/(sc1*sc2),1/sc2**2),real_I0)
    real_f0 = np.multiply((1,1,1/sc1,1/np.sqrt(np.abs(sc1*sc2)),1/sc2),real_f0)
    if verbose:
        print("satisfies KZs 3:     ", np.round(evalKZs(real_I0, real_f0),15))

    ##UNDO rotate to make Iy=0,Ix>0.
    real_I0,real_f0 = rotateit(real_I0.reshape(1,-1), real_f0.reshape(1,-1), -info[1])
    real_I0,real_f0 = real_I0[0],real_f0[0]
    if verbose:
        print("satisfies KZs 2:     ", np.round(evalKZs(real_I0, real_f0),15))

    ##UNDO divide by I to make I=1.
    real_I0 = real_I0*info[0]
    real_f0 = real_f0
    if verbose:
        print("satisfies KZs 1:     ", np.round(evalKZs(real_I0, real_f0),15))

    return np.array(real_I0),np.array(real_f0)
    

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


def valid(f): #mask out the NaNs -- set them to zero.
    return np.nan_to_num(np.multiply(M, f))
