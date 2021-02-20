
#from TestOnRealData4.py
#i really hope that when I give the system the ground truth's (a,b), it can spit out the ground truth (c,d,e)... otherwise we have a (noise?) problem.

###############################################################
#module loading
###############################################################

from HelperFunctions import *
import matplotlib, ast, sys, time, os, socket, warnings
KHcomputer = ("dhcp" and ".harvard.edu" in socket.gethostname()) or (socket.gethostname() == "Kathryns-MacBook-Pro-2.local")
if not KHcomputer: #if not KH's MBP
    matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
if KHcomputer: #if KH's MBP
    from tensorflow.python.util import deprecation
    deprecation._PRINT_DEPRECATION_WARNINGS = False
    try:
        from tensorflow.python.util import module_wrapper as deprecation2
    except ImportError:
        from tensorflow.python.util import deprecation_wrapper as deprecation2
    deprecation2._PER_MODULE_WARNING_LIMIT = 0
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation
from matplotlib.pyplot import *
from mpl_toolkits.mplot3d import *
np.set_printoptions(suppress=True)
from time import time
from numpy import genfromtxt, matlib
from numpy.random import randint
print("\nfinished loading packages")


###############################################################
###############################################################
###############################################################
# from TORD.py
###############################################################
###############################################################
###############################################################

###############################################################
#paramter choice
###############################################################

#"1571426425.884375" and "1500"
#"1569372194.313534" and "5000"
modelname   = "1572314742.3791122"#   #"1569212026.0217707" #"1569258849.5705116"
iteration   = "49500"              #   #"49500"#"3300"#
noisevar    = "0_000"
directoryin = str(os.getcwd()+"/Data/Noise_"+noisevar+"/from_036051")
nbhd        = True
nbhd_gr     = 50
nbhd_sz     = 1
printmore   = False
noiseon     = False
l = "1" #single fixed light source


###############################################################
#data reading
###############################################################

real_f  = np.array(ast.literal_eval(open(directoryin+"_f.txt", "r").read()))
real_x = real_f[:,:2]
real_y = real_f[:,2:]
real_I = np.array(ast.literal_eval(open(directoryin+"_I_"+l+".txt", "r").read()))
real_t = np.array([getT(i) for i in real_I])

numplots = 4
fig2 = plt.figure(figsize=(10,4))
grid = plt.GridSpec(2,4, wspace=0.1, hspace=0.1)

for i in [0]:#range(numplots): #number of different surface f's

    #plot a particular point on each graph
    real_I0,real_f0 = rotateit(real_I[i].reshape(1,-1), real_f[i].reshape(1,-1), real_t[i])
    real_x0  = real_f0[0,:2]
    real_y0  = real_f0[0,2:]
    real_I0 = np.delete(real_I0,2)
    real_I0 = real_I0/np.linalg.norm(real_I0)
    
#    if noiseon:
#        mean = 0
#        vari = .1
#        real_I0 = real_I0 + np.random.normal(mean,vari,real_I0.shape)

    nbhd_x = get_x(real_x0.reshape(1,-1),nbhd_sz,nbhd_gr)
    nbhd_y = np.full((len(nbhd_x),3),1)              #placeholder
    nbhd_I = np.matlib.repmat(real_I0, len(nbhd_x),1) #just a bunch of repeats

    ###############################################################
    #graph setup
    ###############################################################

    ##import pre-trained model #k iteration #l
    ##this should come from ... /Experiments/Outputs/ONESTEP/k/iterl.*
    directory = str(os.getcwd()+"/Outputs/ONESTEP/"+modelname+"/iter"+iteration)
    metadirec = directory+"/model.meta"
    imported_graph=tf.get_default_graph()
    loader = tf.train.import_meta_graph(metadirec)
    y         = imported_graph.get_tensor_by_name('out_h:0')
    x         = imported_graph.get_tensor_by_name('inp_h:0')
    I         = imported_graph.get_tensor_by_name('inp_g:0')
    y_out     = imported_graph.get_tensor_by_name('layo_h:0')

    ###############################################################
    #network evaluation session
    ###############################################################

    with tf.Session(graph=imported_graph) as sess:
        loader.restore(sess, directory+"/model")
        _y_out = sess.run(y_out,  {x: real_x0.reshape(1,-1), I: real_I0.reshape(1,-1), y: real_y0.reshape(1,-1)})
        _y_nhbd_out=sess.run(y_out, {x: nbhd_x, I: nbhd_I, y: nbhd_y})
        if printmore:
            print("\ny from NN: \n",_y_out,"\n actual y:  \n",real_y,"\n candidate y:  \n",_y_nhbd_out)

    #these are rotated to Iy=0 still...
    candidatesf = np.concatenate((nbhd_x, _y_nhbd_out), axis=1)
    truethisf = np.concatenate((real_x0,real_y0),axis = 0)
    minethisf = np.concatenate((real_x0,_y_out.flatten()),axis = 0) #my guess for c,d,e using the oracle's a,b

    print(candidatesf)


    ###############################################################
    #printing and plotting outputs
    ###############################################################


    ##################
    trueIfi = truethisf.reshape(1,-1)
    candIfi = np.concatenate((nbhd_x, _y_nhbd_out), axis=1)
    distsi  = pairwise_distances(trueIfi[:,:],candIfi[:,:])
    minindi = np.unravel_index(np.argmin(distsi, axis=None), distsi.shape)
    bestthisf = candIfi[minindi[1]]

    ################## rotate 'em back.
    rotI,best_rotf = rotateit(np.insert(real_I0,2,0).reshape(1,-1),bestthisf.reshape(1,-1),-real_t[i])
    rotI,true_rotf = rotateit(np.insert(real_I0,2,0).reshape(1,-1),truethisf.reshape(1,-1),-real_t[i])
    print("\nIvec:" , rotI)
    print("true:" , true_rotf)
    print("mine:" , best_rotf)

    ################## plots!
    pm = 0 if i<2    else 1
    pn = 0 if i%2==1 else 1
    pm,pn = int(pm),int(pn)

    ax3 = fig2.add_subplot(grid[pm,pn])
    ax3.imshow(plotquadI(rotI), cmap=cm.Greys_r)
    axL1 = fig2.add_subplot(grid[pm,pn+2], projection='3d')
    XYZ = plotquadf(best_rotf)
    axL1.plot_surface(XYZ[0],XYZ[1],XYZ[2])
    XYZ = plotquadf(true_rotf)
    axL1.plot_surface(XYZ[0],XYZ[1],XYZ[2])

plt.show()

print()
