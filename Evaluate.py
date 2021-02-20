
#used to be CVPR_Figures.py
#### PHOTOMETRIC STEREO PROOF OF CONCEPT.


############################################
############################################
#MODULE LOADING

import matplotlib, sys, time, os, socket, warnings, math
import tensorflow as tf

sys.path.insert(1, os.path.join(sys.path[0], 'Utilities'))
sys.path.insert(1, os.path.join(sys.path[0], 'Data Generation'))
from Utilities_General import *
from Utilities_Model import *
from Utilities_TDG import *

KHcomputer = ("dhcp" and ".harvard.edu" in socket.gethostname()) or (socket.gethostname() == "Kathryns-MacBook-Pro-2.local")
if KHcomputer: #if KH's MBP
    from tensorflow.python.util import deprecation
    deprecation._PRINT_DEPRECATION_WARNINGS = False
    try:
        from tensorflow.python.util import module_wrapper as deprecation2
    except ImportError:
        from tensorflow.python.util import deprecation_wrapper as deprecation2
    deprecation2._PER_MODULE_WARNING_LIMIT = 0
else: #if not KH's MBP
    matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
    
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation
from matplotlib.pyplot import *
from mpl_toolkits.mplot3d import *
from numpy.linalg import norm
np.set_printoptions(suppress=True)
from scipy import optimize
from numpy import genfromtxt, matlib

print("\nfinished loading packages")

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#END MODULE LOADING
############################################
############################################
#HYPERPARAMETER DEFINITION

from General_Parameters import *
from Evaluation_Parameters import *
from DataGen_Parameters import *

if datatype == "photometric":
    toyopt = 5
elif datatype == "co-quadratic":
    toyopt = 6
elif datatype == "training":
    toyopt = 0

#END HYPERPARAMETER DEFINITION
############################################
############################################
#DATA READING

dataout, dataout_, dataout_t, etcout = DataLoadingMain(time.time(), 1, toyopt, 0, False, "", scale_discrim, loadflags)
idx0, train_stop, test_stop, datatime, scaletodiscriminate, info = etcout
real_t = info[:, 1]
real_X, real_Y, real_I = dataout
real_f = np.real(np.concatenate((real_X, real_Y), axis=1))

#DATA READING
############################################
############################################
#SETUP TENSORFLOW ARCHITECTURE

##import pre-trained model #k iteration #l
##this should come from ... /Experiments/Outputs/ONESTEP/k/iterl.*
dirmodel = os.getcwd()+"/TrainingOutput/"+modelname
metadirec = dirmodel+"/model_iter-"+str(iteration)+".meta"
imported_graph = tf.get_default_graph()
loader = tf.train.import_meta_graph(metadirec)

I         = imported_graph.get_tensor_by_name('inp_g:0')
x         = imported_graph.get_tensor_by_name('inp_h:0')
y         = imported_graph.get_tensor_by_name('out_h:0')
y_out     = imported_graph.get_tensor_by_name('layo_h:0')

#END SETUP TENSORFLOW ARCHITECTURE
#############################################
#############################################
#BIG LIGHTING LOOP

def unflatten(z,n):
    return int((z-z%n)/n),int(z%n)

if plotson and not allplotsoff:
    figL1 = plt.figure(figsize=(10, 5))
    grid  = plt.GridSpec(2,6,wspace=0.1,hspace=0.1)
    axL1  = figL1.add_subplot(grid[:,2:], projection='3d')

count = 0
all_cand = []
for l in np.array(lights).astype(int):

    real_I0 = real_I[l]
    real_f0 = real_f[l]
    real_x0 = real_X[l]
    real_y0 = real_Y[l]
    real_i0 = info[l]
    real_t0 = real_t[l]

    if verbose:
        print("\nafter compression I:  ",real_I0)
        print("after compression f:  ",real_f0)

    #########################
    #########################
    ##do a pre-rotation on the base a,b grid
    #########################
    #########################
    nbhd_x = get_x(np.array((0,0)).reshape(1,-1),nbhd_sz,nbhd_gr)
    nbhd_y = np.full((len(nbhd_x),3),1)               #placeholder
    nbhd_I = np.matlib.repmat(real_I0, len(nbhd_x),1) #just a bunch of repeats

    #pre-rotate
    nbhd_f = np.concatenate((nbhd_x, nbhd_y), axis=1)
    dummy = np.ones((nbhd_I.shape[0],6))
    dummy, nbhd_f = rotateit(dummy, nbhd_f, real_t0)
    nbhd_x = nbhd_f[:,:2]

    ###############################################################
    #network evaluation session
    ###############################################################

    with tf.Session(graph=imported_graph) as sess:
        loader.restore(sess, dirmodel+"/model_iter-"+str(iteration))
        _y_out = sess.run(y_out, {x: real_x0.reshape(1,-1), I: real_I0.reshape(1,-1), y: real_y0.reshape(1,-1)})[0]
        _y_nhbd_out = sess.run(y_out, {x: nbhd_x, I: nbhd_I, y: nbhd_y})
        if verbose:
            print("input compressed I-vectors:  ",real_I0)
            print("input X-vectors:             ",real_x0)
            print("\n y from NN: \n", _y_out, "\n actual y:  \n", real_y0)
                #,"\n candidate y:  \n",_y_nhbd_out)

    candidatesf = np.concatenate((nbhd_x, _y_nhbd_out), axis=1)

    ###############################################################
    #printing and plotting outputs
    ###############################################################

    # then rotate back by t...
    # 2020: NO WE MUST NOW RECONSTRUCT NOT JUST ROTATE.
    # USE ALL INFO IN info0.
    expand_nbhd_I = np.insert(nbhd_I,           2,-1, axis=1)
    expand_nbhd_I = np.insert(expand_nbhd_I,    0, 0, axis=1)
    expand_nbhd_I = np.insert(expand_nbhd_I,    0, 1, axis=1)
    expand_nbhd_I = np.insert(expand_nbhd_I,    0, 1, axis=1)

    rotIFT = np.stack([np.array(inv_exploit_symmetries(i,f,real_i0)) for i,f in zip(expand_nbhd_I,candidatesf)])
    rotI,rotf_cand = np.stack(rotIFT[:,0]),np.stack(rotIFT[:,1])
        # I vector has been decompressed
    rotI,rotf_real = inv_exploit_symmetries(expand_nbhd_I[0],real_f0,real_i0)
        # I vector has been decompressed

    ##we also need to undo the "scaling to distinguish" factor we did on the y in Training.py
    rotf_real = (1,1,1/scale_discrim,1/scale_discrim,1/scale_discrim)*rotf_real
    rotf_cand = (1,1,1/scale_discrim,1/scale_discrim,1/scale_discrim)*rotf_cand

    pm = 0 if count<2    else 1
    pn = 0 if count%2==1 else 1
    pm,pn = int(pm),int(pn)

    def gammaencode(im):
        im = (im-np.amin(im))/(np.amax(im)-np.amin(im))
        return im**(1/2.2)

    if plotson and not allplotsoff:
        axL0 = figL1.add_subplot(grid[pm,pn])
        axL0.imshow(gammaencode(plotquadI(np.real(rotI))), cmap=cm.Greys_r)
        axL0.set_title("light:"+str(count))
        axL1.scatter3D(rotf_cand[:,2],rotf_cand[:,3],rotf_cand[:,4],s=2)
        axL1.scatter3D(rotf_real[2],rotf_real[3],rotf_real[4],'k',s=5)
        axL1.set_xlabel("fxx")
        axL1.set_ylabel("fxy")
        axL1.set_zlabel("fyy")
        axL1.autoscale()
        figL1.suptitle('All Varieties', fontsize=12)

    all_cand.append(rotf_cand.tolist())
    count+=1

all_cand = np.array(all_cand)

if verbose:
    #this used to be ad hoc, i.e. if lambda are not both 1.
    lambda1=1
    lambda2=1
    sc=(lambda1,lambda1,lambda2,lambda2,lambda2)
    tmp = pairwise_distances(all_cand[0]*sc, all_cand[1]*sc)
    tmp2 = np.array([unflatten(elem, nbhd_gr**2) for elem in np.argsort(tmp.flatten())])
    closest_to_intersection_01 = tmp2[:3]
    print("\n************ RESULTS FOR VARIETIES 0,1\n")
    print("true fvec:                  \n", rotf_real, "\n\n")
    w=[[print(all_cand[0, cti[0],:]), print(all_cand[1, cti[1],:]), print(tmp[cti[0], cti[1]]), print("")] for cti in closest_to_intersection_01]

if plotson and not allplotsoff:
    fig8 = plt.figure()
    axL9 = fig8.add_subplot("110", projection='3d')
    axL9.scatter3D(rotf_real[2],rotf_real[3],rotf_real[4],c='k',s=20)
    axL9.scatter3D(all_cand[0,:,2],all_cand[0,:,3],all_cand[0,:,4],s=2)
    axL9.scatter3D(all_cand[1,:,2],all_cand[1,:,3],all_cand[1,:,4],s=2)
    #axL9.scatter3D(all_cand[2,:,2],all_cand[2,:,3],all_cand[2,:,4],s=2)
    #axL9.scatter3D(all_cand[3,:,2],all_cand[3,:,3],all_cand[3,:,4],s=2)
    plt.legend(["true",0,1])#,2,3])
    axL9.set_xlabel("fxx")
    axL9.set_ylabel("fxy")
    axL9.set_zlabel("fyy")


#########################################
#########################################
#########################################
#########################################


print("\n\n")


if not allplotsoff:

    if finalpaper:
        fig8 = plt.figure()
        axL9 = fig8.add_subplot("110", projection='3d')
        axL9.scatter3D(all_cand[3,:,2],all_cand[3,:,3],all_cand[3,:,4],s=2)
        axL9.scatter3D(all_cand[2,:,2],all_cand[2,:,3],all_cand[2,:,4],c='k',s=2)
        axL9.set_xlabel("fxx")
        axL9.set_ylabel("fxy")
        axL9.set_zlabel("fyy")
        axL9.autoscale()

    ############ OPTIM TYPE 1
    k=2 #2 or 0
    if len(lights)>1:
        tmp = pairwise_distances(all_cand[0,:,k:],all_cand[1,:,k:])
        tmp2 = np.array([unflatten(elem,nbhd_gr**2) for elem in np.argsort(tmp.flatten())])
        closest_to_intersection_01 = tmp2[:3]
        if verbose:
            print("\n************ RESULTS FOR VARIETIES 0,1\n")
            print("true fvec:                  \n", rotf_real)
            print("\n\n")
            w=[[print(all_cand[0,cti[0],:]),print(all_cand[1,cti[1],:]),print(tmp[cti[0],cti[1]]),print("")] for cti in closest_to_intersection_01]
    if len(lights)>2:
        tmp = pairwise_distances(all_cand[1,:,k:],all_cand[2,:,k:])
        tmp2 = np.array([unflatten(elem,nbhd_gr**2) for elem in np.argsort(tmp.flatten())])
        closest_to_intersection_12 = tmp2[:3]
        if verbose:
            print("\n************ RESULTS FOR VARIETIES 1,2\n")
            print("true fvec:                  \n", rotf_real)
            print("\n\n")
            w=[[print(all_cand[1,cti[0],:]),print(all_cand[2,cti[1],:]),print(tmp[cti[0],cti[1]]),print("")] for cti in closest_to_intersection_12]
    if len(lights)>3:
        tmp = pairwise_distances(all_cand[2,:,k:],all_cand[3,:,k:])
        tmp2 = np.array([unflatten(elem,nbhd_gr**2) for elem in np.argsort(tmp.flatten())])
        closest_to_intersection_23 = tmp2[:3]
        if verbose:
            print("\n************ RESULTS FOR VARIETIES 2,3\n")
            print("true fvec:                  \n", rotf_real)
            print("\n\n")
            w=[[print(all_cand[2,cti[0],:]),print(all_cand[3,cti[1],:]),print(tmp[cti[0],cti[1]]),print("")] for cti in closest_to_intersection_23]
    print("******\n\n")

    if len(lights)>3:
        all_0123 =np.array(
            [all_cand[0,closest_to_intersection_01[0],:],
            all_cand[1,closest_to_intersection_01[1],:],
            all_cand[1,closest_to_intersection_12[0],:],
            all_cand[2,closest_to_intersection_12[1],:],
            all_cand[2,closest_to_intersection_23[0],:],
            all_cand[3,closest_to_intersection_23[1],:]])
        all_0123 = np.mean(np.concatenate(all_0123),axis=0)
        print("avg over all clo-to-int points: \n", all_0123,"\n\n")


    ############# OPTIM TYPE 2. Let's compare sheets together
    combtype = "mult"  # "mult" #or "sum"

    ssd = np.zeros(nbhd_gr**2) if combtype=="sum" else  np.zeros(nbhd_gr**2)+1
    ssp = np.zeros(nbhd_gr**2) if combtype=="sum" else  np.zeros(nbhd_gr**2)+1
    count = 0
    if plotson and not allplotsoff:
        if finalpaper:
            fig4 = plt.figure(figsize=(15,3))
            grid4 = plt.GridSpec(1,math.factorial(len(all_cand)-1),wspace=0.1,hspace=0.1)

            fig10 = plt.figure(figsize=(9,3))
            grid10 = plt.GridSpec(1,len(all_cand),wspace=0.1,hspace=0.1)
            ax10 = fig10.add_subplot(grid10[0,0])
            ax10.scatter(all_cand[0,:,0], all_cand[0,:,1], c=0.5+np.zeros((nbhd_gr**2)))
            ax10.scatter(rotf_real[0], rotf_real[1], c='m')

        fig6 = plt.figure(figsize=(7,7))
        grid6 = plt.GridSpec(len(all_cand)-1,len(all_cand)-1,wspace=0.1,hspace=0.1)
    #    fig7 = plt.figure(figsize=(7,7))
    #    grid7 = plt.GridSpec(1,1,wspace=0.1,hspace=0.1)


    for shtA in np.arange(len(all_cand)):
        for shtB in np.arange(shtA):
            dists = np.array([np.linalg.norm(all_cand[shtA,i,2:] - all_cand[shtB,i,2:]) for i in np.arange(nbhd_gr**2)])

            probs = -np.log(dists)
            probs = (probs - np.amin(probs))/(np.amax(probs)-np.amin(probs))

            ssd = dists + ssd if combtype=="sum" else np.multiply(dists, ssd)
            ssp = probs + ssp if combtype=="sum" else np.multiply(probs, ssp)

            if plotson and not allplotsoff:
                c0 = np.log(ssd)

                ax40 = fig4.add_subplot(grid4[0,count])
                ax40.scatter(all_cand[shtA,:,0], all_cand[shtA,:,1], c=c0)
                ax40.scatter(rotf_real[0], rotf_real[1], c='m')

                ax60 = fig6.add_subplot(grid6[shtA-1,shtB])
                ax60.scatter(all_cand[shtA,:,0], all_cand[shtA,:,1], c=probs)
                ax60.scatter(rotf_real[0], rotf_real[1], s=3, c='m')
                fig6.suptitle('Pairwise predictions', fontsize=12)

                if finalpaper:
    #                plt.figure()
    #                ths = plt.scatter(all_cand[shtA,:,0], all_cand[shtA,:,1], c=probs)
    #                plt.scatter(rotf_real[0], rotf_real[1], s=3, c='m')
    #                cbar = plt.colorbar(ths)

                    if shtB==0 and plotson_more:
                        ax10 = fig10.add_subplot(grid10[0,shtA])
                        ax10.scatter(all_cand[shtA,:,0], all_cand[shtA,:,1], c=ssp)
                        ax10.scatter(rotf_real[0], rotf_real[1], c='m')
                        fig10.suptitle('Accumulation of info with added images.', fontsize=12)

                    ax40 = fig4.add_subplot(grid4[0,count])
                    ax40.scatter(all_cand[shtA,:,0], all_cand[shtA,:,1], c=ssp)
                    ax40.scatter(rotf_real[0], rotf_real[1], c='m')
                    fig4.suptitle('Accumulation of info with added images: all pair combinations.', fontsize=12)

                count+=1
    #        ax70 = fig7.add_subplot(grid7[0], projection='3d')
    #        ax70.scatter3D(all_cand[shtA,:,0], all_cand[shtA,:,1], np.log(dists), c=np.log(dists))



    ############# OPTIM TYPE 0. inforcing A0=A1 and B0=B1

    ##find the minimum index of the summed dists
    bestabinds = np.argmin(ssd)
    bestfs = all_cand[:,bestabinds]
    print("each sheet's best f:\n" , bestfs)
    #print("INTERSECTIONS ON THE FOUR SHEETS: \n\n\n",bestfs)
    avgbestfs = np.mean(bestfs,axis=0).reshape(1,-1)[0]
    print("4 SHEETS' AVERAGE F: \n",avgbestfs,"\n")
    print("CONFIDENCE:      ",np.round(1/np.sum(np.var(bestfs,axis=0).reshape(1,-1)),decimals=4),"\n")
    print("THIS SHOULD BE CLOSE TO TRUE F: \n",rotf_real,"\n\n\n")
    if plotson and not allplotsoff:
        ax40.scatter(avgbestfs[0], avgbestfs[1], c='k')
        ax60.scatter(avgbestfs[0], avgbestfs[1], c='k')

    if plotson_col:
        ###########COLORED SHEETS
        allmin = np.amin(all_cand)
        allmax = np.amax(all_cand)
        normcand = np.reshape((all_cand - allmin)/(allmax-allmin),(len(lights),nbhd_gr,nbhd_gr,5))
        count = 0
        fig2 = plt.figure(figsize=(7, 7))
        grid  = plt.GridSpec(2,2,wspace=0.1,hspace=0.1)
        for normcandi in normcand:
            pm = 0 if count<2    else 1
            pn = 0 if count%2==1 else 1
            ax2=fig2.add_subplot(grid[int(pm),int(pn)])
            ax2.imshow(gammaencode(normcandi[:,:,2:]))
            count+=1

    plt.show()



