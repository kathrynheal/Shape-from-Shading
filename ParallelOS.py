# A simple Tensorflow 2 layer dense network example
# https://towardsdatascience.com/custom-tensorflow-loss-functions-for-advanced-machine-learning-f13cdd1d188a

#rsync -urP NN/OneStepNN.py heal@odyssey.rc.fas.harvard.edu:/n/home08/heal/Dropbox/Research/Experiments/NN
#python -W ignore NN/OneStepNN.py

############################################
############################################
#MODULE LOADING

KHsockets = ['dhcp-10-250-168-155.harvard.edu','Kathryns-MacBook-Pro-2.local']
from HelperFunctions import *
import matplotlib, ast, sys, time, os, socket, warnings
if socket.gethostname() not in KHsockets: #if not KH's MBP
    matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
if socket.gethostname() in KHsockets: #if KH's MBP
    from tensorflow.python.util import deprecation
    deprecation._PRINT_DEPRECATION_WARNINGS = False
    try:
        from tensorflow.python.util import module_wrapper as deprecation2
    except ImportError:
        from tensorflow.python.util import deprecation_wrapper as deprecation2
    deprecation2._PER_MODULE_WARNING_LIMIT = 0
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.pyplot import *
from mpl_toolkits.mplot3d import *
np.set_printoptions(suppress=True)
from time import time
from numpy.matlib import repmat

print("finished loading packages")

tf.reset_default_graph()

#END MODULE LOADING
############################################
############################################
#HYPERPARAMETER DEFINITION

##NETWORK PARAMS
numIters = 100000    #np.floor(1e2).astype(int)
stepsize = .001 #.001
#depth_h = 0        ##don't change this! there are a bunch of dependencies below
width_h = 15#50         #truly this needs to be about 500
width_g = 25#25
depth_g = 2
batch_size = 100   #int(len(X)/10)
runtype="ONESTEP"
iterprintinterval = max(1,np.ceil(numIters/100))
animate_on  = True
plots_on    = True
log_on      = True
Adamb1 = .99       #for TF optimizer. default term 0.99

##DATA PARAMS
toyopt = 0
subset = True
size0 = 10000
ttfrac = 0.98
codeis="_TOY"
uniquetime = "Synth/"#"036051_small" #"036051_large"   #"367616"#"998128"
prefix = "";            #"Dropbox/Research/Experiments/" # change this depending on whether you're running from command line, or from within the mathematica gui

uniquenum = str(time())
print("\nRun #: ",uniquenum,"\n")

##SAVE & SHOW PARAMS
params = [uniquetime, numIters, stepsize, batch_size, "(fixed) one hidden layer", width_h, depth_g, width_g, ttfrac, toyopt, subset, size0, Adamb1];

PrintParams(uniquenum,params) ##KH FUNCTION

directory = str(os.getcwd()+"/Outputs/ONESTEP/"+uniquenum+"/")
if not os.path.exists(directory):
    os.makedirs(directory)
    print("made new subdirectory for outputs: " + directory)
f = WriteTrainingOutputs(uniquenum,params) ##KH FUNCTION
print("\nParameters written to: ",f ,"\n")

starttime = time()

#END HYPERPARAMETER DEFINITION
############################################
############################################
#DATA LOADING

I0,X0,Y0 = loaddata(toyopt,size0,subset,uniquetime,0) ##KH FUNCTION
F0 = np.concatenate([X0,Y0],axis=1)
print("finished loading data")
print(I0.shape,F0.shape)
#[print("satisfies KZs 0:       ", np.round(evalKZs(I0[t],F0[t]),15)) for t in range(30)]


##optional: could trim F0,I0 to remove too-large c,d,e points and ease training
prunecde = np.arange(len(I0))[np.amax(np.abs(F0[:,2:]),axis=1)<10]
I0 = I0[prunecde]
F0 = F0[prunecde]

##for some reason, some of these don't solve the KZs. Remove them.
badinds = np.asarray(np.where([np.max(np.abs(evalKZs(I0[t], F0[t])))>0.005 for t in range(len(I0))]))

I0 = I0[~badinds][0]
F0 = F0[~badinds][0]
#KZeval=np.asarray([ np.round(evalKZs(I0[t], F0[t]),15) for t in range(len(I0))])


#END DATA LOADING
############################################
############################################
#COMPRESSION OF I-VECTORS

#group into distinct I's
uniqueIs = np.unique([tuple(row) for row in I0],axis=0)
indsuqIs = np.array([np.where((I0==elem).all(axis=1)) for elem in uniqueIs])
justuniqueinds = [k[0][0] for k in indsuqIs]
print("finished grouping data")
#print([np.round(evalKZs(I0[t],F0[t]),15) for t in justuniqueinds])

#Compression of I vectors -- new as of January 2020. Being smarter about computations.
print(I0.shape,F0.shape," I,F shapes")
IFT0 = np.stack([exploit_symmetries(I0[t],F0[t],[],False) for t in justuniqueinds])
Iout = np.stack(IFT0[:,0])
Fout = np.stack(IFT0[:,1])
Ic = np.stack(IFT0[:,0]) ##not being used yet. to test reconstruction make sure (I,T) --> I0.
T0 = np.stack(IFT0[:,2]) ##not being used yet. to test reconstruction make sure (I,T) --> I0.
I = np.concatenate(np.array([repmat(Ic[t],len(indsuqIs[t][0]),1) for t in range(len(T0))]))
I = I[:,3:5]
T = np.concatenate(np.array([repmat(T0[t],len(indsuqIs[t][0]),1) for t in range(len(T0))]))
newordering = np.concatenate(np.concatenate(indsuqIs))
F = np.array([exploit_symmetries(I0[newordering[t]], F0[newordering[t]], T[t], False)[1] for t in range(len(T))])
X,Y = F[:,:2],F[:,2:]
print("finished compressing data")


#fig = plt.figure()
#plt.scatter(I[:,0],I[:,1])
#plt.show()


#END COMPRESSION OF I-VECTORS
############################################
############################################
#DATA PROCESSING FOR NETWORK

#def normalize(A):
#    return (A-np.mean(A,axis=0))/np.var(A,axis=0)
#X,I = normalize(X), normalize(I)

print("I len",len(I))
prunecde = np.arange(len(I))[np.amax(np.abs(Y),axis=1)<10]
I = I[prunecde]
X = X[prunecde]
Y = Y[prunecde]
print("prunecde len",len(prunecde))

scaletodiscriminate=1
Y = Y*scaletodiscriminate

X = np.round(X,decimals=15)
Y = np.round(Y,decimals=15)
I = np.round(I,decimals=15)

X,Y,I, X_,Y_,I_, X_t,Y_t,I_t, idx0,train_stop,test_stop = splittraintest(X,Y,I,ttfrac,toyopt,subset) ##KH FUNCTION

print(Y_.shape,Y_t.shape,Y.shape)

nhX = X.shape[1]
nhY = Y.shape[1]
ngI = I.shape[1]
ngT = width_h*nhX+width_h*nhY+width_h+nhY #+ depth_h*width_h^2

print("*****")
print("X sz: ", X.shape)
print("Y sz: ", Y.shape)
print("I sz: ", I.shape)
print("T sz: ", (X.shape[0],ngT))

datatime = time()
print("\nData preprocessing time: ", datatime - starttime)

#END DATA PREPROCESSING
############################################
############################################
#SETUP TENSORFLOW NETWORK

nvals = [ngI,ngT,nhX,nhY]

# these placeholders serve as our input tensors
i = tf.placeholder(tf.float32, [None, ngI], name='inp_g')
x = tf.placeholder(tf.float32, [None, nhX], name='inp_h')
y = tf.placeholder(tf.float32, [None, nhY], name='out_h')

htime, yo, to, t1, W1, b1,Wo,bo,to = network(params,nvals,datatime,i,x,y,[],[])  ##KH FUNCTION

loss = tf.losses.mean_squared_error(labels = y, predictions = yo)
train_step = tf.train.AdamOptimizer(stepsize,Adamb1).minimize(loss)

############################################
############################################

if animate_on:
    fig = plt.figure(figsize=plt.figaspect(1))
    ax = fig.add_subplot(1, 1, 1, projection='3d')

training_epochs = numIters
batches = int(np.ceil(len(X_) / batch_size))
#allloss = np.arange(training_epochs).astype('float')
yo_all=[]

if log_on:
    progressfile =str("Progress"+uniquenum+".txt")
    f = open(progressfile,"w+")
    f.write("\n\nProgress for run ''"+uniquenum+"'', out of "+str(training_epochs)+":\n")
    f.flush()

# ***NOTE tf.global_variables_initializer must be called before creating a tf.Session()!***
init_op = tf.global_variables_initializer()

saver = tf.train.Saver()

############################################
############################################
##BEGIN SESSION

# create a session for training and feedforward (prediction). Sessions are TF's way to run
# feed data to placeholders and variables, obtain outputs and update neural net parameters
with tf.Session() as sess:

    sess.run(init_op)

    ##STOCHASTIC GRADIENT DESCENT
    epoch=0
    losses=0
    allloss=[]
    tol = 1
    toleranceflag = True
    stopidx = min(batch_size,X_.shape[0])
    while epoch < training_epochs and toleranceflag:
        losses = 0
        for j in range(batches):
            rng = np.arange(len(X_))
            np.random.shuffle(rng)
            idx = rng[:stopidx]

            X_b,Y_b,I_b = X_[idx],Y_[idx],I_[idx]
            _, loss_, yo_,to_,t1_,y_ ,w_= sess.run([train_step, loss, yo, to,t1,y,Wo], feed_dict={x: X_, y: Y_, i:I_})
            losses = losses + np.sum(loss_)
            isdiverging(yo_) #check that you're not getting NaNs

        epcloss = losses if batches==0 else losses/batches
        allloss.append(epcloss)

        if epoch%iterprintinterval==0:
            yo_all.append(yo_)
            strout = str("Epoch %.8d (avg train over %.2d batches)\n     train loss: %.8f\n" % (epoch,batches,epcloss))
            print(strout)
            print("mine\n", yo_[:4])
            print("true\n", y_[:4])
            if log_on:
                f = open(progressfile,"a+")
                f.write(strout+"\n")
                f.flush()


                yt_ = sess.run(yo, feed_dict={x:X_t,  y:Y_t,  i:I_t})
                directory = str(os.getcwd()+"/Outputs/ONESTEP/"+uniquenum+"/iter"+str(epoch))
                if not os.path.exists(directory):
                    os.makedirs(directory)
                fn = saver.save(sess, directory+"/model")
            fw = tf.summary.FileWriter('./posgraphs') #tensorboard --logdir="./posgraphs" --port 6006
            fw.add_graph(sess.graph)
#        if epoch == 2:
#            print("\n\nweights at epoch 2:\n",w_)
#        if epoch == training_epochs-1:
#            print("\n\nweights at epoch n-3:\n",w_)

        epoch = epoch + 1

#    train_writer = tf.summary.FileWriter(TBOARD_LOGS_DIR) train_writer.add_graph(sess.graph)

    print("\n\n ********* Training is complete!")

    ##SAVE OUTPUTS
    if animate_on:
        yo_all ,Y_ = np.real(yo_all), np.real(Y_)
        ani = animation.FuncAnimation(fig, animate, frames=len(yo_all), fargs=([ax,Y_,yo_all],), interval=400, repeat=True)
        if subset:
            plt.show()
        print(" ********* Training video created successfully.")
    if log_on:
        f = open(progressfile,"a+")
        f.write("DONE.\n")
        f.flush()
        f.close()
        tvars_vals = sess.run(tf.trainable_variables())

    ##TEST AND REPORT
    print(X_.shape,Y_.shape,I_.shape)
    loss_, yo_ = sess.run([loss, yo], feed_dict={x:X_,  y:Y_,  i:I_})
    _, yt_     = sess.run([loss, yo], feed_dict={x:X_t, y:Y_t, i:I_t})

    yo_ = np.round(yo_,decimals=15)/scaletodiscriminate
    yt_ = np.round(yt_,decimals=15)/scaletodiscriminate

Y_t = np.real(Y_t)

##END SESSION
############################################
############################################


if log_on:
    directory = str(os.getcwd()+"/Outputs/ONESTEP/"+uniquenum)
    np.savetxt(directory+"/Costs.csv", allloss, delimiter=',',newline='\n')
    np.savetxt(directory+"/ytest.csv",yt_, delimiter=',',newline='\n')
    np.save(   directory+"/Wghts_" +uniquenum,       tvars_vals)
    if not subset:
        np.savetxt(directory+"/indtr.csv", idx0[:train_stop],          delimiter=',',newline='\n')
        np.savetxt(directory+"/indte.csv", idx0[len(idx0)-test_stop:], delimiter=',',newline='\n')

if animate_on:
    anifile = directory+"/training_"+uniquenum+".mp4";
    writer = animation.FFMpegWriter(codec="mpeg4")
    ani.save(anifile,writer=writer)
    print(" ********* Training video saved successfully.")

trainAndAnimatetime = time()
print("Training and Animation time: ", trainAndAnimatetime - htime)


if log_on:
    fig2 = plt.figure(figsize=plt.figaspect(1))
    ax = fig2.add_subplot(1, 1, 1, projection='3d')
    sc1 = ax.scatter3D(yt_[:,0],yt_[:,1],yt_[:,2])
    sc4 = ax.scatter3D(Y_t[:,0],Y_t[:,1],Y_t[:,2])
    ax.set_xlabel('f_{xx}')
    ax.set_ylabel('f_{xy}')
    ax.set_zlabel('f_{yy}')
    ax.legend([sc1, sc4], ['Testing Output from NN', 'Ground Truth'], numpoints = 1)
    ax.set_title('NN Results');
    plt.savefig('Outputs/ONESTEP/'+uniquenum+'/TESTplot.png', bbox_inches='tight')
    plottime = time()
    print("Training and Animation time: ", plottime - trainAndAnimatetime)


    ###
    uniqueIts = np.unique([tuple(row) for row in I_t],axis=0)
    indsuqIs  = [np.where((I_t==elem).all(axis=1)) for elem in uniqueIts]

    directory = str(os.getcwd()+'/Outputs/ONESTEP/'+uniquenum+'/IPlots')
    if not os.path.exists(directory):
        os.makedirs(directory)
        print("made new directory for plots: "+directory)

fig5 = plt.figure(figsize=plt.figaspect(1))
ax5 = fig5.add_subplot(1, 1, 1)
sc15 = ax5.plot(np.log(allloss))
plt.xlabel("Iteration")
plt.title("Log of Costs")
saveto = str(directory+'/Costs.png')
plt.savefig(saveto, bbox_inches='tight')
plt.show()

#if plots_on:
#    for i in range(max(10,len(uniqueIts))):
#        fig4 = plt.figure(figsize=plt.figaspect(1))
#        ax4 = fig4.add_subplot(1, 1, 1, projection='3d')
#        sc1 = ax4.scatter3D(yt_[indsuqIs[i],0],yt_[indsuqIs[i],1],yt_[indsuqIs[i],2])
#        sc2 = ax4.scatter3D(Y_t[indsuqIs[i],0],Y_t[indsuqIs[i],1],Y_t[indsuqIs[i],2])
#        ax4.set_xlabel('f_{xx}')
#        ax4.set_ylabel('f_{xy}')
#        ax4.set_zlabel('f_{yy}')
#        ax4.set_title('test plot for an individual I')
#        saveto = str(directory+'/p'+str(i)+'.png')
#        plt.savefig(saveto, bbox_inches='tight')
#        if subset:
#            plt.show()



print("")
###
