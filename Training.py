




############################################
############################################
#MODULE LOADING

import matplotlib, sys, os, socket, warnings
import tensorflow as tf
import numpy as np

sys.path.insert(1, os.path.join(sys.path[0], 'Utilities'))
sys.path.insert(1, os.path.join(sys.path[0], 'Data Generation'))
from Utilities_General import *
from Utilities_Model import *
from Utilities_TDG import *

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
np.set_printoptions(suppress=True)

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
    
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.pyplot import *
from mpl_toolkits.mplot3d import *
from numpy.matlib import repmat

print("finished loading packages")

tf.reset_default_graph()

#END MODULE LOADING
############################################
############################################
#HYPERPARAMETER DEFINITION

from General_Parameters import *
from Training_Parameters import *

uniquenum = str(time.time())
print("\nRun #: ",uniquenum,"\n")

##SAVE & SHOW PARAMS
params = [uniquetime, numIters, stepsize, batch_size, "(fixed) one hidden layer", width_h, depth_g, width_g, ttfrac, toyopt, subset, size0, Adamb1];
PrintParams(uniquenum,params)

directory = str(os.getcwd()+"/TrainingOutput/"+uniquenum+"/")
if not os.path.exists(directory):
    os.makedirs(directory)
    print("made new subdirectory for outputs: " + directory)
f = WriteTrainingOutputs(uniquenum,params)
print("\nParameters written to: ",f ,"\n")

#END HYPERPARAMETER DEFINITION
############################################
############################################
#DATA LOADING/COMPRESSING/PREPROCESSING

dataout,dataout_,dataout_t,etcout = DataLoadingMain(time.time(),ttfrac,toyopt,size0,subset,uniquetime)
X,   Y,   I   = dataout
X_,  Y_,  I_  = dataout_
X_t, Y_t, I_t = dataout_t
idx0,train_stop,test_stop,datatime,scaletodiscriminate,info = etcout
T = info[:,1]
ngI,nhX,nhY = I.shape[1],X.shape[1],Y.shape[1]
ngT = width_h*nhX+width_h*nhY+width_h+nhY

#fig = plt.figure()
#plt.scatter(I[:,0],I[:,1])
#plt.show()

#END DATA LOADING/COMPRESSING/PREPROCESSING
############################################
############################################
#SETUP TENSORFLOW ARCHITECTURE


# these placeholders serve as our input tensors
i = tf.placeholder(tf.float32, [None, ngI], name='inp_g')
x = tf.placeholder(tf.float32, [None, nhX], name='inp_h')
y = tf.placeholder(tf.float32, [None, nhY], name='out_h')

htime, yo, to, t1, W1, b1,Wo,bo,to = network(params,[ngI,ngT,nhX,nhY],datatime,i,x,y,[],[])  ##KH FUNCTION

loss = tf.losses.mean_squared_error(labels = y, predictions = yo)
train_step = tf.train.AdamOptimizer(stepsize,Adamb1).minimize(loss)

#END SETUP TENSORFLOW ARCHITECTURE
#############################################
#############################################

if animate_on:
    fig = plt.figure(figsize=plt.figaspect(1))
    ax = fig.add_subplot(1, 1, 1, projection='3d')

training_epochs = numIters
batches = int(np.ceil(len(X_) / batch_size))
yo_all=[]

if log_on:
    progressfile =str("Progress"+uniquenum+".txt")
    f = open(progressfile,"w+")
    f.write("\n\nProgress for run ''"+uniquenum+"'', out of "+str(training_epochs)+":\n")
    f.flush()

if ws_on:
    tf.compat.v1.train.warm_start("TrainingOutput/"+ws_dir)
init_op = tf.global_variables_initializer()

saver = tf.train.Saver(max_to_keep=5)

#############################################
#############################################
##BEGIN SESSION

# create a session for training and feedforward (prediction). Sessions are TF's way to run
# feed data to placeholders and variables, obtain outputs and update neural net parameters
with tf.Session() as sess:

    sess.run(init_op)

    ##(STOCHASTIC) GRADIENT DESCENT
    epoch, losses, allloss, tol, toleranceflag = 0, 0, [], 1, True
    stopidx = min(batch_size,X_.shape[0])
    while epoch < training_epochs and toleranceflag:
        losses = 0
        for j in range(batches):
            rng = np.arange(len(X_))
            np.random.shuffle(rng)  # shuffles in place
            idx = rng[:stopidx]

            X_b,Y_b,I_b = X_[idx],Y_[idx],I_[idx]
            _, loss_, yo_,to_,t1_,y_ ,w_= sess.run([train_step, loss, yo, to,t1,y,Wo], feed_dict={x: X_, y: Y_, i:I_})
            losses = losses + np.sum(loss_)
            isdiverging(yo_)  # check that you're not getting NaNs. kill if so.

        epcloss = losses if batches==0 else losses/batches
        allloss.append(epcloss)

        if epoch%iterprintinterval==0:
            yo_all.append(yo_)
            strout = str("Epoch %.8d (avg train over %.2d batches)\n     train loss: %.8f\n" % (epoch,batches,epcloss))
            print(strout)
            print("mine\n", yo_[:4]/scaletodiscriminate)
            print("true\n", y_[:4]/scaletodiscriminate)
            if log_on:
                f = open(progressfile,"a+")
                f.write(strout+"\n")
                f.flush()

            yt_ = sess.run(yo, feed_dict={x:X_t,  y:Y_t,  i:I_t})
            directory = str(os.getcwd()+"/TrainingOutput/"+uniquenum)
            if not os.path.exists(directory):
                os.makedirs(directory)
            fn = saver.save(sess, "TrainingOutput/"+uniquenum+"/model_iter", global_step=epoch)
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
    print("train mine\n", np.real(yo_[:12]))
    print("train true\n", np.real( Y_[:12]))

    yo_ = np.round(yo_,decimals=15)#/scaletodiscriminate
    yt_ = np.round(yt_,decimals=15)#/scaletodiscriminate


Y_t = np.real(Y_t)

##END SESSION
############################################
############################################

#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#

############################################
############################################
##RESULTS.....
##notice that we are NOT "unprocessing" the I vectors... we want them to stay in the 2-form for evaluation.

directory = str(os.getcwd()+"/TrainingOutput/"+uniquenum)
np.savetxt(directory+"/Costs.csv", allloss, delimiter=',',newline='\n')
np.savetxt(directory+"/ytest.csv",yt_, delimiter=',',newline='\n')
#np.save(   directory+"/Wghts_" +uniquenum,       tvars_vals)
if not subset:
    np.savetxt(directory+"/indtr.csv", idx0[:train_stop],          delimiter=',',newline='\n')
    np.savetxt(directory+"/indte.csv", idx0[len(idx0)-test_stop:], delimiter=',',newline='\n')

if animate_on:
    anifile = directory+"/training_"+uniquenum+".mp4";
    writer = animation.FFMpegWriter(codec="mpeg4")
    ani.save(anifile,writer=writer)
    print(" ********* Training video saved successfully.")

trainAndAnimatetime = time.time()
print("Training and Animation time: ", trainAndAnimatetime - htime)

if toyopt == 0:
#    fig2 = plt.figure(figsize=plt.figaspect(1))
#    ax = fig2.add_subplot(1, 1, 1, projection='3d')
#    sc1 = ax.scatter3D(yt_[:,0],yt_[:,1],yt_[:,2])
#    sc4 = ax.scatter3D(Y_t[:,0],Y_t[:,1],Y_t[:,2])
#    ax.set_xlabel('f_{xx}')
#    ax.set_ylabel('f_{xy}')
#    ax.set_zlabel('f_{yy}')
#    ax.legend([sc1, sc4], ['Testing Output from NN', 'Ground Truth'], numpoints = 1)
#    ax.set_title('NN Results');
#    plt.savefig(directory+"/testplot.png", bbox_inches='tight')
    plottime = time.time()
    print("Training and Animation time: ", plottime - trainAndAnimatetime)


    ###
    uniqueIts = np.unique([tuple(row) for row in I_t],axis=0)
    indsuqIs  = [np.where((I_t==elem).all(axis=1)) for elem in uniqueIts]

    directory2 = str(os.getcwd()+"/TrainingOutput/"+uniquenum+"/IPlots")
    if not os.path.exists(directory2):
        os.makedirs(directory2)
        print("made new directory for plots: "+ directory2)

fig5 = plt.figure(figsize=plt.figaspect(1))
ax5 = fig5.add_subplot(1, 1, 1)
sc15 = ax5.plot(np.log(allloss))
plt.xlabel("Iteration")
plt.title("Log of Costs")
saveto = str(directory+"/Costs.png")
plt.savefig(saveto, bbox_inches='tight')

plt.show()

#scale'em back down
yt_ /= scaletodiscriminate
Y_t /= scaletodiscriminate

if plots_on:
    print("\n\n\n")
    for i in range(min(10,len(uniqueIts))):
        print("test mine: \n",yt_[indsuqIs[i]])
        print("test true: \n",Y_t[indsuqIs[i]])
        print("\n")
        fig4 = plt.figure(figsize=plt.figaspect(1))
        ax4 = fig4.add_subplot(1, 1, 1, projection='3d')
        sc1 = ax4.scatter3D(yt_[indsuqIs[i],0],yt_[indsuqIs[i],1],yt_[indsuqIs[i],2])
        sc2 = ax4.scatter3D(Y_t[indsuqIs[i],0],Y_t[indsuqIs[i],1],Y_t[indsuqIs[i],2])
        ax4.set_xlabel('f_{xx}')
        ax4.set_ylabel('f_{xy}')
        ax4.set_zlabel('f_{yy}')
        ax4.set_title('test plot for an individual I')
#        saveto = str(directory+'/p'+str(i)+'.png')
#        plt.savefig(saveto, bbox_inches='tight')
    if subset:
        plt.show()


print("")
###
