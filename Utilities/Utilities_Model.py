
"""Called by Training.py"""


import tensorflow as tf
import numpy as np
import time


#####################################################################################
#####################################################################################


def lintran(z,A,c):
    tmp = tf.reshape(z,(tf.shape(z)[0],1,tf.shape(z)[1]))
    return tf.reshape(tf.matmul(tmp,A),tf.shape(c)) + c


#####################################################################################
#####################################################################################


def network(hyperparams,nvals,datatime,i,x,y,varsW,varsb):
    """params = [uniquetime, numIters, stepsize, batch_size, "(fixed) one hidden layer",
    width_h, depth_g, width_g, ttfrac, toyopt, subset, size0].
    
    Make sure this still matches ParallelOS.py file!"""
    
    uniquetime, numIters, stepsize, batch_size, depth_h, width_h, depth_g, width_g, ttfrac, toyopt, subset, size0, Adamb1 = hyperparams  # unpacking params
    ngI,ngT,nhX,nhY = nvals
    
    #############THIS IS THE G NETWORK###########
    if len(varsW)==0: #if creating a new graph
        print("Creating new random inits for weights and biases in HelperFunctions.")
        
        # TF Variables = NN parameter tensors. initialize them to random (gaussian) values
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
    
    else: #if restoring an existing graph, whose variable values pass into this function as params
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

    gtime = time.time()
    #print("g-network setup time: ", gtime - datatime)

    #############THIS IS THE H NETWORK###########
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

    #############SETUP OF LOSS & OPT###########
    htime = time.time()
    #print("h-network setup time: ",htime - gtime )

    return htime,yo,to,t1,W1,b1,Wo,bo,to
