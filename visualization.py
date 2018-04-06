import numpy as np
import tensorflow as tf
import weightedmatrix as wm
import matplotlib.pyplot as plt

NUM_HIDDEN_UNITS = 1000;
NUM_OUTPUT_UNITS = 10;
BATCH_SIZE = 500;
NUM_PIXELS = 784
LAMDA = 0.0003
BEST_LEARNING_RATE = 0.0001
NUM_HIDDEN_UNITS = 1000
KEEP_RATE = 0.5 
EARLY_STOP_POINT = 4379

if __name__ == "__main__":

    with np.load("notMNIST.npz") as data:
        Data, Target = data ["images"], data["labels"]
        np.random.seed(521)
        randIndx = np.arange(len(Data))
        np.random.shuffle(randIndx)
        Data = Data[randIndx]/255.
        Target = Target[randIndx]
        trainData, trainTarget = Data[:15000], Target[:15000]
        validData, validTarget = Data[15000:16000], Target[15000:16000]
        testData, testTarget = Data[16000:], Target[16000:] 
    
    ###setting up lists to store error
   
    #store cross entropy error
    crossTrainVals = []
    crossValidVals = [] 
    crossTestVals = []
    
    #store minimum value to detect early stopping
    minCrossTrain = 9999
    minCrossValid = 9999
    minCrossTest = 9999

    start = 0;
        
    ###setting up tf's graph

    #reset graph to start from blank
    tf.reset_default_graph()
    
    tf.set_random_seed(1002473496)
    
    #current hidden units
    numHiddenUnits = NUM_HIDDEN_UNITS 

    ###format data
    
    numTrainingPoints = trainData.shape[0]

    trainX = np.reshape(trainData, (trainData.shape[0], -1) );
    trainY = trainTarget.astype(np.float64);
    
    num_batches = trainX.shape[0] // BATCH_SIZE
    
    validX = np.reshape(validData, (validData.shape[0], -1) );
    validY = validTarget.astype(np.float64);    
    
    testX = np.reshape(testData, (testData.shape[0], -1) );
    testY = testTarget.astype(np.float64); 
    
    quarter = 3*EARLY_STOP_POINT//4.0

    ###set up graphs to train 2 cnn. 1 with dropout and one without

    #input layer
    x0 = tf.placeholder(tf.float64, name="input_layer", shape = [None, NUM_PIXELS]);
    #target values for input
    y = tf.placeholder(tf.int64, name="target");
    
    #one hot enncoding for softmax
    faty = tf.one_hot(indices = y, depth = 10);
    
    #hidden layer1
    [w1, b1, s1] = wm.weighted_matrix(x0, numHiddenUnits);
    [w1d, b1d, s1d] = wm.weighted_matrix(x0, numHiddenUnits);
    
    x1 = tf.nn.relu(s1);
    x1Drop = tf.nn.relu(tf.nn.dropout(s1d, KEEP_RATE));

    [w2, b2, s2] = wm.weighted_matrix(x1, NUM_OUTPUT_UNITS);
    [w2d, b2d, s2d] = wm.weighted_matrix(x1Drop, NUM_OUTPUT_UNITS);
    
    yhat = s2
    yhatDrop = s2d
    
    #error
    cross = tf.nn.softmax_cross_entropy_with_logits(logits = yhat, labels = faty);
    weightDecay = (tf.reduce_sum(w1**2) + tf.reduce_sum(w2**2) )*(LAMDA)    
    
    crossDrop = tf.nn.softmax_cross_entropy_with_logits(logits = yhatDrop, labels = faty);
    weightDecayDrop = (tf.reduce_sum(w1d**2) + tf.reduce_sum(w2d**2) )*(LAMDA)    
    
    lossDrop = tf.reduce_mean((crossDrop + weightDecayDrop)/2.0)
    loss = tf.reduce_mean((cross + weightDecay)/2.0)

    learningRate = BEST_LEARNING_RATE

    descendGradient = tf.train.AdamOptimizer(learningRate).minimize(loss); 
    descendGradientDrop = tf.train.AdamOptimizer(learningRate).minimize(lossDrop); 
    
    #initializer has to be done after declaring adam optimizer
    initializer = tf.global_variables_initializer();
    
    ###gradient descent
    with tf.Session() as sess:
        sess.run(initializer)
        tf.set_random_seed(1002473496) 
        for i in range(EARLY_STOP_POINT): 
            
            
            end = start + BATCH_SIZE;
            if(i == quarter):
                weights = sess.run(w1, feed_dict={x0:  trainX[start:end], y: trainY[start: end] })
                weights = np.transpose(weights[:,0:100])
                print(weights.shape) 
                numRows = 10
                numCols = 10
                picW = 28
                picH = 28
                
                #set up a 10x10 of 28x28 image with a white background and black border
                figure, axes = plt.subplots(numRows, numCols, figsize=(picW,picH), facecolor ='w', edgecolor='k', frameon = False)
                print(axes.shape)

                axes = axes.ravel()

                for picNum in range(numRows*numCols):
                    #format image for figure
                    picture = weights[picNum,:]
                    picture = np.reshape(picture, [28,28])
                    axes[picNum].imshow(picture, cmap="gray")
                plt.show()
        
            #sess.run(descendGradientDrop, feed_dict={x0:  trainX[start:end], y: trainY[start: end] })
            sess.run(descendGradient, feed_dict={x0:  trainX[start:end], y: trainY[start: end] })
            
            #if( ((i+1)% num_batches) == 0):
                #print(i)
           
            if((i == quarter) or (i == EARLY_STOP_POINT)):
                print("early stopping point")
                print(i)

            #increment batch
            start = end % numTrainingPoints
        

    ###plotting

    
