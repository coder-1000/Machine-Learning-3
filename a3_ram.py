import numpy as np
import tensorflow as tf
import weightedmatrix as wm
import matplotlib.pyplot as plt
import time
from math import exp

#NUM_HIDDEN_UNITS = 1000;
NUM_UNITS_OUTPUT_LAYER = 10;
BATCH_SIZE = 500;
NUM_ITERATIONS = 5000;
NUM_PIXELS = 784
#LAMDA = 0.0003
#LEARNING_RATES = [0.005, 0.001, 0.0001]

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


    # variable initialization
    #store cross entropy error
    crossTrainVals = [[],[],[]]
    crossValidVals = [[],[],[]] 
    crossTestVals = [[],[],[]]
        
    #store minimum value to detect early stopping
    minCrossTrain = [9999, 9999, 9999]
    minCrossValid = [9999, 9999, 9999]
    minCrossTest = [9999, 9999, 9999]

    #store classifcation error
    classTrainVals = [[],[],[]]
    classValidVals = [[],[],[]]
    classTestVals = [[],[],[]] 
        
    #store min value for early stopping
    minClassTrain = [9999, 9999, 9999]
    minClassValid = [9999, 9999, 9999]
    minClassTest = [9999, 9999, 9999]
        
    earlyStoppingIteration = [0, 0, 0];

    learningRate = []


    for itr in range(5):

        print("iteration number: " + str(itr))
        np.random.seed(int(time.time()))
        lr = np.arange(-7.5,-4.5,0.1)
        numLayer = np.arange(1,5,1)
        numHiddenUnits = np.arange(100,500,10).astype(np.int32)
        lamda = np.arange(-9,-6,0.1)

        np.random.shuffle(lr)
        np.random.shuffle(numLayer)
        np.random.shuffle(numHiddenUnits)
        np.random.shuffle(lamda)

        lr = exp(lr[0])
        learningRate.append(lr)
        numLayer = numLayer[0]
        numHiddenUnits = numHiddenUnits[0]
        lamda = exp(lamda[0])
        dropFlag = int(time.time())%2

        print(lr)
        print(numLayer)
        print(numHiddenUnits)
        print(lamda)
        print(dropFlag)

        #format data
        numTrainingPoints = trainData.shape[0]

        trainX = np.reshape(trainData, (trainData.shape[0], -1) );
        trainY = trainTarget.astype(np.float64);
        
        validX = np.reshape(validData, (validData.shape[0], -1) );
        validY = validTarget.astype(np.float64);    
        
        testX = np.reshape(testData, (testData.shape[0], -1) );
        testY = testTarget.astype(np.float64); 
        
        #input layer
        x0 = tf.placeholder(tf.float64, name="input_layer", shape = [None, NUM_PIXELS]);
        #target values for input
        y = tf.placeholder(tf.int64, name="target");
        #one hot enncoding for softmax
        faty = tf.one_hot(indices = y, depth = 10);
        
        # w - weight
        # b - bias
        # s - sensitivity/weighted
        x1 = []
        wd = []
        for i in range(numLayer):

            # hidden layer
            [wh, bh, sh] = wm.weighted_matrix(x0, numHiddenUnits)
            xh = tf.nn.relu(sh)
            if(dropFlag == 1):
                xh = tf.nn.dropout(xh, 0.5)

            # calculate the weight decay
            if(i == 0):
                wd = tf.reduce_sum(wh**2)
            else:
                wd = wd + tf.reduce_sum(wh**2)

            # x1 = output of all hidden layers
            if(i == (numLayer-1)):
                x1 = xh

        


        # output layer
        [w2, b2, s2] = wm.weighted_matrix(x1, NUM_UNITS_OUTPUT_LAYER);
        
        # error
        cross = tf.nn.softmax_cross_entropy_with_logits(logits = s2, labels = faty);
        weightDecay = (wd + tf.reduce_sum(w2**2) )*(lamda)
        
        loss = tf.reduce_mean((cross + weightDecay)/2.0)

        misclassification = tf.reduce_mean( tf.cast(tf.not_equal(tf.argmax(s2,axis=1), y), tf.float64) )
        #misclassification = tf.argmax(s2,axis=1);
  


        with tf.Session() as sess:
            
            start = 0;
           
            num_batches = trainX.shape[0] // BATCH_SIZE
                
            descendGradient = tf.train.AdamOptimizer(learningRate[itr]).minimize(loss);
                
            #initializer has to be done after declaring adam optimizer
            initializer = tf.global_variables_initializer();
                
            sess.run(initializer)
                
            for i in range(NUM_ITERATIONS): 
                
                end = start + BATCH_SIZE;
                sess.run(descendGradient, feed_dict={x0:  trainX[start:end], y: trainY[start: end] })
                
                if( ((i+1)% num_batches) == 0):
                    #print(i)

                        #get cross entropy loss values
                    trainCross = sess.run(loss, feed_dict={x0:  trainX, y: trainY })
                    validCross = sess.run(loss, feed_dict={x0: validX , y: validY })
                    testCross = sess.run(loss, feed_dict={x0: testX , y: testY })              
                    
                    minCrossTrain[itr]= min(minCrossTrain[itr], trainCross)
                    minCrossValid[itr] = min(minCrossValid[itr], validCross)
                    minCrossTest[itr] = min(minCrossTest[itr], testCross)

                    crossTrainVals[itr].append(trainCross)
                    crossValidVals[itr].append(validCross)
                    crossTestVals[itr].append(testCross)
                        
                        #get classification error values
                    trainClass = sess.run(misclassification, feed_dict ={x0: trainX, y:  trainY})
                    validClass = sess.run(misclassification, feed_dict ={x0: validX, y:  validY})
                    testClass = sess.run(misclassification, feed_dict ={x0: testX, y:  testY})

                    minClassTrain[itr] = min(minClassTrain[itr], trainClass)
                        
                    if(minClassValid[itr] > validClass):
                        earlyStoppingIteration[itr] = i;
                        minClassValid[itr] = validClass; 
                    minClassValid[itr] = min(minClassValid[itr], validClass)
                    minClassTest[itr] = min(minClassTest[itr], testClass)

                    classTrainVals[itr].append(trainClass)
                    classValidVals[itr].append(validClass)
                    classTestVals[itr].append(testClass)
                    
                #increment batch
                start = end % numTrainingPoints
            

        
    #for itr in range(5):
        print("iteration number: " + str(itr))

        #get list of numbers from 0 to num iterations
        xVals = np.arange(len(crossTrainVals[itr]))
        
        print("early stopping = " + str(earlyStoppingIteration[itr]))

        print("learning rate  = " + str(learningRate[itr]))

        print("minimum cross Train was " + str(minCrossTrain[itr]))
        print("minimum cross Valid  was " + str(minCrossValid[itr]))
        print("minimum cross Test was " + str(minCrossTest[itr]))

        print("minimum class Train was " + str(minClassTrain[itr]))
        print("minimum class Valid was " + str(minClassTrain[itr]))
        print("minimum class Test was " + str(minClassTrain[itr]))

        plt.plot(xVals, crossTrainVals[itr], label= "training pts" )        
        plt.plot(xVals, crossValidVals[itr], label= "validation pts" )
        plt.plot(xVals, crossTestVals[itr], label="test points")

        plt.xlabel('Epoch #');
        plt.ylabel('Loss');
        plt.legend();
        plt.title("Cross Entropy Loss vs Num Epochs with lr = " + str(np.round(learningRate[itr],decimals=6)))

        plt.show() 
        
        plt.figure()

        plt.plot(xVals, classTrainVals[itr], label="training pts" )
        plt.plot(xVals, classValidVals[itr], label="validation pts" )
        plt.plot(xVals, classTestVals[itr], label="test points")

        plt.xlabel('Epoch #');
        plt.ylabel('Classification Error');
        plt.legend();
        plt.title("Classification Error vs Num Epochs with lr = " + str(np.round(learningRate[itr],decimals=6)))
            
        plt.show()
    
   

