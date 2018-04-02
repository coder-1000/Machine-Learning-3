import numpy as np
import tensorflow as tf
import weightedmatrix as wm
import matplotlib.pyplot as plt

NUM_HIDDEN_UNITS = 1000;
NUM_UNITS_OUTPUT_LAYER = 10;
BATCH_SIZE = 500;
NUM_ITERATIONS = 5000;
NUM_PIXELS = 784
LAMDA = 0.0003

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
    
    #hidden layer1
    [w1, b1, s1] = wm.weighted_matrix(x0, NUM_HIDDEN_UNITS);
    x1 = tf.nn.relu(s1);
    
    #output layer
    [w2, b2, s2] = wm.weighted_matrix(x1, NUM_UNITS_OUTPUT_LAYER);
    
    #error
    cross = tf.nn.softmax_cross_entropy_with_logits(logits = s2, labels = faty);
    weightDecay = (tf.reduce_sum(w1**2) + tf.reduce_sum(w2**2) )*(LAMDA)
    
    loss = tf.reduce_mean((cross + weightDecay)/2.0)

    misclassification = tf.reduce_mean( tf.cast(tf.not_equal(tf.argmax(s2,axis=1), y), tf.float64) )
    #misclassification = tf.argmax(s2,axis=1);

    learningRate = 0.001
    
    descendGradient = tf.train.AdamOptimizer(learningRate).minimize(loss);
    
    initializer = tf.global_variables_initializer();
   
    #store cross entropy error
    crossTrainVals = []
    crossValidVals = []
    crossTestVals = []
    
    #store minimum value to detect early stopping
    minCrossTrain = 9999
    minCrossValid = 9999
    minCrossTest = 9999

    #store classifcation error
    classTrainVals = []
    classValidVals = []
    classTestVals = [] 
    
    #store min value for early stopping
    minClassTrain = 9999
    minClassValid = 9999
    minClassTest = 9999

    with tf.Session() as sess:
        
        sess.run(initializer)
        start = 0;
       
        num_batches = trainX.shape[0] // BATCH_SIZE
        
        for i in range(NUM_ITERATIONS):
            
            end = start + BATCH_SIZE;
            sess.run(descendGradient, feed_dict={x0:  trainX[start:end], y: trainY[start: end] })
            if( ((i+1)% num_batches) == 0):
                print(i)

                #get cross entropy loss values
                trainCross = sess.run(loss, feed_dict={x0:  trainX, y: trainY })
                validCross = sess.run(loss, feed_dict={x0: validX , y: validY })
                testCross = sess.run(loss, feed_dict={x0: testX , y: testY })              
                
                minCrossTrain = min(minCrossTrain, trainCross)
                minCrossValid = min(minCrossValid, validCross)
                minCrossTest = min(minCrossTest, testCross)

                crossTrainVals.append(trainCross)
                crossValidVals.append(validCross)
                crossTestVals.append(testCross)
                
                #get classification error values
                trainClass = sess.run(misclassification, feed_dict ={x0: trainX, y:  trainY})
                validClass = sess.run(misclassification, feed_dict ={x0: validX, y:  validY})
                testClass = sess.run(misclassification, feed_dict ={x0: testX, y:  testY})

                minClassTrain = min(minClassTrain, trainClass)
                minClassValid = min(minClassValid, validClass)
                minClassTest = min(minClassTest, testClass)

                classTrainVals.append(trainClass)
                classValidVals.append(validClass)
                classTestVals.append(testClass)
            
            #increment batch
            start = end % numTrainingPoints
        
        #get list of numbers from 0 to num iterations
        xVals = np.arange(len(crossTrainVals));

        print("minimum cross Train was " + str(minCrossTrain))
        print("minimum cross Valid  was " + str(minCrossValid))
        print("minimum cross Test was " + str(minCrossTest))

        print("minimum class Train was " + str(minClassTrain))
        print("minimum class Train was " + str(minClassTrain))
        print("minimum class Train was " + str(minClassTrain))





        plt.plot(xVals, crossTrainVals, label= "training pts" )
        plt.plot(xVals, crossValidVals, label= "validation pts" )
        plt.plot(xVals, crossTestVals, label="test points")

        plt.xlabel('Epoch #');
        plt.ylabel('Loss');
        plt.legend();
        plt.title("Cross Entropy Loss vs Num Epochs")

        plt.show() 
    
        plt.figure()

        plt.plot(xVals, classTrainVals, label="training pts" )
        plt.plot(xVals, classValidVals, label="validation pts" )
        plt.plot(xVals, classTestVals, label="test points")

        plt.xlabel('Epoch #');
        plt.ylabel('Classification Error');
        plt.legend();
        plt.title("Classification Error vs Num Epochs")
        
        plt.show()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    #misclassifications = []
    #learning_rate = [0.001, 0.005, 0.0001]
        

    #for lR in learning_rate:
        # we now need a loss function of some kind
        #xLayer0 = inputs; #our initial inputs
        #the weighted sum matrix should return a 1000x1 vector 
        #sLayer1 = weighted_matrix(inputs, NUM_HIDDEN_UNITS) #our hidden layer
        #our x output layer should also be 1000x1
        #xLayer1 = tf.nn.relu(sLayer1); #technically our output layer
        #this should output a 1x10 vector with the 10 classes
        #outputLayer = weighted_matrix(xLayer1, NUM_HIDDEN_UNITS_OUTPUT_LAYER);
        
        #add some weight decay to prevent overfitting
        #

    
    #misclassifications = []
    #learning_rate = [0.001, 0.005, 0.0001]
        

    #for lR in learning_rate:
        # we now need a loss function of some kind
        #xLayer0 = inputs; #our initial inputs
        #the weighted sum matrix should return a 1000x1 vector 
        #sLayer1 = weighted_matrix(inputs, NUM_HIDDEN_UNITS) #our hidden layer
        #our x output layer should also be 1000x1
        #xLayer1 = tf.nn.relu(sLayer1); #technically our output layer
        #this should output a 1x10 vector with the 10 classes
        #outputLayer = weighted_matrix(xLayer1, NUM_HIDDEN_UNITS_OUTPUT_LAYER);
        
        #add some weight decay to prevent overfitting
        #loss = tf.nn.softmax_cross_entropy_with_logits(logits = outputLayer, label = actualLabel);
	#trainStep = tf.train.GradientOptimizer(lR).minimize(loss);

	#misclassfications, add to the misclassifications list if they are not equal, therfore misclassified
	#missclassifications.append(not tf.equal(tf.transpose(outputLayer), actualLabel))
