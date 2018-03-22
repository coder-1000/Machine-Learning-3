##Data (18724, 28, 28)
##Target (18724,)
##TrainData (15000, 28, 28)
##TrainTarget (15000,)
##ValidData (1000, 28, 28)
##ValidTarget (1000,)
##TestData (2724, 28, 28)
##TestTarget (2724,)




import tensorflow as tf
import numpy as np


#we only have one layer in this case
def weighted_matrix(inputs, numHiddenUnits):
    with tf.variable_scope("weighted_matrix", reuse=tf.AUTO_REUSE):
        w = tf.get_variable(
            name = "weight", 
            shape = [784, numHiddenUnits],
            initializer = tf.contrib.layers.xavier_initializer(),
            dtype = tf.float64
            
            );
        #b = tf.Variable(0.0, name='bias') #where do we add the bias?
        x = tf.placeholder(dtype = tf.float64, name = 'inputs'); #inputs 
        
        weighted_sum = tf.matmul(x,w) 
        #res = tf.Session().run(weighted_sum, feed_dict = {x: inputs})
        #return res



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



    start = 0;
    BATCH_SIZE = 500;
    TRAINING_NUM_POINTS = 15000;
    trainNumBatches = TRAINING_NUM_POINTS/BATCH_SIZE;	
    NUM_ITERATIONS = 5000;

    trainX = np.reshape(trainData, (TRAINING_NUM_POINTS, -1) );
    trainY = trainTarget.astype(np.float64);


    #descend the gradient 
    for i in range(NUM_ITERATIONS):
    
        start = (start+ BATCH_SIZE) % TRAINING_NUM_POINTS;
        end = start + BATCH_SIZE;
        
        weighted_matrix(trainX[start:end], 1000)        
