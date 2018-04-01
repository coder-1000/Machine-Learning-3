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

NUM_PIXELS = 784;
batch_num = 0;

#we only have one layer in this case
#input shape : [batchsize, pixels]
#initialize a layer in a cnn
def weighted_matrix(inputs, numHiddenUnits):
    global batch_num    
    #gives a xavier distribution for our initial weights
    w = tf.get_variable(
        name = 'w{0}'.format(batch_num), 
        shape = [inputs.shape[1], numHiddenUnits], 
        initializer = tf.contrib.layers.xavier_initializer(),
        dtype =tf.float64
    ) 
    b = tf.get_variable(
            initializer = tf.zeros_initializer(), 
            name = 'b{0}'.format(batch_num), 
            dtype = tf.float64,
	    shape = (numHiddenUnits)   
	)
 
    batch_num = batch_num + 1;
        
    weighted_sum = tf.matmul(inputs, w) + b
   
    return [w, b, weighted_sum]


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
        
        res = weighted_matrix(trainX[start:end], 1000)      
