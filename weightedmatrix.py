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
    W = tf.get_variable("weight", shape=[784, numHiddenUnits],initializer=tf.contrib.layers.xavier_initializer());
    #b = tf.Variable(0.0, name='bias') #where do we add the bias?
    x = tf.placeholder("inputs", tf.float32); #inputs 
    
    weighted_sum = tf.matmul(x,W) #matrix multiplication 
    
    return weighted_sum



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



    with tf.Session() as sess:
        start = 0;
        BATCH_SIZE = 500;
        TRAINING_NUM_POINTS = 15000;
        trainNumBatches = TRAINING_NUM_POINTS/BATCH_SIZE;	
        NUM_ITERATIONS = 5000;

        trainX = np.reshape(trainData, (TRAINING_NUM_POINTS, -1) );
        trainY = trainTarget.astype(np.float64);


        #descend the gradient 
        for i in range(NUM_ITERATIONS):
        
            start = (start+ batchSize) % NUM_POINTS;
            end = start + batchSize;

            res = sess.run(weighted_sum(inputs, numHiddenUnits), feed_dict={inputs: trainX[start : end] , numHiddenUnits:1000}); 
            print(res.shape)
