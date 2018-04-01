import numpy as np
import tensorflow as tf
import weightedmatrix as wm

NUM_HIDDEN_UNITS = 1000;
NUM_UNITS_OUTPUT_LAYER = 10;
BATCH_SIZE = 500;
NUM_ITERATIONS = 500;
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

    trainX = np.reshape(trainData, (15000, -1) );
    trainY = trainTarget.astype(np.float64);
    
    
    x0 = tf.placeholder(tf.float64, name="input_layer", shape = [None, NUM_PIXELS]);
    y = tf.placeholder(tf.int64, name="target");
   
    faty = tf.one_hot(indices = y, depth = 10);
    [w1, b1, s1] = wm.weighted_matrix(x0, NUM_HIDDEN_UNITS);
    x1 = tf.nn.relu(s1);
    
    [w2, b2, s2] = wm.weighted_matrix(x1, NUM_UNITS_OUTPUT_LAYER);
  
    cross = tf.nn.softmax_cross_entropy_with_logits(logits = s2, labels = faty);
    weightDecay = (tf.reduce_sum(w1**2) + tf.reduce_sum(w2**2) )*(LAMDA)
    
    loss = tf.reduce_mean((cross + weightDecay)/2.0)
    
    learningRate = 0.005

    trainStep = tf.train.GradientDescentOptimizer(learningRate).minimize(loss);
    
    initializer = tf.global_variables_initializer();
    
    crossValues = []

    with tf.Session() as sess:
        
        sess.run(initializer)
        start = 0;
       
        num_batches = 150000 // BATCH_SIZE
        
        for i in range(NUM_ITERATIONS):
            
            end = start + BATCH_SIZE;
            [cv, blah] = sess.run([loss, trainStep], feed_dict={x0:  trainX[start:end], y: trainY[start: end] })
            if( (i% num_batches) == 0):
                crossValues.append(cv)
            start = end
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
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
