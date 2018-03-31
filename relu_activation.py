import numpy as np
import tensorflow as tf

NUM_HIDDEN_UNITS = 1000;
NUM_HIDDEN_UNITS_OUTPUT_LAYER = 10;
 
if __name__ == "__main__":
    misclassifications = []
    learning_rate = [0.001, 0.005, 0.0001]
        

    for lR in learning_rate:
        # we now need a loss function of some kind
        xLayer0 = inputs; #our initial inputs
        #the weighted sum matrix should return a 1000x1 vector 
        sLayer1 = weighted_matrix(inputs, NUM_HIDDEN_UNITS) #our hidden layer
        #our x output layer should also be 1000x1
        xLayer1 = tf.nn.relu(sLayer1); #technically our output layer
        #this should output a 1x10 vector with the 10 classes
        outputLayer = weighted_matrix(xLayer1, NUM_HIDDEN_UNITS_OUTPUT_LAYER);
        
        #add some weight decay to prevent overfitting
        loss = tf.nn.softmax_cross_entropy_with_logits(logits = outputLayer, label = actualLabel);
	 trainStep = tf.train.GradientOptimizer(lR).minimize(loss);

	 #misclassfications, add to the misclassifications list if they are not equal, therfore misclassified
	 missclassifications.append(not tf.equal(tf.transpose(outputLayer), actualLabel))
