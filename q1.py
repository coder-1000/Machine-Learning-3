#question 1
'''
1. layer-wise building block: Write a vectorized Tensorflow Python function that takes the
2
1.2 Effect of hyperparameters [5 pt.] 1 NEURAL NETWORKS [25 PT.]
hidden activations from the previous layer then return the weighted sum of the inputs(i.e. the
z) for the current hidden layer. You will also initialize the weight matrix and the biases in
the same function. You should use Xavier initialization for the weight matrix. Your function
should be able to compute the weighted sum for all the data points in your mini-batch at
once using matrix multiplication. It should not contain loops over the training examples in
the mini-batch. The function should accept two arguments, the input tensor and the number
of the hidden units. Include the snippets of the Python code. [3 pt.]
'''

#initialize weight matrix and bias
#then return weight sum of inputs ( wx + b)??
def foo(x numHidden):
    #bias = 0
    #weight matrix = xavier initialized
    #https://www.tensorflow.org/api_docs/python/tf/get_variable
    #https://www.tensorflow.org/api_docs/python/tf/contrib/layers/xavier_initializer    
    
    #stuff that might be useful

    #https://stackoverflow.com/questions/47167409/using-weights-initializer-with-tf-nn-conv2di
    #https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/convolutional_network_raw.py
