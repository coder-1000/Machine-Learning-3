import numpy as np
import tensorflow as tf
import weightedmatrix as wm
import matplotlib.pyplot as plt

NUM_HIDDEN_UNITS = 500
NUM_CLASSES = 10
 
BATCH_SIZE = 500
NUM_ITERATIONS = 5000
TRAIN_SIZE = 15000
VALID_SIZE = 1000


lamda = 0.01

#implement weight decay

if __name__ == "__main__":
	learning_rate = [0.001, 0.005, 0.0001]
	misclassTrain = []
	crossEntropyLossTrain = []
	misclassValid = []
	crossEntropyLossValid = []
	misclassTest = []
	crossEntropyLossTest = []

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

	trainData = np.reshape(trainData.astype(float),(15000,784))
	validData = np.reshape(validData.astype(float),(1000,784))
	testData = np.reshape(testData.astype(float),(2724,784))
	trainTarget = trainTarget.astype(float)
	validTarget = validTarget.astype(float)
	testTarget = testTarget.astype(float)


	for lR in learning_rate:
      	
		x = tf.placeholder(tf.float64, [None, 784])
		y = tf.placeholder(tf.float64)

        #the weighted sum matrix should return a 500 x batch size vector 
		w1,b1,sLayer1 = wm.weighted_matrix(x, NUM_HIDDEN_UNITS) #our hidden layer

        #our x output layer should also be 500 x batch size
		xLayer1 = tf.nn.relu(sLayer1)

		#the weighted sum matrix should return a 500 x batch size vector 
		w2,b2,sLayer2 = wm.weighted_matrix(xLayer1, NUM_HIDDEN_UNITS) #our hidden layer

        #our x output layer should also be 500 x batch size
		xLayer2 = tf.nn.relu(sLayer2)

        #expend dimensions     
		fatY = tf.one_hot(tf.cast(y,dtype=tf.int32), NUM_CLASSES)
	
		#500x10
		outputW,outputB,softmaxW = wm.weighted_matrix(xLayer2, NUM_CLASSES)

        #add some weight decay to prevent overfitting
		loss = tf.nn.softmax_cross_entropy_with_logits(logits = softmaxW, labels = fatY);
		#weight decay
		decay = lamda /2 * tf.reduce_sum(outputW**2)
   		loss = tf.add(tf.reduce_mean(loss,0),decay)

		with tf.Session() as sess:
			Step = tf.train.AdamOptimizer(lR).minimize(loss)
			tf.global_variables_initializer().run()
			start = 0

			for i in range(NUM_ITERATIONS):
    			
				#training data
				start = (start+ BATCH_SIZE) % TRAIN_SIZE
				end = start + BATCH_SIZE
				batchx = trainData[start:end]
				batchy = trainTarget[start:end]
	
				sess.run(Step, feed_dict = {x:batchx, y:batchy})
				#not for training purposes
				trainLoss = sess.run(loss, feed_dict = {x:trainData, y:trainTarget})
				validLoss = sess.run(loss, feed_dict = {x:validData, y:validTarget})
				testLoss = sess.run(loss, feed_dict = {x:testData, y:testTarget})
				start = (start + BATCH_SIZE) % 15000  
				
				if((i+1)%BATCH_SIZE == 0):
					print(trainLoss)
					predValTrain = tf.nn.softmax(softmaxW)
					predValTrain = sess.run(predValTrain, feed_dict = {x:trainData,y:trainTarget})
					actValTrain = sess.run(y,feed_dict = {y:trainTarget})
					misclassificationTrain = (tf.not_equal(tf.argmax(predValTrain,axis=1), tf.argmax(actValTrain,axis=1)))
					misclassTrain.append(misclassificationTrain)
					crossEntropyLossTrain.append(trainLoss)

					#validation data
					predValValid = tf.nn.softmax(softmaxW)
					predValValid = sess.run(predValValid, feed_dict = {x:validData,y:validTarget})
					actValValid = sess.run(y,feed_dict = {y:validTarget})
					misclassificationValid = (tf.not_equal(tf.argmax(predValValid,axis=1), tf.argmax(actValValid,axis=1)))
					misclassValid.append(misclassificationValid)
					crossEntropyLossValid.append(trainLoss)
				
					#testing data
					predValTest = tf.nn.softmax(softmaxW)
					predValTest = sess.run(predValTest, feed_dict = {x:testData,y:testTarget})
					actValTest = sess.run(y,feed_dict = {y:testTarget})
					misclassificationTest = (tf.not_equal(tf.argmax(predValTest,axis=1), tf.argmax(actValTest,axis=1)))
					misclassTest.append(misclassificationTest)
					crossEntropyLossTest.append(testLoss)

				
			
					


		epochs = [x for x in range(NUM_ITERATIONS/BATCH_SIZE)]
		plt.figure()
		plt.plot(epochs, np.array(misclassTrain),label='learning rate:0.005')
		plt.plot(epochs, np.array(misclassValid),label='learning rate:0.005')
		plt.plot(epochs, np.array(misclassTest),label='learning rate:0.005')
		plt.legend()
		plt.xlabel('Epochs')
		plt.ylabel('Misclassification')
		plt.title('Epochs vs Misclassification')
		plt.show()


		plt.figure()
		plt.plot(epochs, np.array(crossEntropyLossTrain),label='learning rate:0.005')
		plt.plot(epochs, np.array(crossEntropyLossValid),label='learning rate:0.005')
		plt.plot(epochs, np.array(crossEntropyLossTest),label='learning rate:0.005')
		plt.legend()
		plt.xlabel('Epochs')
		plt.ylabel('Entropy Loss')
		plt.title('Epochs vs Entropy Loss')
		plt.show()






