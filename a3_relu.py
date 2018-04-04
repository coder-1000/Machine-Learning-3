import numpy as np
import tensorflow as tf
import weightedmatrix as wm
import matplotlib.pyplot as plt

NUM_HIDDEN_UNITS = 1000
NUM_CLASSES = 10
 
BATCH_SIZE = 500
NUM_ITERATIONS = 50000
TRAIN_SIZE = 15000
VALID_SIZE = 1000


lamda = 0.0003

#implement weight decay

if __name__ == "__main__":
	learning_rate = [0.005]
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

        #the weighted sum matrix should return a 1000x500 vector 
		w1,b1,s1 = wm.weighted_matrix(x, NUM_HIDDEN_UNITS) #our hidden layer

        #our x output layer should also be 1000x500
		s1 = tf.nn.relu(s1)

        #expend dimensions     
		fatY = tf.one_hot(tf.cast(y,dtype=tf.int32), NUM_CLASSES)
	
		#500x10
		w2,b2,s2 = wm.weighted_matrix(s1, NUM_CLASSES)

        #add some weight decay to prevent overfitting
		loss = tf.nn.softmax_cross_entropy_with_logits(logits = s2, labels = fatY);
		#weight decay
		decay = lamda /2 * tf.reduce_sum(w2**2)
   		loss = tf.add(tf.reduce_mean(loss,0),decay)

		misclassification = tf.reduce_mean(tf.cast((tf.not_equal(tf.cast(tf.argmax(s2,axis=1),tf.float64),y)), tf.float32))
		print("cross entropy loss train:")
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

				trainMisclass = sess.run(misclassification, feed_dict = {x:trainData, y:trainTarget})
				validMisclass = sess.run(misclassification, feed_dict = {x:validData, y:validTarget})
				testMisclass = sess.run(misclassification, feed_dict = {x:testData, y:testTarget})

				start = (start + BATCH_SIZE) % 15000  
				
				if(i%BATCH_SIZE == 0):
					print(trainLoss)
					#training data
					misclassTrain.append(trainMisclass)
					crossEntropyLossTrain.append(np.mean(trainLoss))

					#validation data
					misclassValid.append(validMisclass)
					crossEntropyLossValid.append(trainLoss)
				
					#testing data
					misclassTest.append(testMisclass)
					crossEntropyLossTest.append(testLoss)

			print("misclassification train:")
			print(misclassTrain)
			print("cross entropy loss valid:")
			print(crossEntropyLossValid)
			print("misclassification test:")
			print(misclassValid)
			print("cross entropy loss test:")
			print(crossEntropyLossTest)
			print("misclassification test:")
			print(misclassTest)

				
			
					


		epochs = [x for x in range(NUM_ITERATIONS/BATCH_SIZE)]
		plt.figure()
		plt.plot(epochs, np.array(misclassTrain),label='training set')
		plt.plot(epochs, np.array(misclassValid),label='validation set')
		plt.plot(epochs, np.array(misclassTest),label='testing set')
		plt.legend()
		plt.xlabel('Epochs')
		plt.ylabel('Misclassification')
		plt.title('Epochs vs Misclassification')
		plt.show()


		plt.figure()
		plt.plot(epochs, np.array(crossEntropyLossTrain),label='training set')
		plt.plot(epochs, np.array(crossEntropyLossValid),label='validation set')
		plt.plot(epochs, np.array(crossEntropyLossTest),label='testing set')
		plt.legend()
		plt.xlabel('Epochs')
		plt.ylabel('Entropy Loss')
		plt.title('Epochs vs Entropy Loss')
		plt.show()















