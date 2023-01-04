import tensorflow as tf
import csv
import statistics
import numpy as np

# Function written by Mads Holm Peters
def load_data(data_file, targets_file, delim):
    data = []
    with open(data_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=delim)
        for row in csv_reader:
            temp_data = []
            for i in range(16):
                temp_data.append(float(row[i]))
            # temp_data = np.array(temp_data)
            data.append(temp_data)
    
    # Normalize input data
    means = []
    stdev = []
    for i in range(len(data[0])):
      temp = []
      for j in range(len(data)):
        temp.append(data[j][i])
      
      means.append(statistics.mean(temp))
      stdev.append(statistics.stdev(temp))

    standardized_data = []
    for i in range(len(data)):
      temp_data = []
      for j in range(len(data[i])):
        temp_data.append((data[i][j] - means[j]) / stdev[j])
        #temp_data.append(data[i][j])
      temp_data = np.array(temp_data)
      standardized_data.append(temp_data)
    
    targets = []
    with open(targets_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=delim)
        for row in csv_reader:
            temp_data = []
            temp_data.append(float(row[0]))
            temp_data.append(float(row[1]))
            temp_data = np.array(temp_data)
            targets.append(temp_data)
    
    return (np.array(data), np.array(targets))

class MLP():

	def buildLayer(self, network, nNeurons, activiationFunction='relu', useDropout = True, useBatchNorm = True):
		network = tf.keras.layers.Dense(nNeurons,activation=activiationFunction)(network)
		if useDropout:
			network = tf.keras.layers.Dropout(0.5)(network)
		if useDropout:
			network = tf.keras.layers.BatchNormalization()(network)
		return network

	def buildModel(self, inputSize, nHiddenLayers, nFirstLayerNeurons):

		# See this for building a model using Keras
		# https://www.tensorflow.org/api_docs/python/tf/keras/Model

		# Define the input size of the model
		input = tf.keras.layers.Input(inputSize)

		# Initialize the network struture so it can be expanded upon
		network = input

		# Create nHiddenLayers in the model.
		# The structure of this network is pyramid-like, with the middel layer being the largest.
		for i in range(nHiddenLayers):
			if i < nHiddenLayers/2:
				network = self.buildLayer(network,int(nFirstLayerNeurons))
				print("Layer " + str(i) + " with " + str(nFirstLayerNeurons) + " neurons")
				if i + 1  < nHiddenLayers/2: # Don't increment if the next layer is the middel layer
					nFirstLayerNeurons *= 2
			else:
				if np.mod(nHiddenLayers,2) == 1:
					nFirstLayerNeurons /= 2	
				network = self.buildLayer(network,int(nFirstLayerNeurons))
				print("Layer " + str(i) + " with " + str(nFirstLayerNeurons) + " neurons")
				if np.mod(nHiddenLayers,2) == 0:
					nFirstLayerNeurons /= 2

		network = tf.keras.layers.Dense(2)(network)
		
		# For debugging the shape of the current output node of the network
		#print(network.shape)

		# Construct the model based on the network structure
		self.model = tf.keras.Model(inputs=input, outputs=network)

	def train(self, trainingDataset, validationDataset, epochs, learningRate, modelName):
		
		# Compile the model
		# Since this model does regression, the loss-function is chosen as mean squared error
		self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learningRate), loss='mean_squared_error', metrics=['mean_squared_error'])

		# Make a learning rate scheduler
		def scheduler(epoch):
			if epoch < 3:
				return 0.0001
			else:
				return 0.00005

		lr_schedule = tf.keras.callbacks.LearningRateScheduler(scheduler)

		# Create a ModelCheckpoint callback to save the best version of the model as it trains
		checkpoint_name = modelName + ".hdf5"
		model_checkpoint = tf.keras.callbacks.ModelCheckpoint(checkpoint_name, monitor='val_loss', verbose=1, save_best_only=True)

		# Train the model
		# Validation_steps is based on "validation size / batch size"
		#stats = self.model.fit(trainingDataset, epochs=epochs, validation_data=validationDataset, callbacks=[model_checkpoint, lr_schedule], batch_size=10)
		stats = self.model.fit(trainingDataset, epochs=epochs, validation_data=validationDataset,
						validation_steps=6, callbacks=[model_checkpoint, lr_schedule], batch_size=10)


		# Load the best version of the model
		self.model.load_weights(checkpoint_name)

		return stats

	def predict(self):
		pass

def writeMetricsToFile(filename, data):
	with open(str(filename) + ".txt", 'a') as f:
		writer = csv.writer(f)
		writer.writerow(data)

if __name__ == '__main__':

	performTrain = False

	# Training and validation filepaths
	trainDataFilePath = '../data/train/features.txt'
	trainTargetFilePath = '../data/train/targets.txt'
	valDataFilePath = '../data/val/features.txt'
	valTargetFilePath = '../data/val/targets.txt'
	testDataFilePath = '../data/test_data.txt'
	testTargetFilePath = '../data/test/test.txt'

	# Loading the data.
	# Training dataset consists of 240 samples.
	# Validation dataset consists of 60 samples.
	# Test dataset consists of XX samples.
	trainingData, trainingLabels = load_data(trainDataFilePath, trainTargetFilePath, ',')
	validationData, validationLabels = load_data(valDataFilePath, valTargetFilePath, ',')
	testData, testLabels = load_data(testDataFilePath,testTargetFilePath, ' ')
	print("Training data shape: ", trainingData.shape) # 240 training samples
	print("Training labels shape: ", trainingLabels.shape)
	print("Validation data shape: ", validationData.shape) # 60 validation samples
	print("Validation labels shape: ", validationLabels.shape)
	print("Test data shape: ", testData.shape) # XX test samples
	print("Test labels shape: ", testLabels.shape)

	# Creating the datasets.
	trainingBatchSize = 16
	validationBatchSize = 10
	# Training dataset is created in batches of size 16, yielding 15 (240/16) total batches.
	# Validation dataset is created in batches of size 10, yielding 6 (60/10) total batches.
	# The batch size influences how quickly the network learns. If the training dataset is split into e.g. 16 batches, the network is updated 16 times in one epoch.
	# Using batches also has advantages when it comes to memory-usage, and TensorFlow running muliplte batches at the same time on a GPU.
	# This is why the validation dataset is also split into batches.
	trainingDataset = tf.data.Dataset.from_tensor_slices((trainingData, trainingLabels)).batch(trainingBatchSize)
	validationDataset = tf.data.Dataset.from_tensor_slices((validationData, validationLabels)).batch(validationBatchSize)
	testDataset = tf.data.Dataset.from_tensor_slices((testData, testLabels)).batch(1)

	if performTrain:
		

		for i in range(4):

			avgValMSE = []
			avgTrainMSE = []

			for j in range(5):

				# MODEL PARAMETERS
				# Define the input size of the model. Each datapoint contains 16 values.
				inputSize = 16
				# Define the number of hidden layers in the model, this includes the output-layer, even though it is not hidden.
				nHiddenLayers = 4 + i
				# Number of neurons in the first hidden layer
				nFirstLayerNeurons = 30

				# TRAINING PARAMETERS
				# Number of epochs to train
				epochs = 1500
				# Starting learning rate
				learningRate = 1e-2

				# Create and train model
				modelName = str(nHiddenLayers) + "HiddenLayers"
				mlp = MLP()
				mlp.buildModel(inputSize,nHiddenLayers,nFirstLayerNeurons)
				stats = mlp.train(trainingDataset, validationDataset, epochs, learningRate, modelName)

				# Get metrics from training
				valMSE = stats.history["val_mean_squared_error"]
				trainMSE = stats.history["mean_squared_error"]

				avgValMSE.append(valMSE)
				avgTrainMSE.append(trainMSE)

			print(avgValMSE)
			avgValMSE = np.average(avgValMSE,axis= 0)
			print(avgValMSE)
			avgTrainMSE = np.average(avgTrainMSE,axis= 0)
			writeMetricsToFile("ValMSE", avgValMSE)
			writeMetricsToFile("TrainMSE", avgTrainMSE)
	else:

		# Load model
		model = tf.keras.models.load_model('4HiddenLayers.hdf5')
		# Make predictions
		predictions = model.predict(testDataset)
		# Calculated differences
		diff = predictions - testLabels
		squaredDiff = np.square(diff)
		meanX = np.sum(squaredDiff[0]) / len(squaredDiff[0])
		meanY = np.sum(squaredDiff[1]) / len(squaredDiff[1])
		print(meanX)
		print(meanY)
		
	