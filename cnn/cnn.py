import sys
import csv
import pandas as pd
import numpy as np
import statistics 

from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Conv1D, Flatten, Dense
from tensorflow.keras.layers import MaxPool1D
from tensorflow.keras.layers import Dropout, BatchNormalization
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint
from tensorflow.keras.utils import plot_model
from tensorflow.data import Dataset
from tensorflow import keras

# load data
def load_data(data_file, targets_file):
    data = []
    with open(data_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
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
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            temp_data = []
            temp_data.append(float(row[0]))
            temp_data.append(float(row[1]))
            temp_data = np.array(temp_data)
            targets.append(temp_data)
    
    return (np.array(data), np.array(targets))

# CNN class
class CNN():
    def build_cnn_layer(self, net, n_filters, use_batch_norm=False, use_max_pool=True):
        net = Conv1D(n_filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(net)
        if use_batch_norm:
            net = BatchNormalization()(net)
        if use_max_pool:
          net = MaxPool1D(2)(net) # most likely don't use max pooling as the input data is so small only 16
        return net

    def load_model(self, model_path):
        """
        Loads the CNN model, which has been pretrained on colab

        Parameters
        ----------
        model_path (string): The path to the stored model
        """
        self.model = keras.models.load_model(model_path)

    def build_model(self, input_size, n_layers, n_filters, use_batch_norm, plot_and_print_summary=False):
        """
        Builds the CNN model

        Parameters
        ----------
        input_size (tuple): The inputs size of the image for the model
        n_layers (int): The number of conv layers
        n_filters (itn): The number of filters in the first layer
        use_batch_norm (bool): If True there will be used batch normalization
        """
        inputs = Input(input_size)
        net = inputs

        # Create the conv blocks

        for i in range(n_layers):
            use_max_pool = False
            if i % 2 == 0:
              use_max_pool = True
            net = self.build_cnn_layer(net, n_filters, use_batch_norm, False)
            print(net.get_shape())
            net = self.build_cnn_layer(net, n_filters, use_batch_norm, use_max_pool)
            print(net.get_shape())
            n_filters *= 2

        # Create dense layers
        net = Flatten()(net)
        print(net.get_shape())
        
        net = Dense(256, activation='relu')(net)
        net = Dropout(0.5)(net)
        net = BatchNormalization()(net)
        print(net.get_shape())


        net = Dense(128, activation='relu')(net)
        net = Dropout(0.5)(net)
        net = BatchNormalization()(net)

        net = Dense(2)(net)
        self.model = Model(inputs=inputs, outputs=net)
        print(net.get_shape())


        if plot_and_print_summary:
            self.model.summary()

            # Try to make a PDF with a plot of the model
            try:
                plot_model(self.model, to_file='model.pdf', show_shapes=True)
            except:
                print("To get plot of model fix error:", sys.exc_info()[0])

    def train(self, train, val, steps_per_epoch, epochs, lr):
        """
        Trains the model

        Parameters
        ----------
        train_set (generator): A image+mask generator for the training data
        val_set (generator): A image+mask generator for the validation data
        steps_per_epoch (int): The number of steps per epoch
        epochs (int): The number of epochs
        lr (float): The learning rate
        """
        # Compile the model
        self.model.compile(optimizer=Adam(learning_rate=lr), loss='mean_squared_error', metrics=['mean_squared_error'])

        # Make a learning rate scheduler
        def scheduler(epoch):
            if epoch < 3:
                return 0.0001
            else:
                return 0.00005

        lr_schedule = LearningRateScheduler(scheduler)

        # Create a ModelCheckpoint callback to save teh best version of the model as it trains
        checkpoint_name = "/content/drive/MyDrive/project_in_AI_data/models/retina_extraction.hdf5"
        model_checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_loss', verbose=1, save_best_only=True)

        # Train the model
        # Validation_steps is based on "validation size / batch size"
        history = self.model.fit(train, steps_per_epoch=steps_per_epoch, epochs=epochs, validation_data=val,
                       validation_steps=6, callbacks=[model_checkpoint, lr_schedule], batch_size=10)
        

        # Load the best version of the model
        self.model.load_weights(checkpoint_name)

        return history

    def predict(self, test_data, test_targets):
        """
        Predict on the given data

        Parameters
        ----------
        Test data: test data
        test_targets: test targets
        """
        prediction = self.model.predict(test_data)
        print("prediction and targets")
        print(prediction)
        print(test_targets)

if __name__ == '__main__':
    # Files
    train_data_file = 'data/train/features.txt'
    train_targets_file = 'data/train/targets.txt'
    val_data_file = 'data/val/features.txt'
    val_targets_file = 'data/val/targets.txt'

    train_data, train_labels = load_data(train_data_file, train_targets_file)
    val_data, val_labels = load_data(val_data_file, val_targets_file)
    train_dataset = Dataset.from_tensor_slices((train_data, train_labels)).batch(16)
    val_dataset = Dataset.from_tensor_slices((val_data, val_labels)).batch(10)


    # Main
    print(train_data[1].shape)
    input_size = (16, 1)  # The input size for the model
    val_accuracy = []
    val_loss = []
    train_loss = []
    train_accuracy = []
    for i in range(2,6):
        val_error_loop = []
        train_error_loop = []
        error = []
        for j in range(3):
            n_layers = i  # The number of con layers in the network
            filter = 32   # The first layers number of filters
            use_batch_norm = True  # Use batch normalization or not
            cnn = CNN()
            cnn.build_model(input_size, n_layers, filter, True)
            epochs = 400  # The number of epochs to train over
            steps_per_epoch = int(240/16) # based on number of training samples and batch size "x_samples/epocs"

            lr = 1e-2  # The learning rate to start with
            history = cnn.train(train_dataset, val_dataset, steps_per_epoch, epochs, lr)
            val_error_loop.append(history.history["val_mean_squared_error"])

        for k in range(len(val_error_loop[0])):
            err = 0
            for l in range(3):
                err += val_error_loop[l][k]
                error.append(err/3)

        val_accuracy.append(error)

    min_val_accuracy_conv4 = 1000
    min_val_accuracy_conv6 = 1000
    min_val_accuracy_conv8 = 1000
    min_val_accuracy_conv10 = 1000

    # Find the minimum error
    for i in range(len(val_accuracy[0])):
        if min_val_accuracy_conv4 > val_accuracy[0][i]:
            min_val_accuracy_conv4 = val_accuracy[0][i]

    for i in range(len(val_accuracy[1])):
        if min_val_accuracy_conv6 > val_accuracy[1][i]:
            min_val_accuracy_conv6 = val_accuracy[1][i]

    for i in range(len(val_accuracy[2])):
        if min_val_accuracy_conv8 > val_accuracy[2][i]:
            min_val_accuracy_conv8 = val_accuracy[2][i]

    for i in range(len(val_accuracy[3])):
        if min_val_accuracy_conv10 > val_accuracy[3][i]:
            min_val_accuracy_conv10 = val_accuracy[3][i]

    print(min_val_accuracy_conv4)
    print(min_val_accuracy_conv6)
    print(min_val_accuracy_conv8)
    print(min_val_accuracy_conv10)

    import matplotlib.pyplot as plt
    epochs = []
    for i in range(0,400):
        epochs.append(i)

    plt.plot(epochs, val_accuracy[0], label="4 error: 1.20")
    plt.plot(epochs, val_accuracy[1], label="6 error: 1.21")
    plt.plot(epochs, val_accuracy[2], label="8 error: 1.10")
    plt.plot(epochs, val_accuracy[3], label="10 error: 1.79")
    plt.xlabel('epochs')
    # naming the y axis
    plt.ylabel('Mean squared error [cm]')
    # giving a title to my graph
    plt.title('Validation error vs number of convolutional layers')
    
    # show a legend on the plot
    plt.legend()

    # train the best network
    n_layers = 4  # The number of con layers in the network
    filter = 32   # The first layers number of filters
    use_batch_norm = True  # Use batch normalization or not
    cnn = CNN()
    cnn.build_model(input_size, n_layers, filter, True)
    epochs = 600  # The number of epochs to train over
    steps_per_epoch = int(240/16) # based on number of training samples and batch size "x_samples/epocs"

    lr = 1e-2  # The learning rate to start with
    history = cnn.train(train_dataset, val_dataset, steps_per_epoch, epochs, lr)
