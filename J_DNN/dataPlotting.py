import matplotlib.pyplot as plt
import csv

files = ["ValMSE.txt", "TrainMSE.txt"]
data = []

# Must be set according to data collected
epochs = range(1,1500+1)
nExperiments = 4
nHiddenLayers = 4 # Number of hidden layers in the first experiment

# Read data
for file in files:
    with open(file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        temp_data = []
        for row in csv_reader:
            row_data = []
            for datapoint in row:
                row_data.append(float(datapoint))
            temp_data.append(row_data)
    data.append(temp_data)

# Plot validation MSE
plt.figure()
for i in range(nExperiments):
    plt.plot(epochs, data[0][i], label=str(nHiddenLayers+i) + " hidden layers")
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error [cm]')
plt.title('Validation MSE')
plt.legend()

# Plot training MSE
plt.figure()
for i in range(nExperiments):
    plt.plot(epochs, data[1][i], label=str(nHiddenLayers+i)+" hidden layers")
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error [cm]')
plt.title('Training MSE')
plt.legend()

plt.show()