from cnn import CNN
import csv
import numpy as np

if __name__ == '__main__':
    cnn = CNN()
    cnn.load_model("cnn/models/retina_extraction.hdf5")

    data_path_test = "data/test_data.txt"
    targets_test_path = "data/test/test.txt"

    test_data = []
    with open(data_path_test) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=' ')
        for row in csv_reader:
            data = []
            for i in range(16):
                data.append(float(row[i]))
            test_data.append(data)

    test_data = np.array(test_data)
    
    test_targets = []
    with open(targets_test_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=' ')
        for row in csv_reader:
            data = []
            data.append(float(row[0]))
            data.append(float(row[1]))
            test_targets.append(data)
            
    print(test_targets)
    test_targets = np.array(test_targets)

    for i in range(len(test_targets)):
        cnn.predict(np.array([test_data[i]]), np.array([test_targets[i]]))