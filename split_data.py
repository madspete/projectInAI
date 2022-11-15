from cv2 import split
from sklearn.utils import shuffle
import csv


def load_data():
    data = []
    data_file = "data/data.txt"
    with open(data_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=' ')
        for row in csv_reader:
            temp_data = []
            for i in range(16):
                temp_data.append(float(row[i]))
            data.append(temp_data)
    
    targets = []
    data_targets = "data/gt/gt_targets.txt"
    with open(data_targets) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=' ')
        for row in csv_reader:
            temp_data = []
            temp_data.append(float(row[0]))
            temp_data.append(float(row[1]))
            targets.append(temp_data)
    
    return data, targets

# Create helper functions for data manipulation
def split_data(data, target):

    data, target = shuffle(data, target)

    n_valid = int(len(data) * 0.2)
    data_train = data[:-n_valid]
    targets_train = target[:-n_valid]
    data_valid = data[-n_valid:]
    targets_valid = target[-n_valid:]

    return data_train, targets_train, data_valid, targets_valid

def write_to_file(data_train, targets_train, data_valid, targets_valid):
    train_features_file = open("data/train/features.txt", "w")
    train_targets_file = open("data/train/targets.txt", "w")
    for i in range(len(data_train)):
        # Write train data
        for j in range(len(data_train[i])):
            if j > 0:
                train_features_file.write(",")
            train_features_file.write(str(data_train[i][j]))
        train_features_file.write("\n")

        # write targets
        train_targets_file.write(str(targets_train[i][0]) + ",")
        train_targets_file.write(str(targets_train[i][1]) + "\n")

    val_features_file = open("data/val/features.txt", "w")
    val_targets_file = open("data/val/targets.txt", "w")
    for i in range(len(data_valid)):
        # Write train data
        for j in range(len(data_valid[i])):
            if j > 0:
                val_features_file.write(",")
            val_features_file.write(str(data_valid[i][j]))
        val_features_file.write("\n")

        # write targets
        val_targets_file.write(str(targets_valid[i][0]) + ",")
        val_targets_file.write(str(targets_valid[i][1]) + "\n")


if __name__ == '__main__':
    data, targets = load_data()
    data_train, targets_train, data_valid, targets_valid = split_data(data, targets)
    write_to_file(data_train, targets_train, data_valid, targets_valid)

