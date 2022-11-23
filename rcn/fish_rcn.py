#!/usr/bin/env python
from pyrcn.echo_state_network import ESNRegressor, ESNClassifier
from pyrcn.extreme_learning_machine import ELMRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import Ridge as skRidge
from sklearn.datasets import make_blobs
from pyrcn.datasets import mackey_glass

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
import numpy as np
import csv
import sys
import math

esn = ESNRegressor()

X, y = mackey_glass(n_timesteps=8000)

X_train = []
y_train = []
X_test = []
y_test = []

with open('data/train/features.txt') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
          i = 0
          x = []
          while i < 16:
            x.append(float(row[i]))
            i = i+1
          X_train.append(x)
          line_count += 1

with open('data/train/targets.txt') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
          i = 0
          x = []
          while i < 2:
            x.append(float(row[i]))
            i = i+1
          y_train.append(x)
          line_count += 1

with open('data/val/features.txt') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
          i = 0
          x = []
          while i < 16:
            x.append(float(row[i]))
            i = i+1
          X_test.append(x)
          line_count += 1

with open('data/val/targets.txt') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
          i = 0
          x = []
          while i < 2:
            x.append(float(row[i]))
            i = i+1
          y_test.append(x)
          line_count += 1

# Define Train/Test lengths
# trainLen = 4000
# X_train, y_train = X[:trainLen], y[:trainLen]
# X_test, y_test = X[trainLen:], y[trainLen:]
X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)
# print(y_test.shape)
# print(len(X_train))


# Initialize and train an ELMRegressor and an ESNRegressor
esn = ESNRegressor().fit(X=X_train, y=y_train)
elm = ELMRegressor(regressor=skRidge()).fit(X=X_train, y=y_train)
print("Fitted models")

y_pred = elm.predict(X_test)

# print(y_pred.shape)
# print(X_train.shape)
# print(y_train.shape)

x_mse = 0
y_mse = 0
mse = 0
y_dist_list = []
x_dist_list = []
for i in range(len(y_pred)):
  x_dist = math.sqrt(pow(y_test[i][0] - y_pred[i][0],2))
  y_dist = math.sqrt(pow(y_test[i][1] - y_pred[i][1],2))
  y_dist_list.append(y_dist)
  x_dist_list.append(x_dist)
  dist = math.sqrt(pow((y_test[i][0] - y_pred[i][0]),2) + pow((y_test[i][1] - y_pred[i][1]),2))
  x_mse += pow(x_dist,2)
  y_mse += pow(y_dist,2)
  mse += pow(dist,2)
x_mse = x_mse/len(y_pred)
y_mse = y_mse/len(y_pred)
mse = mse/len(y_pred)
x_rmse = math.sqrt(x_mse)
y_rmse = math.sqrt(y_mse)
rmse = math.sqrt(mse)
print("x mse: ", x_mse)
print("y mse: ", y_mse)
print("mse: ", mse)

amount = list(range(0,len(y_pred)))
# print(amount)
# print(y_pred)
# print(len(y_test))
fig, ax = plt.subplots()


# colors = cm.gist_ncar(np.linspace(0, 1, len(amount)))
# for y, c in zip(amount, colors):
#     ax.scatter(y_pred[y,0], y_pred[y,1], color=c)
#     ax.scatter(y_test[y,0], y_test[y,1], color=c, marker="x")
# plt.title("ELM Test Data")
# plt.xlabel("X coordinate [cm]")
# plt.ylabel("Y coordinate [cm]")
# plt.legend(['Prediction', 'Test data'])
# plt.show()

ax.scatter(y_test[:,0],x_dist_list)
ax.scatter(y_test[:,1],y_dist_list)
# i = 0
# for x, y in zip(y_test[:,0], x_dist_list):
#     i = i+1
#     ax.text(x, y, str(i), color="red", fontsize=12)
# i = 0
# for x, y in zip(y_test[:,1], y_dist_list):
#     i = i+1
#     ax.text(x, y, str(i), color="red", fontsize=12)
plt.title("ELM Error vs Coordinate")
plt.xlabel("Coordinate [cm]")
plt.ylabel("Error [cm]")
plt.legend(['X coordinate', 'Y coordinate'])
plt.show()