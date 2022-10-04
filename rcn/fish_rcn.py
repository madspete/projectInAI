#!/usr/bin/env python
from pyrcn.echo_state_network import ESNRegressor, ESNClassifier
from pyrcn.extreme_learning_machine import ELMRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import Ridge as skRidge
from sklearn.datasets import make_blobs
from pyrcn.datasets import mackey_glass

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import csv
import sys
import math

esn = ESNRegressor()

X, y = mackey_glass(n_timesteps=8000)

# Define Train/Test lengths
trainLen = 4000
X_train, y_train = X[:trainLen], y[:trainLen]
X_test, y_test = X[trainLen:], y[trainLen:]

# Initialize and train an ELMRegressor and an ESNRegressor
esn = ESNRegressor().fit(X=X_train.reshape(-1, 1), y=y_train)
elm = ELMRegressor(regressor=skRidge()).fit(X=X_train.reshape(-1, 1), y=y_train)
print("Fitted models")

y_pred = elm.predict(y_test.reshape(-1, 1))

print(y_pred.shape)
print(X_train.shape)
print(y_train.shape)

amount = list(range(0,trainLen))
fig, ax = plt.subplots()
ax.plot(amount, y_pred, color="blue", marker=" ")
ax.plot(amount, y_test, color="green", marker=" ")
plt.show()