from model import Model
from import_data import read_prepared_data

import numpy as np
from sklearn.utils import shuffle
from tools import read_numpy_array_from_path
from import_data import load_data
from import_data import show_image_with_title
from import_data import label

# HyperParams
rate = 0.001
l1 = 1000
l2 = 1000
epochs = 1000

# TRAIN
X, Y = read_prepared_data()
Y_bool = (Y < 2)
Y_bool = np.reshape(Y_bool, Y_bool.shape[0])

X = X[Y_bool]
Y = Y[Y_bool]

# the disgust samples are not enough
Y_bool = (Y == 1)
Y_bool = np.reshape(Y_bool, Y_bool.shape[0])
X_second = X[Y_bool]
X_second = X_second * 4
Y_second = Y[Y_bool]
Y_second = Y_second * 4


X = np.vstack((X, X_second))
Y = np.vstack((Y, Y_second))
X, Y = shuffle(X, Y)

model = Model()
model.init_random_weights(X.shape[1])
model.fit(X, Y, l1, l2, rate, epochs)



# TEST

X, Y = load_data()

Y_bool = (Y < 2)
Y_bool = np.reshape(Y_bool, Y_bool.shape[0])

X = X[Y_bool]
Y = Y[Y_bool]


def test(IND):
    global model
    TEST_INPUT = X[IND]
    TEST_INPUT = TEST_INPUT / TEST_INPUT.mean()
    W = read_numpy_array_from_path("results/best_weights.csv")
    B = read_numpy_array_from_path("results/bias.csv")
    model = Model()
    model.init_weights(W, B)
    result = np.round(model.predict(TEST_INPUT))
    result = int(result[0][0])
    show_image_with_title(TEST_INPUT, "Exp: " + label[Y[IND]] + " pred: " + label[result])

from random import randint
while True:
    test(randint(0, 5500))
