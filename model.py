import numpy as np
import tools
import os
import matplotlib.pyplot as plt


class Model:
    def __init__(self):
        self.__bias = None
        self.__w = None

    @staticmethod
    def __sigmoid(z):
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def __forward(x, w, bias):
        return Model.__sigmoid(x.dot(w) + bias)

    @staticmethod
    def __classification_rate(t, y):
        return (y == t).mean()

    @staticmethod
    def __cross_entropy(t, y):
        E = 0
        for i in range(0, t.shape[0], 1):
            if t[i] == 1:
                E -= np.log(y[i])
            else:
                E -= np.log(1 - y[i])
        return E

    def predict(self, x):
        return Model.__forward(x, self.__w, self.__bias)

    def fit(self, x, t, lambda_l1, lambda_l2, learning_rate, epochs):
        print("train")

        Y = self.__forward(x, self.__w, self.__bias)
        bias_list = []
        CEE = []

        for i in range(0, epochs, 1):
            print("Epoch: " + str(i))
            Y = Model.__forward(x, self.__w, self.__bias)
            delta = x.T.dot(t - Y) - lambda_l2 * self.__w - lambda_l1 * np.sign(self.__w)  # elastic net
            self.__bias = self.__bias + learning_rate * (t - Y).sum()
            bias_list.append(self.__bias)
            __w = self.__w + learning_rate * delta
            CEE.append(Model.__cross_entropy(t, Y).mean())

        plt.plot(bias_list)
        plt.show()

        print(CEE)
        plt.plot(CEE)
        plt.show()

        print(self.__w.shape)
        print(x.shape)

        if not os.path.exists("results"):
            os.makedirs("results")
        np.savetxt("results/best_weights.csv", self.__w, delimiter=',')

        if not os.path.exists("results"):
            os.makedirs("results")
        file = open("results/bias.csv", "w")
        file.write(str(self.__bias))
        file.close()

        print(Model.__classification_rate(np.round(t), Y))

    def init_random_weights(self, D):
        self.__w = np.random.randn(D, 1)
        self.__bias = 0

    def init_weights(self, w, bias):
        self.__w = w
        self.__bias = bias

    def save(self, name):
        tools.serialize(self, name)

    def load(self, name):
        tools.serialize(self, name)
