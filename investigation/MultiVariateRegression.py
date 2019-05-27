from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import math
import numpy as np
import matplotlib.pyplot as plt


class MultiVariateRegression:
    def __init__(self, data, labels):
        self.coefs = None
        self.b = None
        self.model = None
        self.poly = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.log = [1, 2, 3, 4, 5, 6]
        self.space = {
            'poly': hp.choice('poly', self.poly),
            'log': hp.choice('log', self.log),
        }
        self.data = data
        self.labels = labels

    def add_poly_features(self, degree):
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        return poly.fit_transform(self.data)

    def add_log_features(self, base, data):
        for i in range(2, base + 1):
            log = np.log(self.data[:, 0]) / np.log(i)
            data = np.hstack([data, log.reshape(self.data[:, 0].size, 1)])

        return data

    def train(self, params, new=False):
        train_data = self.data
        train_data = self.add_poly_features(params['poly'])
        train_data = self.add_log_features(params['log'], train_data)

        model = LinearRegression()
        model.fit(train_data, self.labels)
        score = model.score(train_data, self.labels)

        if new:
            # predictions = model.predict(train_data)
            # plt.plot(train_data[:, 0], predictions, 'r', train_data[:, 0], self.labels, 'b')
            # plt.show()
            c = model.coef_
            b = model.predict(np.zeros(shape=(1, train_data.shape[1])))

            return model, b, c
        else:
            return {'loss': 1 - score, 'status': STATUS_OK}  # 1-score because we want to maximize the score
