from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from hyperopt import hp, STATUS_OK
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


class MultivariateRegression:
    def __init__(self, data, labels):
        self.coefs = None
        self.b = None
        self.model = None
        self.poly = [2, 4, 6, 8, 10, 14, 18, 22, 30, 35, 40, 50]
        self.log = [1, 2, 3, 4, 5, 6]
        self.cos = [False, True]
        self.space = {
            'poly': hp.choice('poly', self.poly),
            # 'log': hp.choice('log', self.log),
            # 'cos': hp.choice('cos', self.cos)
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

    def _cos(self, data):
        data = data.reshape(data.size, 1)
        cos = np.cos(data)
        return cos

    def add_cos_features(self, degree, data):
        cos = np.apply_along_axis(self._cos, 0, data[:, 0])
        data = np.hstack([data, cos])

        return data

    def create_model(self, params, new=False):
        train_data = self.data
        train_data = self.add_poly_features(params['poly'])
        # train_data = self.add_log_features(params['log'], train_data)
        # if params['cos']:
        #     train_data = self.add_cos_features(params['cos'], train_data)

        model = LinearRegression()
        model.fit(train_data, self.labels)
        mse = mean_squared_error(self.labels, model.predict(train_data))
        score = model.score(train_data, self.labels)
        print(params)
        print(score)
        print(mse)
        if new:
            self.coefs = model.coef_
            self.b = model.intercept_
            self.model = model
            return self.model, self.b, self.coefs
        else:
            return {'loss': mse, 'status': STATUS_OK}  # 1-score because we want to maximize the score
