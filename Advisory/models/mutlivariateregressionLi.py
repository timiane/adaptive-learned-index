from hyperopt import Trials, fmin, tpe

from models.model import model
import models.MultivariateRegression as mvr
import numpy as np
import math


class MultiVariateRegression(model):
    def __init__(self, data, labels):
        self.model = self.create_model(data, labels)

    def extract_model_parameters(self, model, best):
        new_dict = {}
        for row in best:
            fields = model.__dict__
            new_dict[row] = fields[row][best[row]]

        return new_dict

    def optimize_model(self, model, evals):
        trial = Trials()
        best = fmin(model.create_model, model.space, algo=tpe.suggest, max_evals=evals, trials=trial)
        param = self.extract_model_parameters(model, best)
        return param

    def model_to_string(self, b, c, params):
        function = str(b)
        c = c.T
        for index in range(c.size):
            if index < params['poly']:
                function += "+( " + str(c[index]) + "*pow(x," + str(index + 1) + "))"
            # elif index < params['poly'] + params['log'] - 1:
            #     function += "+(log(" + str(index - params['poly'] + 2) + ",x)*" + str(c[index]) + ")"
            # else:
            #     function += "+(" + str(c[index]) + "*math.cos(x))"

        return function

    def create_model(self, data, labels):
        data = data.reshape(-1, 1)
        reg = mvr.MultivariateRegression(data, labels)
        optimal_params = self.optimize_model(reg, 12)
        reg.create_model(optimal_params, True)
        return self.model_to_string(reg.b, reg.coefs, optimal_params)

    def predict(self, data):
        predictions = []
        for x in data:
            predictions.append(eval(self.model))
        return predictions
