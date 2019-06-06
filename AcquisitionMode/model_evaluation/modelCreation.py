from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt
import numpy as np
from joblib import dump
from sklearn.linear_model import LinearRegression
from sklearn.isotonic import IsotonicRegression
from glob import glob
import os
from scipy.interpolate import InterpolatedUnivariateSpline
import external.MultiVariateRegression as mvr
from hyperopt import fmin, tpe, Trials
import external.nn_model as nn
import json
import functools
import csv
import timeit
import math
import Path

time_table = []


def plot_result_vs_actual(data, predicted):
    y = np.array(range(0, len(data)))
    plt.plot(data, np.array(y), 'r', data, predicted, 'b')
    plt.show()


def model_execution(model, data, y):
    model.fit(data, y)

def create_nnr(data, data_set_type, number):
    y = np.array(range(0, len(data)))
    data = data.reshape(-1, 1)
    neigh = KNeighborsRegressor(n_neighbors=2, weights='distance')
    time = timeit.timeit(functools.partial(model_execution, neigh, data, y), number=1)
    time_table.append(["nnr_" + data_set_type + "_" + str(number), str(time)])
    dump(neigh, 'models/nnr_' + data_set_type + "_" + str(number))

def spline_test(data, y):
    InterpolatedUnivariateSpline(data, y)

def create_spline(data, data_set_type, number):
    y = np.array(range(0, len(data)))
    data = data.reshape(-1, 1)
    ius = InterpolatedUnivariateSpline(data, y)
    time = timeit.timeit(functools.partial(spline_test, data, y), number=20)
    time_table.append(["spline_" + data_set_type + "_" + str(number), str(time)])
    dump(ius, 'models/spline_' + data_set_type + "_" + str(number))

def isotonic_regression(data, data_set_type, number):
    y = np.array(range(0, len(data)))
    input = np.array(data).flatten()
    ir = IsotonicRegression()
    ir.fit(input, y)
    time = timeit.timeit(functools.partial(model_execution, ir, data, y), number=20)
    time_table.append(["ir_" + data_set_type + "_" + str(number), str(time)])
    dump(ir, 'models/ir_' + data_set_type + "_" + str(number))


def linear_regression(data, data_set_type, number):
    y = np.array(range(0, len(data)))
    data = data.reshape(-1, 1)
    lr = LinearRegression().fit(data, y)
    time = timeit.timeit(functools.partial(model_execution, lr, data, y), number=20)
    time_table.append(["lr_" + data_set_type + "_" + str(number), str(time)])
    lr_model_to_string(lr.intercept_, lr.coef_[0], data_set_type, str(number))
    # dump(lr, 'models/lr_' + data_set_type + "_" + str(number))


def extract_model_parameters(model, best):
    new_dict = {}
    for row in best:
        fields = model.__dict__
        new_dict[row] = fields[row][best[row]]

    return new_dict


def optimize_model(model, evals):
    trial = Trials()
    best = fmin(model.create_model, model.space, algo=tpe.suggest, max_evals=evals, trials=trial)
    param = extract_model_parameters(model, best)
    return param

def mr_create(reg, optimal_params):
    reg.create_model(optimal_params, True)

def model_to_string(b, c, params, data_set_type, number, data):
    function = str(b)
    c = c.T
    for index in range(c.size):
        if index < params['poly']:
            function += "+( " + str(c[index]) + "*pow(x," + str(index + 1) + "))"
        # elif index < params['poly'] + params['log'] - 1:
        #     function += "+(log(" + str(index - params['poly'] + 2) + ",x)*" + str(c[index]) + ")"
        else:
            function += "+(" + str(c[index]) + "*math.cos(x))"

    text_file = open('models/mvr_' + data_set_type + "_" + str(number), "w")
    text_file.write(function)
    text_file.close()

def lr_model_to_string(b, c, data_set_type, number):
    function = str(c) + "* x + " + str(b)
    text_file = open('models/lr_' + data_set_type + "_" + str(number), "w")
    text_file.write(function)
    text_file.close()

def multiple_regression(data, data_set_type, number):
    labels = np.array(range(0, len(data)))
    data = data.reshape(-1, 1)
    reg = mvr.MultiVariateRegression(data, labels)
    optimal_params = optimize_model(reg, 6)

    time = timeit.timeit(functools.partial(mr_create, reg, optimal_params), number=20)
    time_table.append(["mvr_" + data_set_type + "_" + str(number), str(time)])

    model_to_string(reg.b, reg.coefs, optimal_params, data_set_type, number, data)

def normalize_max(data):
        return data / data.max()

def create_neural_network(training_data, data_set_type, number):

    labels = np.array(range(0, len(training_data)))
    training_data = training_data.reshape(-1, 1)
    test_data = training_data.copy()
    training_data = np.hstack((training_data, labels.reshape(-1, 1)))
    neural_network = nn.NeuralNetwork(training_data)
    params = optimize_model(neural_network, 4)

    json_best = json.dumps(params)
    f = open("nn_models_config/nn_" + data_set_type + "_" + str(number) + "config" + '.json', 'w')
    f.write(json_best)
    f.close()
    time = timeit.timeit(functools.partial(mr_create, neural_network, params), number=1)
    time_table.append(["nn_" + data_set_type + "_" + str(number), str(time)])
    model_json = neural_network.model.to_json()
    with open("nn_models/" + data_set_type + "_" + str(number) + "_model" + ".json", "w") as json_file:
        json_file.write(model_json)
    neural_network.model.save_weights("nn_models/" + data_set_type + "_" + str(number) + "_weights" + ".h5")

    pred_data = test_data / neural_network.key_normalizer
    prediction = neural_network.model.predict(pred_data).flatten()
    prediction *= neural_network.label_normalizer

    return neural_network


def load_dataset():
    PATH = Path.BasePath + "datasets"

    EXT = "*.csv"
    return [file for path, subdir, files in os.walk(PATH)
            for file in glob(os.path.join(path, EXT))]

def createModels():
    import re
    datasets = load_dataset()
    p = re.compile('[a-z]+\d+')
    iteration = 1
    for dataset in datasets:

        temp = p.search(dataset).group()
        name = re.search("[a-z]+", temp).group()
        number = int(re.search("\d+", temp).group())

        data = np.loadtxt(dataset)
        linear_regression(data, name, number)
        create_spline(data, name, number)
        create_nnr(data, name, number)
        isotonic_regression(data, name, number)
        multiple_regression(data, name, number)
        create_neural_network(data, name, number)

        print("iteration " + str(iteration) + " out of " + str(len(datasets)))
        iteration += 1

    with open("times.csv", 'a') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows(time_table)

    csvFile.close()

