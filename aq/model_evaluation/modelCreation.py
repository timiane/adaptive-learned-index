from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt
import numpy as np
import external.kernel_regression as kn
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


def create_kernel(data, data_set_type, number):
    y = np.array(range(0, len(data)))
    data = data.reshape(-1, 1)
    kernel = kn.KernelRegression(gamma=10)
    kernel.fit(data, y)
    time = timeit.timeit(functools.partial(model_execution, kernel, data, y), number=20)
    time_table.append(["kernel_" + data_set_type + "_" + str(number), str(time)])
    dump(kernel, 'models/kernel_' + data_set_type + "_" + str(number))


# def create_lasso(data, data_set_type, number):
#     from sklearn.linear_model import Lasso
#     y = np.array(range(0, len(data)))
#     data = data.reshape(-1, 1)
#     lasso = Lasso()
#     lasso.fit(data, y)
#     time = timeit.timeit(functools.partial(model_execution, lasso, data, y), number=20)
#     time_table.append(["lasso_" + data_set_type + "_" + str(number), str(time)])
#     dump(lasso, 'models/lasso_' + data_set_type + "_" + str(number))

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
    PATH = "C:/Users/Timian/Documents/Code/Acquisition/datasets"
    EXT = "*.csv"
    return [file for path, subdir, files in os.walk(PATH)
            for file in glob(os.path.join(path, EXT))]

def plot_knn():
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn import neighbors

    np.random.seed(0)
    X = np.array([0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9, 1.4]).reshape(-1,1)
    T = np.linspace(0.1, 1.4, 100)[:, np.newaxis]
    y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]).reshape(-1,1)

    # Add noise to targets
    # y[::5] += 1 * (0.5 - np.random.rand(8))

    # #############################################################################
    # Fit regression model
    n_neighbors = 2

    for i, weights in enumerate(['distance']):
        knn = neighbors.KNeighborsRegressor(n_neighbors, weights=weights)
        y_ = knn.fit(X, y).predict(T)

        plt.scatter(X, y, c='k', label='data')
        plt.scatter(0.4, 5, c='r', label='prediction point')
        plt.plot(T, y_, c='g', label='prediction')
        plt.axis('tight')
        plt.legend()

    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

def plot_spline():
    from scipy import interpolate
    x = np.array([1, 2, 4, 5])  # sort data points by increasing x value
    y = np.array([2, 2.5, 4, 4.5])
    arr = np.arange(np.amin(x), np.amax(x), 0.01)
    s = interpolate.CubicSpline(x, y)

    fig, ax = plt.subplots(1, 1)
    plt.plot(x, y, 'bo', label='Data Point')
    plt.plot(arr, s(arr), 'k-', label='Polynomial Spline', lw=1)
    k = 3
    # knots = s.get_knots()
    # coefs = s.get_coeffs()
    y_eval = s(arr)
    for i in range(x.shape[0] - 1):
        segment_x = np.linspace(x[i], x[i + 1], 100)
        # A (4, 100) array, where the rows contain (x-x[i])**3, (x-x[i])**2 etc.
        exp_x = (segment_x - x[i])[None, :] ** np.arange(4)[::-1, None]
        # segment_y = coefs.dot(exp_x)

        # Sum over the rows of exp_x weighted by coefficients in the ith column of s.c
        coefs_row = s.c[:, i]
        segment_y = s.c[:, i].dot(exp_x)
        plt.plot(segment_x, segment_y, label='Segment {}'.format(i), ls='--', lw=3)

    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()



if __name__ == '__main__':
    import re
    datasets = load_dataset()
    p = re.compile('[a-z]+\d+')
    iteration = 1
    for dataset in datasets:

        temp = p.search(dataset).group()
        name = re.search("[a-z]+", temp).group()
        number = int(re.search("\d+", temp).group())
        if 21 < number < 120:
            # time_table.append(["nn_" + name + "_" + str(number), str(-1)])
            # print("iteration " + str(iteration) + " out of " + str(len(datasets)))
            # iteration += 1
            continue

        data = np.loadtxt(dataset)
        # linear_regression(data, name, number)
        # create_spline(data, name, number)
        # create_nnr(data, name, number)
        # create_kernel(data, name, number)
        # create_lasso(data, name, number)
        # isotonic_regression(data, name, number)
        # multiple_regression(data, name, number)
        create_neural_network(data, name, number)

        print("iteration " + str(iteration) + " out of " + str(len(datasets)))
        iteration += 1

    with open("timesnn2.csv", 'a') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows(time_table)

    csvFile.close()
    # dump(isotonic_model, 'models/iso_logis1')
    # nn_model = load('models/iso_logis1')
    # nn_pred = nn_model.predict(np.array(dataframe).flatten())
    # plot_result_vs_actual(dataframe, nn_pred)
    # print(nn_pred)

#def splineInterpolation(data):
#
#    input = np.array(sorted(np.array(data).flatten()))
#    x = np.array([0, 1, 1, 2, 3, 4])
#    y = range(0, len(x))
#    print(x.shape, input.shape)
#    output = np.array([ 0.,     0.308,  0.55,   0.546,  0.44 ])
#    spline = interp1d(x, y, kind='cubic', bounds_error=False)
#    f2 = interp1d(x, -y, kind='cubic')
#    xmax = fmin(f2, 2.5)
#    xfit = np.linspace(0, len(x))
#    plt.plot(xfit, spline(xfit), 'r-')
#    plt.plot(xmax, spline(xmax), 'g*')
#    plt.show()


if __name__ == '__main__':
    print("hej")
