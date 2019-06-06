import functools
import os
import random
import timeit
from glob import glob
import re
import numpy as np
import pandas as pd
from joblib import load
from keras.models import model_from_json
from experiments.MCDA.MCDA import weighted_sum_method
from pympler import asizeof
import Path


def load_data():
    data_frame = pd.read_csv("times.csv", header=None, names=['name', 'training_time'])
    data_frame['size'] = 0
    data_frame['execution_time'] = 0
    data_frame['10-off_precision'] = 0
    data_frame['15-off_precision'] = 0
    data_frame['20-off_precision'] = 0
    return data_frame


def get_nn_model(name):
    name = "nn_models/" + name
    json_file = open(name + "_model" + ".json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(name + "_weights" + ".h5")
    return loaded_model


def model_predict(model):
       model.predict(np.array(random.random()).reshape(1, -1))


def model_predict_ir(model):
        model.predict(np.array(random.random()).flatten())


def model_predict_mvr(model):
        x = random.random()
        eval(model)


def get_dataset_name(name):
    import re
    return re.search('[a-z]+[_]\d+', name).group().replace('_', '')


def model_to_mvr_function(name):
    with open('models/' + name, "r") as file:
        return file.read()


def mvr_predict(model, data_set):
    predictions = []
    for x in data_set:
        predictions.append(eval(model))
    return predictions

def spline_predict(model):
    model(np.array(random.random()).reshape(1, -1))

def extract_times(current_dataset):
    name = current_dataset['name']
    if 'nn_' in name:
        nn_name = name.replace("nn_", "")
        model = get_nn_model(nn_name)
    elif 'mvr_' in name or 'lr_' in name:
        try:
            model = model_to_mvr_function(name)
        except:
            current_dataset['size'] = -1
            return current_dataset
    else:
        model = load('models/' + name)

    current_dataset['size'] = asizeof.asizeof(model)

    data_set = np.loadtxt(Path.BasePath + 'datasets/' + get_dataset_name(name) + ".csv")

    y_real = np.array(range(0, len(data_set)))
    if 'ir_' in name:
        time = timeit.timeit(functools.partial(model_predict_ir, model), number=2)
        predicted_datapoints = model.predict(data_set.flatten())

    elif 'mvr_' in name or 'lr_' in name:
        time = timeit.timeit(functools.partial(model_predict_mvr, model), number=2)
        predicted_datapoints = np.array(mvr_predict(model, data_set))
    elif 'spline_' in name:
        time = timeit.timeit(functools.partial(spline_predict, model), number=2)
        predicted_datapoints = model(data_set)
    elif 'nn_' in name:
        time = timeit.timeit(functools.partial(model_predict, model), number=2)
        predicted_datapoints = model.predict(data_set/data_set.max())
        predicted_datapoints *= y_real.max()
    else:
        time = timeit.timeit(functools.partial(model_predict, model), number=2)
        predicted_datapoints = model.predict(data_set.reshape(-1, 1))

    current_dataset['execution_time'] = time

    difference = y_real.reshape(-1,1) - predicted_datapoints.reshape(-1, 1)
    current_dataset['10-off_precision'] = (len(data_set) - sum(abs(i) > 10 for i in difference)) / len(data_set)
    current_dataset['15-off_precision'] = (len(data_set) - sum(abs(i) > 15 for i in difference)) / len(data_set)
    current_dataset['20-off_precision'] = (len(data_set) - sum(abs(i) > 20 for i in difference)) / len(data_set)
    return current_dataset


def evaluate_time(data_frame):
    for i in range(0, len(data_frame)):
        print('itteration ' + str(i + 1) + " out of " + str(len(data_frame)))
        current_dataset = data_frame.iloc[i,:]
        new_times = extract_times(current_dataset.copy())
        data_frame.iloc[i, :] = new_times
        os.system('cls' if os.name == 'nt' else 'clear')

    data_frame.to_csv('timesComplete.csv', index=None)


def load_data_frame():
    return pd.DataFrame.from_csv('timesComplete.csv')


def load_data_set_names():
    p = re.compile('[a-z]+\d+')
    PATH = Path.BasePath + "datasets"
    EXT = "*.csv"
    names = []
    files = [file for path, subdir, files in os.walk(PATH)
            for file in glob(os.path.join(path, EXT))]

    for file in files:
        temp = p.search(file).group()
        name = re.search("[a-z]+", temp).group()
        number = re.search("\d+", temp).group()
        names.append(name+"_"+number)
    return names


def calculate_aar(data_frame, size_weight, time_weight, accuracy_weigt):
    names = ['nn', 'nnr', 'ir', 'lr', 'mvr', 'spline']
    dictionary = {}
    data_set_names = load_data_set_names()
    data_frame_result = pd.DataFrame(columns=names, index=data_set_names)
    i = 0
    for name in data_set_names:
        i += 1
        data_frame_benifitial = pd.DataFrame()
        data_frame_non_benifitial = pd.DataFrame()
        dictionary['nn'] = data_frame.loc['nn_' + name, :]
        dictionary['nnr'] = data_frame.loc['nnr_' + name, :]
        dictionary['ir'] = data_frame.loc['ir_' + name, :]
        dictionary['lr'] = data_frame.loc['lr_' + name, :]
        dictionary['mvr'] = data_frame.loc['mvr_' + name, :]
        dictionary['spline'] = data_frame.loc['spline_' + name, :]
        data_frame_temp = pd.DataFrame(dictionary).transpose()

        data_frame_benifitial = data_frame_benifitial.append(data_frame_temp['10-off_precision'])
        data_frame_benifitial = data_frame_benifitial.transpose()
        data_frame_non_benifitial = data_frame_non_benifitial.append(data_frame_temp['size'])
        data_frame_non_benifitial = data_frame_non_benifitial.append(data_frame_temp['execution_time'])
        data_frame_non_benifitial = data_frame_non_benifitial.transpose()
        weights = [size_weight, time_weight, accuracy_weigt]

        data_frame_result.loc[name] = weighted_sum_method(data_frame_benifitial, data_frame_non_benifitial, weights)
    data_frame_result.to_csv('experiments/MCDA/MCDA_test' + '[' + str(size_weight) + ',' + str(time_weight) + ',' + str(accuracy_weigt) + ']')
    return data_frame_result


def print_boxplot(size, time, accuracy):
    aar = pd.DataFrame.from_csv('experiments/MCDA/MCDA_test[' + str(size) + ',' + str(time) + ',' + str(accuracy) + ']')
    import matplotlib.pyplot as pyplot
    aar.boxplot(showbox=True, grid=False, showfliers=False)

    pyplot.show()

def evaluateModels(use_old_eval):
    print("Evaluating models")
    data_frame = load_data()
    if not use_old_eval:
        evaluate_time(data_frame)

    # print_boxplot(0.4,0.2,0.4)
    loaded_data_frame = load_data_frame()
    calculate_aar(loaded_data_frame, 0.33, 0.33, 0.33)
    calculate_aar(loaded_data_frame, 0.1, 0.1, 0.8)
    calculate_aar(loaded_data_frame, 0.1, 0.8, 0.1)
    calculate_aar(loaded_data_frame, 0.8, 0.1, 0.1)
    calculate_aar(loaded_data_frame, 0.5, 0.25, 0.25)
    calculate_aar(loaded_data_frame, 0.25, 0.5, 0.25)
    calculate_aar(loaded_data_frame, 0.25, 0.25, 0.5)
    print('Models evaluated')

