import models.NeuralNetwork as nn
import models.MultivariateRegression as mvr
import numpy as np
from hyperopt import fmin, tpe, Trials
import time
import keras
import pandas as pd
from meta_feature_extraction.extract_meta_features import extract_meta_features as me
from stages.Stage import Stage
from stages.kerasify import export_model
import static_functions as sf
from models.spline import Spline
from models.isotonicregressor import IsotonicRegressor
from models.mutlivariateregressionLi import MultiVariateRegression
from models.linear import Linear
from pympler import asizeof
import csv
import Path

MODEL_PREDICTOR = 0
USE_OLD_MODELS = False
MODEL_PR_LAYER = []

def model_to_string(b, c, params, data_name):
    function = str(b[0])
    c = c.T
    for index in range(c.size):
        if index < params['poly']:
            function += "+( " + str(c[index]) + "*pow(x," + str(index + 1) + "))"
        elif index < params['poly'] + params['log'] - 1:
            function += "+(log(" + str(index - params['poly'] + 2) + ",x)*" + str(c[index]) + ")"
        else:
            function += "+(" + str(c[index]) + "*cos(x))"
    print(function)
    function = function.replace('[', '').replace(']', '')
    text_file = open("mvr/" + data_name + ".txt", "w")
    text_file.write(function)
    text_file.close()


def save_models(layers, directory_path):
    for layer in layers:
        layer.save_stage(directory_path)


def load_models(directory_path):
    layers = []
    length = len(MODEL_PR_LAYER)
    for i in range(length - 1):
        stage = Stage(i+1)
        stage.load_stage(directory_path, i)
        layers.append(stage)
    stage = Stage(length)
    stage.load_stage(directory_path, length - 1)
    layers.append(stage)
    return layers


def get_model_data(results, model, models_in_layer):
    if model == 0:
        temp_data = results[np.where(results[:, 2] <= model)]
    elif model >= models_in_layer - 1:
        temp_data = results[np.where(results[:, 2] >= model)]
    else:
        temp_data = results[np.where(results[:, 2] == model)]
    return temp_data


def distribute_next_layer(data, size, models_pred):  # distribute data to next layer
    temp_data = np.array(data)
    index = np.arange(0, size)

    results = np.append([temp_data, index], models_pred, axis=0)
    return results.T


def create_neural_network(training_data, index, labels):
    neural_network = nn.NeuralNetwork(training_data, labels)
    if index == -1:
        params = optimize_model(neural_network, 10)
        sf.save_dict(params, "params")
    else:
        params = sf.load_dict("params")
    train_neural_network(neural_network, params)
    return neural_network


def predict_models(model, prediction_data, total_data_size, next_layer_models):
    prediction = model.predict(prediction_data)
    return np.floor_divide(prediction, (total_data_size / next_layer_models))


def build_or_append_model_pred(model_pred, temp_pred):
    if model_pred is None:  # start or append predictions
        return temp_pred
    else:
        return np.concatenate((model_pred, temp_pred))


def build_multi_staged_model(data_set, models_pr_stage, model_to_use, dataset_name, meta_features, weights=[] ):  # build multi staged model
    size = data_set.size
    stages_count = len(models_pr_stage)
    key_index_model_array = distribute_next_layer(data_set, size, [np.zeros(size)])
    stages = []
    for stage_index in range(stages_count):
        model_pred = None
        stage = Stage(stage_index, np.zeros(stages_count))
        for model_index in range(models_pr_stage[stage_index]):

            temp_data = get_model_data(key_index_model_array, model_index, models_pr_stage[stage_index])
            data_to_test = temp_data[:, 0].reshape(-1, 1)
            index = temp_data[:, 1]

            if model_to_use == 'mvr':
                if models_pr_stage[stage_index] != models_pr_stage[-1]:
                    model = MultiVariateRegression(data_to_test, index)
                else:
                    model = Linear(data_to_test, index)

            if model_to_use == 'nn':
                if models_pr_stage[stage_index] != models_pr_stage[-1]:
                    model = create_neural_network(data_to_test, 1, index)
                else:
                    model = Linear(data_to_test, index)


            if model_to_use == 'adaptive':
                if models_pr_stage[stage_index] == models_pr_stage[-1]:
                    meta_features = me(data_to_test).values

                if len(weights) != 0:
                    if weights == [0.1, 0.1, 0.8]:
                        selected = 'accuracy'
                        meta_features_test = np.append(meta_features, weights + ['accuracy'])

                    elif weights == [0.1, 0.8, 0.1]:
                        selected = 'speed'
                        meta_features_test = np.append(meta_features, weights + ['speed'])
                    else:
                        selected = 'size'
                        meta_features_test = np.append(meta_features, weights + ['size'])

                    meta_features= np.append(meta_features, weights).reshape(1, 11)

                else:
                    if models_pr_stage[stage_index] != models_pr_stage[-1]:
                        selected = 'opt'
                        meta_features_test = np.append(meta_features, [0.1, 0.1, 0.8, 'opt'])
                        meta_features = np.append(meta_features, [0.1, 0.1, 0.8]).reshape(1, 11)
                    else:
                        selected = 'opt'
                        meta_features_test = np.append(meta_features, [0.8, 0.1, 0.1, 'opt'])
                        meta_features = np.append(meta_features, [0.8, 0.1, 0.1]).reshape(1, 11)

                selected_model = MODEL_PREDICTOR.predict(meta_features)[0]

                if selected_model[0] == selected_model.max():
                    model = IsotonicRegressor(data_to_test.flatten(), index)
                    with open("experiments/" + dataset_name +  selected + ".csv", 'a+') as myfile:
                        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
                        wr.writerow(np.append(meta_features_test, 'ir'))
                elif selected_model[2] == selected_model.max():
                    data_to_test = np.sort(data_to_test)
                    model = Spline(data_to_test, index)
                    with open("experiments/" + dataset_name + selected + ".csv", 'a+') as myfile:
                        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
                        wr.writerow(np.append(meta_features_test, 'spline'))
                elif selected_model[1] == selected_model.max():
                    model = Linear(data_to_test, index)
                    with open("experiments/" + dataset_name +selected + ".csv", 'a+') as myfile:
                        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
                        wr.writerow(np.append(meta_features_test, 'lr'))



            if models_pr_stage[stage_index] == models_pr_stage[-1]:  # check if last stage
                stage.models = np.append(stage.models, model)
            else:
                stage.models = np.append(stage.models, model)
                temp_models = predict_models(model, temp_data[:, 0], size, models_pr_stage[stage_index + 1])
                model_pred = build_or_append_model_pred(model_pred, temp_models)

        if not models_pr_stage[stage_index] == models_pr_stage[-1]:  # check if last stage
            model_pred = model_pred.reshape(1, -1)
            key_index_model_array = distribute_next_layer(data_set, size, model_pred)
        stage.length = len(stage.models)
        stages.append(stage)
    return stages


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


def train_neural_network(neural_network, params):
    neural_network.create_model(params, True)


def calculate_next_layer_models(reshaped_data, stage, current_stage_data, next_stage_size):
    predictions = stage.predict_next_model(current_stage_data, next_stage_size).reshape(-1,1)
    result = np.append(reshaped_data, predictions, axis=1)
    return result

def predict_index(stages, keys):
    reshaped_data = keys.reshape(keys.size, 1)
    zeros = np.zeros(keys.size).reshape(keys.size, 1)
    transformed_data = np.append(reshaped_data, zeros, axis=1)
    prediction = []
    start_time = time.time()
    for index, stage in enumerate(stages):
        if not stages[index] == stages[len(stages) - 1]:
            next_stage_size = stages[index + 1].length
            transformed_data = calculate_next_layer_models(reshaped_data, stage, transformed_data, next_stage_size)
        else:
            prediction = stage.predict(transformed_data)
    prediction_time = (time.time() - start_time) * 1000000000  # prediction time in nanoseconds
    return prediction, prediction_time


# def export_models(stages, file_path):
#     for stage_index, stage in enumerate(stages):
#
#             for model_index, model in enumerate(stage.models):
#                 path = file_path + "export/" + str(stage_index) + "." + str(model_index) + ".model"
#                 export_model(model.model, path)



def build_and_predict(keys, file_path, model_to_use, meta_features=[], weights=[]):
    if not USE_OLD_MODELS:
        print("building stages")
        layers = build_multi_staged_model(keys, MODEL_PR_LAYER, model_to_use, file_path, meta_features, weights)
        print("saving stages")
    else:
        print("loading stages")
        layers = load_models(file_path)

    size = 0
    print("evaluating size")

    for layer in layers:
        for model in layer.models:
            size += asizeof.asizeof(model.model)
    prediction, time = predict_index(layers, keys)
    errors = sf.get_min_max_error(prediction, np.array(range(0, len(prediction))))
    result = [size, len(keys), time] + errors
    return result


def rmi_build_and_predict(dataset_name, data, model_per_layer, speed_weight, size_weight, accuracy_weight, use_optimal):
    weights = [size_weight, speed_weight, accuracy_weight]

    global MODEL_PR_LAYER
    MODEL_PR_LAYER = model_per_layer

    json_file = open(Path.BasePath + 'Acquisition/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    global MODEL_PREDICTOR
    MODEL_PREDICTOR = keras.models.model_from_json(loaded_model_json)
    MODEL_PREDICTOR.load_weights(Path.BasePath + 'Acquisition/weights.h5')
    dataframe = pd.DataFrame(columns=['size of model', 'cardinality of dataset', 'pred time', 'max error',
                                      'min error', 'avg error', 'dataset', 'setting'])

    meta_features = me(data).values
    if(use_optimal):
        dataframe = dataframe.append(pd.Series(build_and_predict(data, dataset_name, 'adaptive', meta_features,) + [dataset_name, 'adaptiveOptimal'], index=dataframe.columns), ignore_index=True)
    else:
        dataframe = dataframe.append(pd.Series(build_and_predict(data, dataset_name, 'adaptive', meta_features, weights) + [dataset_name, 'userWeights'], index=dataframe.columns), ignore_index=True)

    dataframe.to_csv("experiments/testresults" + str(dataset_name) + ".csv")

