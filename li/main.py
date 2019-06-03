import models.NeuralNetwork as nn
import models.MultivariateRegression as mvr
import numpy as np
from hyperopt import fmin, tpe, Trials
import time
import keras
from meta_feature_extraction import extract_meta_features as me
from stages.KerasModelStage import KerasModelStage
from stages.LinearModelStage import LinearModelStage
from stages.Stage import Stage
from stages.kerasify import export_model
import static_functions as sf
from models.spline import Spline
from models.isotonicregressor import IsotonicRegressor
from models.mutlivariateregressionLi import MultiVariateRegression
from models.nearest_neighbor_regression import NearestNeighborRegression
from models.linear import Linear

MODEL_PREDICTOR = 0

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
        result = stage.load_stage(directory_path, i)
        layers.append(stage)
    stage = Stage(length)
    result = stage.load_stage(directory_path, length - 1)
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


def create_neural_network(training_data, index):
    neural_network = nn.NeuralNetwork(training_data)
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


def determine_current_stage_type(current_index, stage_count):  # determine if linear or keras layer
    if current_index == stage_count - 1:
        return LinearModelStage(current_index, [])
    else:
        return KerasModelStage(current_index, [])


def build_or_append_model_pred(model_pred, temp_pred):
    if model_pred is None:  # start or append predictions
        return temp_pred
    else:
        return np.concatenate((model_pred, temp_pred))


def build_multi_staged_model(data_set, models_pr_stage):  # build multi staged model
    size = data_set.size
    stages_count = len(models_pr_stage)
    key_index_model_array = distribute_next_layer(data_set, size, [np.zeros(size)])
    stages = []

    for stage_index in range(stages_count):
        model_pred = None
        # stage = determine_current_stage_type(stage_index, stages_count)
        stage = Stage(stage_index, np.zeros(stages_count))
        for model_index in range(models_pr_stage[stage_index]):

            temp_data = get_model_data(key_index_model_array, model_index, models_pr_stage[stage_index])
            data_to_test = temp_data[:, 0]
            index = temp_data[:, 1]
            meta_features = me(data_to_test).values
            meta_features = np.append(meta_features, [[0.1, 0.1, 0.8]]).reshape(1, 11)
            selected_model = MODEL_PREDICTOR.predict(meta_features)[0]

            if selected_model[0] == selected_model.max():
                model = IsotonicRegressor(data_to_test, index)

            elif selected_model[1] == selected_model.max():
                model = Spline(data_to_test, index)
            elif selected_model[2] == selected_model.max():
                model = Linear(data_to_test, index)



            if models_pr_stage[stage_index] == models_pr_stage[-1]:  # check if last stage
                stage.models = np.append(stage.models, model)
            else:
                stage.models = np.append(stage.models, model)
                # sf.plot_results_vs_actual(neural_network.model.predict(temp_data[:, 0]), temp_data[:, 1])
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
    print('prediction took ' + str(prediction_time) + 'nanoseconds for a data set of size ' + str(keys.size))
    return prediction


def export_models(stages, file_path):
    for stage_index, stage in enumerate(stages):
        if type(stage) == KerasModelStage:
            for model_index, model in enumerate(stage.models):
                path = file_path + "export/" + str(stage_index) + "." + str(model_index) + ".model"
                export_model(model.model, path)
        else:
            print("do something with linear stages")


def build_and_predict(keys, file_path):
    if not USE_OLD_MODELS:
        print("building stages")
        layers = build_multi_staged_model(keys, MODEL_PR_LAYER)
        print("saving stages")
        save_models(layers, file_path)
    else:
        print("loading stages")
        layers = load_models(file_path)
    # export_models(layers, file_path)
    print("predicting index")
    prediction = predict_index(layers, keys)
    print("error")
    sf.get_min_max_error(prediction, np.array(range(0, len(prediction))))
    print("plotting stages")
    sf.plot_results_vs_actual(keys, prediction)


def load_data(name):
    result = {
        # 'log_normal': np.loadtxt('data/norm_dist/log.csv'),  # np.sort(np.random.lognormal(3., 1., 2000000)),
        # 'norm': np.loadtxt('data/norm_dist/sorted.txt'),
        # 'sas': np.loadtxt('data/sas.csv'),
        'osm': np.sort(np.loadtxt('data/osm.csv')),
        # 'norm_short': np.loadtxt('data/norm_dist/sorted.txt'),
        # 'sas_short': np.loadtxt('data/sas.csv'),
        # 'osm_short': np.sort(np.loadtxt('data/osm.csv')),
        'beta22': np.sort(np.loadtxt('data/beta22.csv')),
        # 'beta2': np.sort(np.loadtxt('data/beta2.csv')),
        # 'beta3': np.sort(np.loadtxt('data/beta3.csv')),
        # 'beta4': np.sort(np.loadtxt('data/beta4.csv')),
        # 'beta5': np.sort(np.loadtxt('data/beta5.csv')),
        # 'beta6': np.sort(np.loadtxt('data/beta6.csv')),
        # 'beta7': np.sort(np.loadtxt('data/beta7.csv')),
        # 'beta8': np.sort(np.loadtxt('data/beta8.csv'))
    }

    return result.get(name, 'unknown'), 'stages/' + name + '/'


def multiple_regression(data, name):
    labels = np.arange(data.size).reshape(-1, 1)
    data = data.reshape(-1, 1)
    reg = mvr.MultivariateRegression(data, labels)
    optimal_params = optimize_model(reg, 15)
    optimal_model, b, c = reg.create_model(optimal_params, True)
    model_to_string(b, c, optimal_params, name)


if __name__ == '__main__':
    MODEL_PR_LAYER = [1, 1000]  # short = [1,1000] others = [1,10,1000]
    USE_OLD_MODELS = True  # True if use previously created stages else False
    json_file = open('C:/Users/Timian/Documents/Code/ARR/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    MODEL_PREDICTOR = keras.models.model_from_json(loaded_model_json)
    MODEL_PREDICTOR.load_weights('C:/Users/Timian/Documents/Code/ARR/weights.h5')
    data, path = load_data('osm')
    # data = np.array([1,2,3,4,5,6])
    # path = ""
    # multiple_regression(data, str(sys.argv[1]))
    build_and_predict(data, path)
