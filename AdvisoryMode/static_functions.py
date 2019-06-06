import numpy as np
import json
import matplotlib.pyplot as plt


def normalize_max(data):
    return data / data.max()


def save_dict(file, path):
    json_best = json.dumps(file)
    f = open(path + '.json', 'w')
    f.write(json_best)
    f.close()


def load_dict(path):
    with open(path + '.json') as f:
        data = json.load(f)
    return data


def get_min_max_error(predictions, index):
    predictions = predictions.reshape(-1, 1)
    true_index = np.arange(predictions.size).reshape(-1,1)
    error = true_index - predictions
    max_error = error.max()
    min_error = error.min()
    average_error = 0
    for error_element in error:
        average_error += abs(error_element)

    if average_error[0] != 0:
        average_error = average_error[0] / len(error)
    else:
        average_error = average_error[0]


    # print("max_error = " + str(max_error) + " and min error = " + str(min_error) + " average error:" + str(average_error))
    return [max_error, min_error, average_error]
def plot_results_vs_actual(keys, prediction):

    keys = np.array(keys).reshape(-1, 1)
    prediction = np.array(prediction).reshape(-1,1)
    actual_pos = np.arange(keys.size)
    predicted_pos = np.array(prediction).flatten()
    plt.plot(keys, predicted_pos, 'r', keys, actual_pos, 'b')
    plt.show()
    print("done")
