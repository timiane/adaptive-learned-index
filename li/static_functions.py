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
    true_index = np.arange(predictions.size)
    max_error = 0
    min_error = predictions.size
    for i in range(predictions.size):
        error = abs(true_index[i] - predictions[i])[0].astype(int)
        if error < min_error:
            min_error = error
        if error > max_error:
            max_error = error
    print("max_error = " + str(max_error) + " and min error = " + str(min_error) + " for stage " + str(index))

def plot_results_vs_actual(keys, prediction):
    actual_pos = np.arange(keys.size)
    predicted_pos = np.array(prediction).flatten()
    plt.plot(keys, predicted_pos, 'r', keys, actual_pos, 'b')
    plt.show()
    print("done")
