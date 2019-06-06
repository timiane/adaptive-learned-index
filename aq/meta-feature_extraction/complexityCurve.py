from sklearn.decomposition import FastICA
import numpy as np
from scipy.stats import gaussian_kde
from scipy.integrate import simps, trapz
import matplotlib.pyplot as plt
import pandas
import os
from glob import glob
import math


def reduce_dataset_recursive(dataset, new_size):
    element_remove_count = len(dataset) - new_size
    dividend = int(len(dataset) / element_remove_count)

    if dividend < 2:
        temp_size = len(dataset) / 2
        subset = sample_reduction(dataset, temp_size)
        return reduce_dataset_recursive(subset, new_size)
    else:
        return sample_reduction(dataset, dividend)


def reduce_dataset_iteration(dataset, new_size, max_half_reductios):  # Recursive
    for reduction_count in range(0, max_half_reductios):
        element_remove_count = len(dataset) - new_size
        if element_remove_count <= 0:
            random = np.random.choice(dataset)
            np.delete(dataset, random)
            return dataset

        dividend_flaot = len(dataset) / element_remove_count
        dividend = math.ceil(dividend_flaot)

        if new_size is dataset.shape[0]:
            return dataset
        else:
            dataset = sample_reduction(dataset, dividend)

    return dataset


def sample_reduction(dataset, dividend):
    data_index = np.arange(0, len(dataset))
    temp = []
    last = False

    for index, j in enumerate(dataset):
        if index is dividend or index % dividend is 0 and index is not 0:
            subset = data_index[index - dividend:index]
        elif len(dataset) - index < dividend and last is False:
            subset = data_index[index:len(dataset) + 1]
            last = True
        else:
            continue
        to_delete = np.random.choice(subset)
        temp.append(to_delete)

    return np.delete(dataset, temp)

def calculate_normalized_auc(y, x):
    """
    Calculates area under the curve normalized to [0,1].
    """
    area = trapz(y, x=x)
    return float(area) / x[-1]


def hl_distances_from_set(A_list, data, points=65, margin_factor=0.25, bw=None):
    """
    Calculates Hellinger distances of A_list sets from the set B using
    continuous formula.
    """
    if bw is None:
        bw = data.shape[0] ** (-1.0 / 5) * 0.5
    yBs = []
    xs = []

    for j in range(data.shape[1]):
        minx, maxx = data[:, j].min(), data[:, j].max()
        margin = (maxx - minx) * margin_factor
        minx -= margin
        maxx += margin
        xs.append(np.linspace(minx, maxx, points))
        try:
            xs_val = xs[-1]
            kde = gaussian_kde(data[:, j], bw_method=bw)
            yBs.append(kde(xs_val))  # PDF of column J in data.
        except (np.linalg.linalg.LinAlgError, ValueError) as _:
            print("Singular matrix -- unable to perform gaussian KDE.")
            yBs.append(np.zeros(xs[-1].shape))

    for A in A_list:
        if A.shape[0] < 2:
            yield 1.0
        else:
            integral = 1
            for j, yB, x in zip(range(len(yBs)), yBs, xs):  # yB and x are rows in yBs and xs respectively
                try:
                    pdf_of_column_x = gaussian_kde(A[:, j], bw_method=bw)(x)
                    y = (np.sqrt(pdf_of_column_x) - np.sqrt(yB))**2
                    integral *= (1 - 0.5 * simps(y, dx=(x[1] - x[0])))
                    del x, yB
                except np.linalg.linalg.LinAlgError:
                    integral = 0.0
            yield 1 - integral


def complexity_curve(X, points=10000, k=100, bw=None, **kwargs):
    """Calculates complexity curve for a given data set.
    Args:
        X (numpy array): Matrix of data points.
        points (Optional[int]): Number of points to probe along the curve.
        k (Optional[int]): Number of subsets to draw in each point.
        use_ica (Optional[bool]): Whether to preprocess data with ICA first.
    Returns:
        Numpy array of shape (points, 3) with the following columns:
        subset size, mean Hellinger distance value, standard deviation of value.
    """
    sets = []
    l = np.linspace(1, X.shape[0], points)  # Creates a evenly spaced number across an interval. Starts from 1 to X.shape[0] with points as the number of elements
    for i in l:
        for j in range(k):
            index = np.random.choice(range(X.shape[0]), int(i), replace=False)
            sets.append(X[index])

    hellinger = list(hl_distances_from_set(sets, X, bw=bw))
    distances = np.reshape(hellinger, (int(len(sets) / k), k))
    m = distances.mean(axis=1)
    s = distances.std(axis=1)

    l = np.hstack((0, l))
    m = np.hstack((1, m))
    s = np.hstack((0, s))
    return np.array([l, m, s]).T


def calculate_data_complexity(data, points, k, subset_column_number):
    data = data.reshape(int(data.size / subset_column_number), subset_column_number)  # The last 10 indicates the number of columns in the subsets
    cc = complexity_curve(data, points=points, k=k)
    error = cc[:, 1] + cc[:, 2]
    auc = calculate_normalized_auc(error, cc[:, 0])
    return auc, error, cc[:, 0]

def construct_complexityCurve(metaDataset):
    PATH = "C:/Users/Timian/Documents/Code/Acquisition/datasets"
    EXT = "*.csv"
    aucList = []
    i = 1
    all_csv_files = [file
                     for path, subdir, files in os.walk(PATH)
                     for file in glob(os.path.join(path, EXT))]

    for file in all_csv_files:
        print(i)
        dataset = np.loadtxt(file)
        if(len(dataset) > 100.000):
            dataset = reduce_dataset_iteration(dataset, 100000, 1000)
        auc = calculate_data_complexity(dataset, 30, 5, (int(dataset.size/dataset.size)))[0]
        aucList.append(auc)
        i = i + 1

    aucSeries = pandas.Series(aucList)
    metaDataset.insert(loc=0, column='cc', value=aucSeries)
    metaDataset.to_csv('metaFeaturesCC.csv', header=False, index=False)





if __name__ == '__main__':
    dataset = pandas.read_csv('metaFeatures.csv', header=None)
    construct_complexityCurve(dataset)
