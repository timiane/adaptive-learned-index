import random
import numpy as np
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




def writeDataToCsv(data_number, data, id):
    print(str(data_number))
    with open("data2/" + id + str(data_number) + ".csv", mode='w') as writeFile:
        for i in data:
            writeFile.write(str(i) + "\n")


start_range_big = 150000
end_range_big = 200000
start_range_small = 300
end_range_small = 600
number_of_big_samples = 20
number_of_small_samples = 100

# https://docs.scipy.org/doc/numpy-1.10.0/reference/routines.random.html
def createDatasets():
    number = 1

    for i in range(1, number_of_big_samples + 1):
        y = np.random.beta(2, 2, random.randint(start_range_big, end_range_big))
        y = np.sort(y)
        writeDataToCsv(number, y, "beta")
        number += 1
    for i in range(1, number_of_small_samples + 1):
        y = np.random.beta(2, 2, random.randint(start_range_small, end_range_small))
        y = np.sort(y)
        writeDataToCsv(number, y, "beta")
        number += 1

    number = 1
    for i in range(1, number_of_big_samples + 1):
        y = np.random.wald(0.2, 3, random.randint(start_range_big, end_range_big))
        y = np.sort(y)
        y = np.unique(y)
        writeDataToCsv(i, y, "wald")
        number += 1
    for i in range(1, number_of_small_samples + 1):
        y = np.random.wald(0.2, 3, random.randint(start_range_small, end_range_small))
        y = np.sort(y)
        y = np.unique(y)
        writeDataToCsv(number, y, "wald")
        number += 1

    number = 1
    for i in range(1, number_of_big_samples + 1):
        y = np.random.pareto(3, random.randint(start_range_big, end_range_big))
        y = np.unique(y)
        y = np.sort(y)
        writeDataToCsv(number, y, "parteo")
        number += 1
    for i in range(1, number_of_small_samples + 1):
        y = np.random.pareto(3, random.randint(start_range_small, end_range_small))
        y = np.unique(y)
        y = np.sort(y)
        writeDataToCsv(number, y, "parteo")
        number += 1


    number = 1
    for i in range(1, number_of_big_samples + 1):
        y = np.random.lognormal(3, 1, random.randint(start_range_big, end_range_big))
        y = np.sort(y)
        y = np.unique(y)
        writeDataToCsv(number, y, "logn")
        number += 1
    for i in range(1, number_of_small_samples + 1):
        y = np.random.lognormal(3, 1, random.randint(start_range_small, end_range_small))
        y = np.sort(y)
        y = np.unique(y)
        writeDataToCsv(number, y, "logn")
        number += 1

    number = 1
    for i in range(1, number_of_big_samples + 1):
        y = np.random.uniform(0, 1, random.randint(start_range_big, end_range_big))
        y = np.sort(y)
        writeDataToCsv(number, y, "uni")
        number += 1
    for i in range(1, number_of_small_samples + 1):
        y = np.random.uniform(0, 1, random.randint(start_range_small, end_range_small))
        y = np.sort(y)
        writeDataToCsv(number, y, "uni")
        number += 1

    number = 1
    for i in range(1, number_of_big_samples + 1):
        #https://en.wikipedia.org/wiki/Logistic_distribution
        y = np.random.logistic(6, 2, random.randint(start_range_big, end_range_big))
        y = np.sort(y)
        writeDataToCsv(number, y, "logis")
        number += 1
    for i in range(1, number_of_small_samples + 1):
        #https://en.wikipedia.org/wiki/Logistic_distribution
        y = np.random.logistic(6, 2, random.randint(start_range_small, end_range_small))
        y = np.sort(y)
        writeDataToCsv(number, y, "logis")
        number += 1
