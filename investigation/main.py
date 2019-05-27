from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
import matplotlib.pyplot as plt
import time
import featuretools as ft
import pandas as pd
from hyperopt import fmin, tpe, Trials
import sys
import math
import MultiVariateRegression as MVR


def extract_model_parameters(model, best):
    new_dict = {}
    for row in best:
        fields = model.__dict__
        new_dict[row] = fields[row][best[row]]

    return new_dict


def optimize_model(model):
    trial = Trials()
    best = fmin(model.train, model.space, algo=tpe.suggest, max_evals=50, trials=trial)
    params = extract_model_parameters(model, best)

    return params


def feature_tool_test(data):
    es = ft.EntitySet('normal_distribution')
    df = pd.DataFrame(data, columns=['key'])
    df['pos'] = df.index
    es = es.entity_from_dataframe(dataframe=df, entity_id='log')
    es = es.normalize_entity(base_entity_id='log', new_entity_id='pos', index='pos')
    fm, features = ft.dfs(entityset=es, target_entity='log')


if __name__ == '__main__':
    #osm = np.sort(np.genfromtxt('osm.csv', delimiter='\n')).reshape(-1,1)
    norm = np.sort(np.random.normal(100, 25, size=10000)).reshape(-1,1)
    print('Data loaded..')
    #lin_reg_poly(norm, 10, range(0, norm.size))
    reg = MVR.MultiVariateRegression(norm, range(0, norm.size))
    optimal_params = optimize_model(reg)
    optimal_model, b, c = reg.train(optimal_params, True)
    function = str(b[0])
    for index in range(optimal_params['poly']):
        function = function +"+( "+ str(c[index])+"*pow(x,"+str(index+1)+"))"
    for logindex in range(optimal_params['log']-1):
        function = function + "+(log("+ str(optimal_params['log']-logindex) +",x)*"+str(c[-(logindex+1)])+")"
    print(function)
    text_file = open("function.txt", "w")
    text_file.write(function)
    text_file.close()
    print('asd')
