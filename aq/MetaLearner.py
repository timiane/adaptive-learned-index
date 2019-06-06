import keras
from sklearn.preprocessing import LabelEncoder
import pandas
from keras import backend as K
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from keras.wrappers.scikit_learn import KerasClassifier
import sklearn.metrics
from imblearn.pipeline import make_pipeline, Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from keras.models import model_from_json
from sklearn.neural_network import MLPRegressor
import numpy as np
from pympler import asizeof
import matplotlib.pyplot as plt
import collections
from keras import optimizers


osm = np.array([7.29139680e-02, 3.01806293e-14, 1.66165900e+06, 7.46702117e-03, 1.72678373e-02, 1.48537000e+05, 4.98117986e-01, 9.54191297e-01, 0.1, 0.1, 0.8]).reshape(1, 11)


class Metrics(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []


    def on_epoch_end(self, epoch, logs={}):
        val_predict = (np.asarray(self.model.predict(self.validation_data[0]))).round()
        val_targ = self.validation_data[1]
        _val_f1 = sklearn.metrics.f1_score(val_targ, val_predict, average='weighted')
        _val_recall = sklearn.metrics.recall_score(val_targ, val_predict, average='weighted')
        _val_precision = sklearn.metrics.precision_score(val_targ, val_predict, average='weighted')
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        print(' — val_f1: % f — val_precision: % f — val_recall % f' % (_val_f1, _val_precision, _val_recall))

        return


metrics = Metrics()

def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))


def metaLearnerNN():

    model = keras.Sequential()
    model.add(keras.layers.Dense(100, input_dim=11, activation='tanh'))
    model.add(keras.layers.Dense(100, activation='relu'))
    model.add(keras.layers.Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=optimizers.adam(lr=0.001), metrics=['categorical_accuracy'])

    return model


def metaTrain(NN, input, output):
    X = input[:, 0:11].astype(float)
    Y = output[:,7]
    smt = SMOTE()
    #X, Y = smt.fit_sample(X, Y)
    #df = pandas.DataFrame(Y)
    #print(df[0].value_counts())
    encoder = LabelEncoder()
    encoder.fit(Y)
    encoded_Y = encoder.transform(Y)
    labels = keras.utils.to_categorical(encoded_Y)
    seed = 7
    np.random.seed(seed)
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.3, random_state=seed)
    X_train_sm, y_train_sm = smt.fit_sample(X_train, y_train)
    #kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    print(Y)
    print(labels)
    batch = 7000
    callback = keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.000, patience=500, restore_best_weights=True)

    NN.fit(X_train_sm, y_train_sm, validation_data=(X_test, y_test), epochs=10000, batch_size=batch, callbacks=[metrics, callback], shuffle=True)


    #estimator = KerasClassifier(build_fn=metaLearnerNN, epochs=100, batch_size=10000, verbose=1)
    #pipeline = make_pipeline(smt, estimator)
    #results = cross_val_score(estimator, X, encoded_Y, cv=kfold)
    #loss = pipeline.evaluate(X, labels, batch_size=batch)
    #estimator.fit(X_train, y_train)
    prediction = NN.predict_classes(X_test)
    prediction = np.argmax(to_categorical(prediction), axis=1)
    prediction = encoder.inverse_transform(prediction)
    print(collections.Counter(prediction))
    #print("Baseline: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))
    #predictiondf = pandas.DataFrame({'ir':prediction[:, 0], 'lr':prediction[0:, 1], 'spline':prediction[0:, 2]})
    #predictiondf['best'] = predictiondf[['ir', 'spline', 'lr']].idxmax(axis=1)
    #print(predictiondf['best'].value_counts())
    #prediction = predictiondf.values
    #predictionclass = NN.predict_classes(X)
    #f1 = sklearn.metrics.f1_score(Y_test_class, prediction, average='weighted')
    #recall = sklearn.metrics.recall_score(Y_test_class, prediction, average='weighted')
    #precision = sklearn.metrics.precision_score(Y_test_class, prediction, average='weighted')
    scores = NN.evaluate(X_test, y_test)
    print("\n%s: %.2f%%" % (NN.metrics_names[1], scores[1] * 100))
    osmpredict = NN.predict_classes(osm)
    osmpredict = np.argmax(to_categorical(osmpredict), axis=1)
    osmpredict = encoder.inverse_transform(osmpredict)
    print(osmpredict)
    #print("f1 score: " + str(f1))
    #print("recall score: " + str(recall))
    #print("precision score: " + str(precision))
    #print(NN.metrics_names)
    #print(prediction)
    ##print(predictionclass)

    return NN

def explanatoryNN():
    model = keras.Sequential()
    model.add(keras.layers.Dense(18, input_dim=8, activation='relu'))
    #model.add(keras.layers.Dense(18, activation='relu'))
    model.add(keras.layers.Dense(3, activation='relu'))
    model.compile(loss= root_mean_squared_error, optimizer='adam', metrics=['accuracy'])

    return model

def explanatoryTrain(model, input, output):
    X = input[:, 0:9].astype(float)
    labels = output[:, 1:6].astype(float)

    batch = 10
    callback = [
        keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.000, patience=500,
                                      restore_best_weights=True)]

    model.fit(X, labels, epochs=1000, verbose=0, batch_size=batch, callbacks=callback)


def explanatorymodelhandler(input, output):

    nnrDF = pandas.DataFrame()
    # kernelDF = pandas.DataFrame()
    # lassoDF = pandas.DataFrame()
    nnDF = pandas.DataFrame()
    splineDF = pandas.DataFrame()
    irDF = pandas.DataFrame()
    lrDF = pandas.DataFrame()
    mvrDF = pandas.DataFrame()

    for index, line in output.iterrows():
        if "nnr" in line[0]:
            nnrDF = nnrDF.append(line)
        # if "kernel" in line[0]:
        #     kernelDF = kernelDF.append(line)
        # if "lasso" in line[0]:
        #     lassoDF = lassoDF.append(line)
        if "ir" in line[0]:
            irDF = irDF.append(line)
        if "nn" in line[0]:
            nnDF = nnDF.append(line)
        if "spline" in line[0]:
            splineDF = splineDF.append(line)
        if "lr" in line[0]:
            lrDF = lrDF.append(line)
        if "mvr" in line[0]:
            mvrDF = mvrDF.append(line)

    nnrDF['name'] = nnrDF['name'].map(lambda x: x.strip('nnr_'))
    nnrDF['name'] = nnrDF['name'].map(lambda x: x.replace("_", ""))
    nnDF['name'] = nnDF['name'].map(lambda x: x.strip('nn'))
    nnDF['name'] = nnDF['name'].map(lambda x: x.replace("_", ""))
    splineDF['name'] = splineDF['name'].map(lambda x: x.strip('spline'))
    splineDF['name'] = splineDF['name'].map(lambda x: x.replace("_", ""))
    # kernelDF['name'] = kernelDF['name'].map(lambda x: x.strip('kernel_'))
    # kernelDF['name'] = kernelDF['name'].map(lambda x: x.replace("_", ""))
    # lassoDF['name'] = lassoDF['name'].map(lambda x: x.strip('lasso_'))
    # lassoDF['name'] = lassoDF['name'].map(lambda x: x.replace("_", ""))
    irDF['name'] = irDF['name'].map(lambda x: x.strip('ir_'))
    irDF['name'] = irDF['name'].map(lambda x: x.replace("_", ""))
    lrDF['name'] = lrDF['name'].map(lambda x: x.strip('lr_'))
    lrDF['name'] = lrDF['name'].map(lambda x: x.replace("_", ""))
    mvrDF['name'] = mvrDF['name'].map(lambda x: x.strip('mvr_'))
    mvrDF['name'] = mvrDF['name'].map(lambda x: x.replace("_", ""))

    nnrDF = nnrDF.sort_values(by=['name'])
    splineDF = splineDF.sort_values(by=['name'])
    nnDF = nnDF.sort_values(by=['name'])
    # kernelDF = kernelDF.sort_values(by=['name'])
    # lassoDF = lassoDF.sort_values(by=['name'])
    irDF = irDF.sort_values(by=['name'])
    lrDF = lrDF.sort_values(by=['name'])
    mvrDF = mvrDF.sort_values(by=['name'])

    nnrNN = explanatoryNN()
    nnNN = explanatoryNN()
    splineNN = explanatoryNN()
    # kernelNN = explanatoryNN()
    # lassoNN = explanatoryNN()
    irNN = explanatoryNN()
    lrNN = explanatoryNN()
    mvrNN = explanatoryNN()

    
def metamodelhandler():
    dataframe033033033 = pandas.read_csv('metaFeaturesCC.csv', header=None)
    dataframe05025025 = pandas.read_csv('metaFeaturesCC.csv', header=None)
    dataframe02505025 = pandas.read_csv('metaFeaturesCC.csv', header=None)
    dataframe02502505 = pandas.read_csv('metaFeaturesCC.csv', header=None)
    dataframe080101 = pandas.read_csv('metaFeaturesCC.csv', header=None)
    dataframe010801 = pandas.read_csv('metaFeaturesCC.csv', header=None)
    dataframe010108 = pandas.read_csv('metaFeaturesCC.csv', header=None)
    # dataframe001 = pandas.read_csv('metaFeaturesNames.csv', header=None)
    results1 = pandas.read_csv('results_033_033_033.csv')
    results2 = pandas.read_csv('results_05_025_025.csv')
    results3 = pandas.read_csv('results_025_05_025.csv')
    results4 = pandas.read_csv('results_025_025_05.csv')
    results5 = pandas.read_csv('results_08_01_01.csv')
    results6 = pandas.read_csv('results_01_08_01.csv')
    results7 = pandas.read_csv('results_01_01_08.csv')
    # results5 = pandas.read_csv('results_0_0_1.csv')

    results1 = results1.sort_values(by=['name'])
    results2 = results2.sort_values(by=['name'])
    results3 = results3.sort_values(by=['name'])
    results4 = results4.sort_values(by=['name'])
    results5 = results5.sort_values(by=['name'])
    results6 = results6.sort_values(by=['name'])
    results7 = results7.sort_values(by=['name'])
    # results5 = results5.sort_values(by=['name'])

    dataframe033033033 = dataframe033033033.sort_values(by=[7])
    dataframe05025025 = dataframe05025025.sort_values(by=[7])
    dataframe02505025 = dataframe02505025.sort_values(by=[7])
    dataframe02502505 = dataframe02502505.sort_values(by=[7])
    dataframe080101 = dataframe080101.sort_values(by=[7])
    dataframe010801 = dataframe010801.sort_values(by=[7])
    dataframe010108 = dataframe010108.sort_values(by=[7])
    # dataframe001 = dataframe001.sort_values(by=[9])

    dataframe033033033['size'] = 0.33
    dataframe033033033['inference'] = 0.33
    dataframe033033033['accuracy'] = 0.33

    dataframe05025025['size'] = 0.5
    dataframe05025025['inference'] = 0.25
    dataframe05025025['accuracy'] = 0.25

    dataframe02505025['size'] = 0.25
    dataframe02505025['inference'] = 0.5
    dataframe02505025['accuracy'] = 0.25

    dataframe02502505['size'] = 0.25
    dataframe02502505['inference'] = 0.25
    dataframe02502505['accuracy'] = 0.5

    dataframe080101['size'] = 0.8
    dataframe080101['inference'] = 0.1
    dataframe080101['accuracy'] = 0.1

    dataframe010801['size'] = 0.1
    dataframe010801['inference'] = 0.8
    dataframe010801['accuracy'] = 0.1

    dataframe010108['size'] = 0.1
    dataframe010108['inference'] = 0.1
    dataframe010108['accuracy'] = 0.8

    # dataframe001['size'] = 0
    # dataframe001['inference'] = 0
    # dataframe001['accuracy'] = 1


    dataframe = dataframe033033033.append([dataframe05025025, dataframe02505025, dataframe02502505, dataframe080101, dataframe010801, dataframe010108], ignore_index=True)
    dataframe = dataframe[[0, 1, 2, 3, 4, 5, 6, 7, 'size', 'inference',  'accuracy']]
    print(dataframe.dtypes)

    #cols_to_norm = [0, 1, 2, 3, 4, 5, 6, 7, 'size', 'inference', 'accuracy']
    #dataframe[cols_to_norm] = dataframe[cols_to_norm].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

    results = results1.append([results2, results3, results4, results5, results6, results7], ignore_index=True)

    results['best'] = results[['nnr', 'nn', 'ir', 'lr', 'mvr', 'spline']].idxmax(axis=1)
    results = results.values
    metafeatures = dataframe.values
    metaLearner = metaLearnerNN()
    model = metaTrain(metaLearnerNN(), metafeatures, results)
    # serialize model to JSON
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("weights.h5")
    print("Saved model to disk")



#def testNN():
#    dataframe = pandas.read_csv('/home/jesper/Acquisition/data/logar2.csv', header=None)
#    dataset = dataframe.values
#    Y = np.array(range(0, len(dataset)))
#    test = MLPRegressor(activation='relu', solver='adam', max_iter=100000, hidden_layer_sizes=(16, 16), early_stopping=True, verbose=True)
#    test.fit(dataset, Y)
#
#    plt.plot(dataset, Y, 'b', test.predict(dataset), Y, 'r')
#    plt.savefig('test.pdf')


if __name__ == '__main__':
    # plotData()
    metamodelhandler()
    # load json and create model
    #json_file = open('model.json', 'r')
    #loaded_model_json = json_file.read()
    #json_file.close()
    #loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    #loaded_model.load_weights("model.h5")
    #print("Loaded model from disk")
    #loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    #score = loaded_model.evaluate(X, Y, verbose=0)
    #print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1] * 100))
    #NN = metaLearnerNN()
    #loss, model = metaTrain(NN, dataset)
    #print(loss)
