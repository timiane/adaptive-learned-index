import keras
import static_functions as sf
import numpy as np
from hyperopt import hp, STATUS_OK


class NeuralNetwork:

    def __init__(self, data=np.array([[1, 2, 3], [1, 2, 3]])):
        self.activation = ["relu", "linear", "sigmoid"]

        self.activation_last = ["relu", "linear", "sigmoid"]

        self.batch_size = [int(data[:, 1].size / 10), int(data[:, 1].size / 100)]
        self.layer_nodes = [16, 32]
        self.lr = [0.001, 0.01, 0.1]
        self.patience = [10]
        self.data = data
        self.space = {
            'activation': hp.choice('activation', self.activation),
            'activation_last': hp.choice('activation_last', self.activation_last),
            'batch_size': hp.choice('batch_size', self.batch_size),
            'layer_nodes': hp.choice('layer_nodes', self.layer_nodes),
            'lr': hp.choice('lr', self.lr),
            'patience': hp.choice('patience', self.patience)
        }
        self.model = None
        self.key_normalizer = data[:, 0].max()
        # self.key_normalizer = 1
        # self.label_normalizer = 1
        self.label_normalizer = data[:, 1].max()

    def create_model(self, hyper_parameters, new=False):
        model = keras.Sequential()
        # Layers
        model.add(keras.layers.Dense(units=hyper_parameters['layer_nodes'], input_dim=1,
                                     activation=hyper_parameters['activation']))
        # model.add(keras.layers.Dense(units=1, activation=hyper_parameters['activation_last']))
        model.add(keras.layers.Dense(units=1))
        adam = keras.optimizers.adam(lr=hyper_parameters['lr'])
        model.summary()
        model.compile(optimizer=adam, loss='mae')
        if new:
            results = self.train(model, self.data, hyper_parameters)[1]
            self.model = results
        else:
            results = self.train(model, self.data, hyper_parameters)[0]
        return results

    def train(self, m, data, hyper_parameters):
        keys = sf.normalize_max(data[:, 0])
        # keys = data[:, 0]
        # labels = data[:, 1]
        labels = sf.normalize_max(data[:, 1])
        print('The parameters for the model is '+ str(hyper_parameters))
        # batch = hyper_parameters['batch_size']
        batch = data.size
        callback = [
            keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.000, patience=20,
                                          restore_best_weights=True)]
        m.fit(keys, labels, epochs=1000, verbose=1, batch_size=batch, callbacks=callback)
        loss = m.evaluate(keys, labels, batch_size=batch)
        print('a model just finished with loss ' + str(loss))
        return {'loss': loss, 'status': STATUS_OK}, m

    def loss(self, y_true, y_pred):
        return np.max(abs(y_true - y_pred))
