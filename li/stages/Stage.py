from joblib import dump, load
import numpy as np
import os


class Stage:
    def __init__(self, index, models=[]):
        self.models = []
        self.index = index
        self.length = len(models)

    def select_model(self, index):
        if index <= 0:
            return self.models[0]
        elif index >= self.length - 1:
            return self.models[self.length - 1]
        else:
            return self.models[index]

    def save_stage(self, path):
        for position, model in enumerate(self.models):
            dump(model, path + str(self.index) + '.' + str(position) + '.joblib')

    def select_data(self, index, data):
        return data[np.where(data[:, 1] == index)]

    @classmethod
    def load_stage(cls, path, index):
        directory = os.listdir(path)
        directory.sort()
        directory.sort(key=len, reverse=False)
        models = []
        for file in directory:
            if file.startswith(str(index)):
                models.append(load(path + str(file)))
        return cls(index, models)

    def predict_next_model(self, data, next_layer_size):  # predict model to use in next layer
        prediction = []
        models_list = np.unique(data[:, 1])
        models_list = models_list.reshape(models_list.size, 1)
        for model_index in models_list:
            model = self.select_model(int(model_index))
            temp_data = self.select_data(model_index, data)[:, 0]
            temp_results = model.predict(temp_data)
            prediction.append(temp_results)
        prediction = np.vstack(prediction)
        # sf.get_min_max_error(prediction, self.index)

        return self.model_predictions(data[:, 1].size, next_layer_size, prediction)

    def predict(self, data):  # predicts the index of the data
        prediction = []
        models_list = np.unique(data[:, 1])
        models_list = models_list.reshape(models_list.size, 1)
        for model_index in models_list:
            model = self.select_model(int(model_index))
            temp_data = self.select_data(model_index, data)[:, 0]
            temp_results = model.predict(temp_data)
            prediction.append(temp_results)
        prediction = np.array(prediction).reshape(-1, 1)
        prediction = np.vstack(prediction)
        # sf.get_min_max_error(prediction, self.index)
        return prediction

    def model_predictions(self, data_size, next_layer_size,
                          prediction):  # determine model based on predicted index and next layer size
        models_keys_pr_model = (data_size / next_layer_size)
        return np.floor_divide(prediction, models_keys_pr_model)


