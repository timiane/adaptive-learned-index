from sklearn import linear_model
from models.model import model


class Linear(model):

    def __init__(self, dataset, labels):
        self.model = self.create_model(dataset, labels)


    def create_model(self, data, labels):
        reg = linear_model.LinearRegression()
        if data.size > 0:
            labels = labels
            keys = data
            labels = labels.reshape(labels.size, 1)
            keys = keys.reshape(keys.size, 1)
            reg.fit(keys, labels)

        return reg

    # def predict(self, data):  # predicts the index of the data
    #     prediction = []
    #     models_list = np.unique(data[:, 1])
    #     models_list = models_list.reshape(models_list.size, 1)
    #     for model_index in models_list:
    #         model = self.select_model(int(model_index))
    #         temp_data = self.select_data(model_index, data)
    #         temp_results = self.predict_on_model(temp_data, model)
    #         prediction.append(temp_results)
    #     prediction = np.vstack(prediction)
    #     sf.get_min_max_error(prediction, self.index)
    #     return prediction

    def predict(self, data):  # apply the data to the model
        keys = data
        keys = keys.reshape(keys.size, 1)
        return self.model.predict(keys)