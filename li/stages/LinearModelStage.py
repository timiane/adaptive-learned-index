from stages.Stage import Stage
import numpy as np
import static_functions as sf


class LinearModelStage(Stage):

    def predict_next_model(self, data):  # predicts the index of the data
        prediction = []
        models_list = np.unique(data[:, 1])
        models_list = models_list.reshape(models_list.size, 1)
        for model_index in models_list:
            model = self.select_model(int(model_index))
            temp_data = self.select_data(model_index, data)
            temp_results = self.predict_on_model(temp_data, model)
            prediction.append(temp_results)
        prediction = np.vstack(prediction)
        sf.get_min_max_error(prediction, self.index)
        return prediction

    def predict_on_model(self, data, model):  # apply the data to the model
        keys = data[:, 0]
        keys = keys.reshape(keys.size, 1)
        return model.predict_next_model(keys)
