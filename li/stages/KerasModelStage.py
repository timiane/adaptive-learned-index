from stages.Stage import Stage
import numpy as np
import static_functions as sf


class KerasModelStage(Stage):
    def predict_next_model(self, data, next_layer_size):  # predict model to use in next layer
        prediction = []
        models_list = np.unique(data[:, 1])
        models_list = models_list.reshape(models_list.size, 1)
        for model_index in models_list:
            model = self.select_model(int(model_index))
            temp_data = self.select_data(model_index, data)
            temp_results = self.predict_on_model(model, temp_data)
            prediction.append(temp_results)
        prediction = np.vstack(prediction)
        sf.get_min_max_error(prediction, self.index)

        return self.model_predictions(data[:, 1].size, next_layer_size, prediction)

    def predict_on_model(self, model, temp_data):  # apply data to model
        temp_data /= model.key_normalizer
        keys = temp_data[:, 0]
        temp_results = model.model.predict_next_model(keys, batch_size=10000)
        temp_results *= model.label_normalizer
        keys = keys*model.key_normalizer
        #sf.plot_results_vs_actual(keys, temp_results)

        return temp_results



    def model_predictions(self, data_size, next_layer_size,
                          prediction):  # determine model based on predicted index and next layer size
        models_keys_pr_model = (data_size / next_layer_size)
        return np.floor_divide(prediction, models_keys_pr_model)


