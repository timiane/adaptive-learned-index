from models.model import model
from sklearn.neighbors import KNeighborsRegressor
import numpy as np


class NearestNeighborRegression(model):
    def __init__(self, data, labels):
        self.model = self.create_model(data, labels)

    def create_model(self, data, labels):
        y = labels.reshape(-1, 1)
        data = data.reshape(-1, 1)
        return KNeighborsRegressor(n_neighbors=2, weights='distance').fit(data, y)

    def predict(self, data):
        data = np.array(data).reshape(-1, 1)
        return self.model.predict(data)
