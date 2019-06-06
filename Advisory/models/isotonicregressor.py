from models.model import model
from sklearn.isotonic import IsotonicRegression
import numpy as np

class IsotonicRegressor(model):
    def __init__(self, data, labels):
        self.model = self.create_model(data, labels)


    def create_model(self, data, labels):
        x = data
        y = labels
        return IsotonicRegression().fit(x, y)

    def predict(self, data):
        return self.model.predict(data.flatten())

