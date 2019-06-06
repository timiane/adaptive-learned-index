from models.model import model
from scipy.interpolate import CubicSpline
import numpy as np

class Spline(model):
    def __init__(self, data, labels):
        self.model = self.create_model(data, labels)

    def create_model(self, data, labels):
        data = np.unique(data)
        return CubicSpline(data, labels)

    def predict(self, data):
        data = np.array(data).reshape(-1, 1)
        return self.model(data)
