from models.model import model
from scipy.interpolate import InterpolatedUnivariateSpline
import numpy as np

class Spline(model):
    def __init__(self, data, labels):
        self.model = self.create_model(data, labels)

    def create_model(self, data, labels):
        return InterpolatedUnivariateSpline(data, labels)

    def predict(self, data):
        data = np.array(data).reshape(-1, 1)
        return self.model(data)
