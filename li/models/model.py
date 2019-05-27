from abc import abstractmethod, ABC


class model(ABC):
    # def __init__(self):
    #     self.model = None
    #     super().__init__()


    @abstractmethod
    def predict(self, data):
        pass

    @abstractmethod
    def create_model(self, data, labels):
        pass
