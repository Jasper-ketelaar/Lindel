from numpy import ndarray

from scripts.logistic_model.model import LRModel
from scripts.logistic_model.model import onehotencoder


class IndelModel(LRModel):

    def __init__(self, data: ndarray, **kwargs):
        super().__init__(data, "Indel Model", **kwargs)

    def units(self):
        return 2

    def get_loss_function(self):
        return 'binary_crossentropy'

    def split_sets(self):
        for i in range(self.data_length):
            x_append = self.x_train if i < self.train_size else self.x_valid
            y_append = self.y_train if i < self.train_size else self.y_valid
            indels = sum(self._y[i][:-21]), sum(self._y[i][-21:])
            x_append.append(onehotencoder(self.seq_train[i]))
            y_append.append(indels)
