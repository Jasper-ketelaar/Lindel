from numpy import ndarray

from scripts.logistic_model.model import LRModel


class DeletionModel(LRModel):

    def __init__(self, data: ndarray, **kwargs):
        super().__init__(data, "Deletion Model", **kwargs)

    def units(self):
        return 536

    def get_loss_function(self):
        return 'categorical_crossentropy'

    def split_sets(self):
        for i in range(self.data_length):
            x_append = self.x_train if i < self.train_size else self.x_valid
            y_append = self.y_train if i < self.train_size else self.y_valid
            seq_deletions, del_sum = self._y[i, :536], sum(self._y[i, :536])
            if 1 > del_sum > 0:
                x_append.append(self.x[i])
                y_append.append(seq_deletions / del_sum)