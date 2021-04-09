from numpy import ndarray

from scripts.logistic_model.model import LRModel
from scripts.logistic_model.model import one_hot_encoder


class InsertionModel(LRModel):

    def __init__(self, data: ndarray, **kwargs):
        f"""
        Initializer for the Insertion model. The data here defines the training data in the form of an ndarray.
        Note that the super init is called with 'Insertion Model' as second argument since this model has that title

        @param data: The training data
        @param kwargs: Any additional arguments that might be passed to L{LRModel.__init__}
        """
        super().__init__(data, "Insertion Model", **kwargs)

    def units(self):
        """
        Explained in the super class.

        @return 21 because that is how many insertion classes there are
        """
        return 21

    def get_loss_function(self):
        """
        Explained in the super class.

        @return categorical_crossentropy because it is a classification model
        """
        return 'categorical_crossentropy'

    def split_sets(self):
        """
        The insertion model splits the sets based on the last 6 basepairs in the sequence as input and the normalized
        insertion frequency as the expected output.
        """
        for i in range(self.data_length):
            x_append = self.x_train if i < self.train_size else self.x_valid
            y_append = self.y_train if i < self.train_size else self.y_valid
            seq_insertions, ins_sum = self._y[i][-21:], sum(self._y[i][-21:])
            if 1 > ins_sum > 0:
                x_append.append(one_hot_encoder(self.seq_train[i][-6:]))
                y_append.append(seq_insertions / ins_sum)
