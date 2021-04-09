from numpy import ndarray

from scripts.logistic_model.model import LRModel
from scripts.logistic_model.model import one_hot_encoder


class IndelModel(LRModel):

    def __init__(self, data: ndarray, **kwargs):
        f"""
        Initializer for the Indel model. The data here defines the training data in the form of an ndarray.
        Note that the super init is called with 'Indel Model' as second argument since this model has that title
        
        @param data: The training data
        @param kwargs: Any additional arguments that might be passed to L{LRModel.__init__}
        """
        super().__init__(data, "Indel Model", **kwargs)

    def units(self):
        """
        Explained in the super class.

        @return: 2 because that is how many values this model returns (a ratio)
        """
        return 2

    def get_loss_function(self):
        """
        Explained in the super class

        @return: binary_crossentropy because this model is not trying to classify but predict a ration
        """
        return 'binary_crossentropy'

    def split_sets(self):
        """
        This model splits the sets in the one_hot_encoded 20bp target sequence and the sum of the insertion events
        in a tuple with the sum of the deletion events
        """
        for i in range(self.data_length):
            x_append = self.x_train if i < self.train_size else self.x_valid
            y_append = self.y_train if i < self.train_size else self.y_valid
            indels = sum(self._y[i][:-21]), sum(self._y[i][-21:])
            x_append.append(one_hot_encoder(self.seq_train[i]))
            y_append.append(indels)
