from numpy import ndarray

from scripts.logistic_model.model import LRModel


class DeletionModel(LRModel):

    def __init__(self, data: ndarray, **kwargs):
        f"""
        Initializer for the deletion model. The data here defines the training data in the form of an ndarray.
        Note that the super init is called with 'Deletion Model' as second argument since this model has that title
        
        @param data: The training data
        @param kwargs: Any additional arguments that might be passed to L{LRModel.__init__}
        """
        super().__init__(data, "Deletion Model", **kwargs)

    def units(self):
        """
        Explained in super class

        @return: 536 for the amount of deletion classes
        """
        return 536

    def get_loss_function(self):
        """
        Explained in the super class

        @return: categorial crossentropy since the problem is a classification problem
        """
        return 'categorical_crossentropy'

    def split_sets(self):
        """
        For the deletion model the sets are split into the entire set of 3033 features as input and the
        normalized deletion classes as expected output
        """
        for i in range(self.data_length):
            x_append = self.x_train if i < self.train_size else self.x_valid
            y_append = self.y_train if i < self.train_size else self.y_valid
            seq_deletions, del_sum = self._y[i, :536], sum(self._y[i, :536])
            if 1 > del_sum > 0:
                x_append.append(self.x[i])
                y_append.append(seq_deletions / del_sum)
