from abc import ABC, abstractmethod
from functools import cached_property
from typing import Callable, Tuple, List, Any, Union

import numpy as np
from numpy import ndarray
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2, l1
from tensorflow.python.keras.regularizers import Regularizer


def mse(x: np.ndarray, y: np.ndarray) -> Union[np.ndarray, float]:
    """
    Computes mean square error between two arr
    @param x:
    @param y:
    @return:
    """
    return ((x - y) ** 2).mean()


def softmax(output: np.ndarray) -> np.ndarray:
    """
    Performs the softmax formula on a models output
    @param output: The output of a model
    @return: A normalized ndarray containing the values computed after executing the softmax formula
    """
    return np.exp(output) / sum(np.exp(output))


def one_hot_encoder(seq: str) -> np.ndarray:
    """
    Takes a sequence and performs a one hot encoding no it
    @param seq: The input sequence which is to be onehotencoded
    @return:
    """
    nt = ['A', 'T', 'C', 'G']
    head = []
    seq_len = len(seq)

    # list all Single-Nucleotide seq-pos options ['N#'] (length = 4*L)
    for k in range(seq_len):
        for i in range(4):
            head.append(nt[i] + str(k))

    # list all Di-Nucleotide seq-pos options ['NN#] (length = 16*(L-1))
    for k in range(seq_len - 1):
        for i in range(4):
            for j in range(4):
                head.append(nt[i] + nt[j] + str(k))

    # Dict: seq-label --> 1-hot index
    head_idx = {}

    for idx, key in enumerate(head):
        head_idx[key] = idx
    encode = np.zeros(len(head_idx))
    # set applicable single and dinucleotide features to 1
    for j in range(seq_len):
        encode[head_idx[seq[j] + str(j)]] = 1.
    for k in range(seq_len - 1):
        encode[head_idx[seq[k:k + 2] + str(k)]] = 1.
    return encode


class LRModel(ABC):
    """
    A generic representation of the logistic regression model used for all of the models for the Lindel final model.
    This is an abstract class and contains some abstract methods that eventually define the difference between the
    models.
    """

    @property
    def x_train(self):
        """
        @return: list defining the train input
        """
        return self._x_train

    @property
    def y_train(self):
        """
        @return: list defining the train output
        """
        return self._y_train

    @property
    def x_valid(self):
        """
        @return: list definition the validation input
        """
        return self._x_valid

    @property
    def y_valid(self):
        """
        @return: list definition the validation output
        """
        return self._y_valid

    @property
    def name(self):
        """
        @return: The name of the model
        """
        return self._name

    @property
    def file_name_formatted(self):
        """
        @return: A formatted version of the name where spaces are replaced with underscores and it is all lowercase
        """
        return self.name.replace(" ", "_").lower()

    @property
    def observers(self):
        """
        We added the functionality to observe the models at different points in time to make it more compatible with
        notebooks and progress updates and such

        @return: The list of observers that this model has
        """
        return self._observers

    @property
    def train_size(self):
        """
        @return: The training size
        """
        return self._train_size

    @cached_property
    def lambdas(self):
        """
        @return: The possible kernel values for the regularizers. This property was cached because it is called
        multiple times and the extra computations are unnecessary
        """
        return 10 ** np.arange(-10, -1, 0.1)

    @property
    def x(self):
        """
        @return: A matrix representing sequences to binary features
        """
        return self._x

    @property
    def y(self):
        """
        @return: A matrix representing sequences to profiles
        """
        return self._y

    @property
    def seq_train(self):
        """
        @return: The sequences on which trainining is performed
        """
        return self._seq_train

    @property
    def feature_size(self):
        """
        @return: How many binary features there are. This was loaded from the pkl file before but since the
            final version has 3033 this value will, unless altered, always be 3033 without needing to load the whole
            pickle file
        """
        return self._feature_size

    @property
    def data_length(self):
        """
        @return: The total size of the training set
        """
        return self._data_len

    def __init__(
            self,
            data: ndarray,
            title: str,
            feature_size: int = 3033,
            split_proportion: float = .9,
            save_directory=""
    ):
        self._feature_size = feature_size
        self._save_directory = save_directory
        self._name = title
        self._data = data
        self._seqs = self._data[:, 0]
        self._data = self._data[:, 1:].astype('float32')
        self._x = self._data[:, :feature_size]
        self._y = self._data[:, feature_size:]
        self._x_train, self._x_valid = [], []
        self._y_train, self._y_valid = [], []
        self._idx = np.arange(len(self._y))
        self._data_len = len(self._data)
        self._train_size = round(self._data_len * split_proportion)
        self._valid_size = round(self._data_len * (1 - split_proportion))

        # Seed with any value for predictable results not sure if 121 was thought about
        np.random.seed(121)
        np.random.shuffle(self._idx)
        self._x = self._x[self._idx]
        self._y = self._y[self._idx]
        self._seq_train = self._seqs[self._idx]
        self._observers = []
        self._model = None

    @abstractmethod
    def split_sets(self) -> None:
        """
        Abstract method to split the sets. This is what primarily distinguishes the models. Namely what input is used
        and what output is expected. The function itself does not return anything in specific however it adjusts the
        values that this class uses to train the models
        """
        raise NotImplemented

    @abstractmethod
    def get_loss_function(self) -> str:
        """
        Since the models have crossentropy but binary for indel and categorical for deletions, insertions we decided
        to make this abstract. Obviously we could deduce this from the way the sets are split but this was done for
        clarity.

        @return: the loss function name
        """
        return NotImplemented

    @abstractmethod
    def units(self) -> int:
        """
        The units define the expected output size of our model. Once again we could derive this from the split sets
        but for clarity we made it separate.

        @return: the integer value for the units
        """
        return NotImplemented

    def add_observer(self, observer):
        """
        Method to add an observer to a model. These are used to perform certain debugging during the code.
        The variable names of the observer function are analyzed and stored so that we can call the observer functions
        if the parameter names are the same as the variables that were notified during processsing

        @param observer: the observer function that is to be added.
        """
        self._observers.append({
            'observer': observer,
            'params': observer.__code__.co_varnames[0:observer.__code__.co_argcount]
        })

    def _notify_observers(self, **kwargs) -> None:
        """
        Function that can be called to notify the observers about any variables. It checks all observers and
        sees is there are any observers that have an exact match between the arg key names defined in kwargs and the
        arg names of the observer function. We also check if self is part of the params to exclude it in case the
        observer function passed is part of an object.

        @param kwargs: the arguments that changed that observers might have been observing
        """
        key_len = len(kwargs.keys())

        def _notify_observer():
            params: Tuple[str] = observer['params']
            if 'self' in params:
                params = params[1:]
            if len(params) != key_len:
                return

            if not all(x in kwargs.keys() for x in params):
                return

            passing: List[Any] = list(map(lambda param: kwargs.get(param), params))
            fun_callable = observer['observer']
            fun_callable(*passing)

        for observer in self.observers:
            _notify_observer()

    def _train_model(self, train_l1: bool = True, train_l2: bool = True):
        """
        The function where the model gets trained. We added the possibility to exclude training
        l1 or l2 regularized versions for computational improvements once we already knew which performed best.

        This function trains the l1 version, the l2 version or both versions depending on the parameters and if
        both versions were trained it prints out the lowest errors for both versions so that we do not have to look
        closely at a plotted version of the error graph to decide what version to use for the final model.

        After training the best performing model (or only performing model) is also stored as a class variable if
        you wish to continue using it without serialization and deserialization

        @param train_l1: True by default, set to False if the l1 regularizer model version should not be trained
        @param train_l2: True by default, set to False if the l2 regularizer model version should not be trained
        """
        x_train, x_valid = np.array(self.x_train), np.array(self.x_valid)
        y_train, y_valid = np.array(self.y_train), np.array(self.y_valid)

        model_l2 = None
        errors_l2 = []
        min_l2 = -1

        model_l1 = None
        errors_l1 = []
        min_l1 = -1

        if train_l1 is True:
            errors_l1, model_l1, min_l1 = self._train_model_reg(x_train, x_valid, y_train, y_valid)

        if train_l2 is True:
            errors_l2, model_l2, min_l2 = self._train_model_reg(x_train, x_valid, y_train, y_valid, regularizer=l2)

        if train_l1 is True and train_l2 is True:
            self._notify_observers(errors_l1=errors_l1, errors_l2=errors_l2)

        if train_l1 is False:
            self._model = model_l2
        elif train_l2 is False:
            self._model = model_l1
        else:
            self._model = model_l1 if errors_l1[min_l1] < errors_l2[min_l2] else model_l2

        if min_l1 > 0 and min_l2 > 0:
            print(f'Smallest errors l1 regularizer {errors_l1[min_l1]}')
            print(f'Smallest errors l2 regularizer {errors_l2[min_l2]}')

    def _train_model_reg(self,
                         x_train: np.ndarray,
                         x_valid: np.ndarray,
                         y_train: np.ndarray,
                         y_valid: np.ndarray,
                         regularizer: Callable[[float], Regularizer] = l1
                         ) -> (list, Sequential, float):
        """
        Trains a specific model with the inputs and outputs for both training and validation. Also allows you to define
        what regularizer gets used

        @param x_train: the training input
        @param x_valid: the validation input
        @param y_train: the training expected output
        @param y_valid: the validation expected output
        @param regularizer: the regularizer type, l1 by default
        @return: The errors (mse) for the different regularizer kernel values, the model that performed best
            and the lowest error index
        """
        errors = []
        size_input = x_train.shape[1]
        reg_name = "l1" if regularizer == l1 else "l2"
        self._notify_observers(reg_name=reg_name, lambdas=self.lambdas)
        for reg_val in self.lambdas:
            model = self._exec_sequential_model(
                reg_val,
                x_train,
                x_valid,
                y_train,
                y_valid,
                size_input,
                regularizer=regularizer
            )

            y_hat = model.predict(x_valid)
            errors.append(mse(y_hat, y_valid))
            self._notify_observers(reg_name=reg_name, reg_val=reg_val)

        np.save(self._save_directory +
                "{0}_{1}.npy".format(reg_name, self.file_name_formatted), errors)
        min_err = np.argmin(errors)
        opt_reg = self.lambdas[min_err]
        model = self._exec_sequential_model(
            opt_reg,
            x_train,
            x_valid,
            y_train,
            y_valid,
            size_input,
            regularizer=regularizer
        )
        model.save(f'{self._save_directory}{reg_name}_{self.file_name_formatted}.h5')

        self._notify_observers(regularizer=reg_name, lambdas=self.lambdas, errors=errors)
        return errors, model, min_err

    def _exec_sequential_model(
            self,
            reg_val: float,
            x_train: np.ndarray,
            x_valid: np.ndarray,
            y_train: np.ndarray,
            y_valid: np.ndarray,
            size_input: int,
            regularizer: Callable[[float], Regularizer] = l1
    ) -> Sequential:
        """
        Performs a single model construction, compilation and fitting for a specific regularizer value

        @param reg_val: float describing the regularizer value
        @param x_train: the training input
        @param x_valid: the validation input
        @param y_train: the training expected output
        @param y_valid: the validation expected output
        @param size_input: The size of the input
        @param regularizer: the regularizer type, l1 by default
        @return: The model
        """
        np.random.seed(0)
        model = Sequential()
        model.add(Dense(self.units(), activation='softmax', input_shape=(size_input,),
                        kernel_regularizer=regularizer(reg_val)))
        model.compile(optimizer='adam', loss=f'{self.get_loss_function()}', metrics=['mse'])
        model.fit(x_train, y_train, epochs=100, validation_data=(x_valid, y_valid),
                  callbacks=[EarlyStopping(patience=1)], verbose=0)

        return model

    def split_and_train(self, **kwargs):
        """
        Function to call to actually split the sets and train the model
        @param kwargs: Possible arguments to pass to prevent l1 or l2 models from executing
        """
        self.split_sets()
        self._train_model(**kwargs)
