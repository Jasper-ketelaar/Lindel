from abc import ABC, abstractmethod
from functools import cached_property
from typing import Callable, Tuple, List, Any

import numpy as np
from numpy import ndarray
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2, l1
from tensorflow.python.keras.regularizers import Regularizer
import pydot
from tensorflow.python.keras.utils.vis_utils import plot_model


def mse(x, y):
    return ((x - y) ** 2).mean()


def corr(x, y):
    return np.corrcoef(x, y)[0, 1] ** 2


def onehotencoder(seq):
    nt = ['A', 'T', 'C', 'G']
    head = []
    l = len(seq)
    for k in range(l):
        for i in range(4):
            head.append(nt[i] + str(k))

    for k in range(l - 1):
        for i in range(4):
            for j in range(4):
                head.append(nt[i] + nt[j] + str(k))
    head_idx = {}
    for idx, key in enumerate(head):
        head_idx[key] = idx
    encode = np.zeros(len(head_idx))
    for j in range(l):
        encode[head_idx[seq[j] + str(j)]] = 1.
    for k in range(l - 1):
        encode[head_idx[seq[k:k + 2] + str(k)]] = 1.
    return encode


class LRModel(ABC):
    @property
    def x_train(self):
        return self._x_train

    @property
    def y_train(self):
        return self._y_train

    @property
    def x_valid(self):
        return self._x_valid

    @property
    def y_valid(self):
        return self._y_valid

    @property
    def name(self):
        return self._name

    @property
    def file_name_formatted(self):
        return self.name.replace(" ", "_").lower()

    @property
    def observers(self):
        return self._observers

    @property
    def train_size(self):
        return self._train_size

    @cached_property
    def lambdas(self):
        return 10 ** np.arange(-10, -1, 0.1)

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def seq_train(self):
        return self._seq_train

    @property
    def feature_size(self):
        return self._feature_size

    @property
    def data_length(self):
        return self._data_len

    def __init__(
            self,
            data: ndarray,
            title: str,
            feature_size: int = 3033,
            ignore_proportion: float = 0,
            split_proportion: float = .9,
            save_directory=""
    ):
        self._feature_size = feature_size
        self._save_directory = save_directory
        self._name = title
        data_len_ignored = round(len(data) * ignore_proportion)
        print(data_len_ignored)
        self._data = data[data_len_ignored:, :]
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

    @abstractmethod
    def split_sets(self):
        return NotImplemented

    @abstractmethod
    def get_loss_function(self) -> str:
        return NotImplemented

    @abstractmethod
    def units(self):
        return NotImplemented

    def add_observer(self, observer):
        self._observers.append({
            'observer': observer,
            'params': observer.__code__.co_varnames[1:observer.__code__.co_argcount]
        })

    def _notify_observers(self, **kwargs):
        key_len = len(kwargs.keys())

        def _notify_observer():
            params: Tuple[str] = observer['params']
            if len(params) != key_len:
                return

            if not all(x in kwargs.keys() for x in params):
                return

            passing: List[Any] = list(map(lambda param: kwargs.get(param), params))
            fun_callable = observer['observer']
            fun_callable(*passing)

        for observer in self.observers:
            _notify_observer()

    def _train_model(self):
        x_train, x_valid = np.array(self.x_train), np.array(self.x_valid)
        y_train, y_valid = np.array(self.y_train), np.array(self.y_valid)
        errors_l1 = self._train_model_reg(x_train, x_valid, y_train, y_valid)
        errors_l2 = self._train_model_reg(x_train, x_valid, y_train, y_valid, regularizer=l2)
        self._notify_observers(errors_l1=errors_l1, errors_l2=errors_l2)

    def _train_model_reg(self,
                         x_train,
                         x_valid,
                         y_train,
                         y_valid,
                         regularizer: Callable[[float], Regularizer] = l1
                         ):

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

        if self._save_directory != "":
            np.save(self._save_directory +
                    "{0}_{1}.npy".format(reg_name, self.file_name_formatted), errors)
            opt_reg = self.lambdas[np.argmin(errors)]
            self._exec_sequential_model(
                opt_reg,
                x_train,
                x_valid,
                y_train,
                y_valid,
                size_input,
                regularizer=regularizer,
                save=True
            )

        self._notify_observers(regularizer=reg_name, lambdas=self.lambdas, errors=errors)
        return errors

    def _exec_sequential_model(
            self,
            reg_val,
            x_train,
            x_valid,
            y_train,
            y_valid,
            size_input,
            metrics=None,
            save=False,
            regularizer: Callable[[float], Regularizer] = l1
    ) -> Sequential:
        if metrics is None:
            metrics = ['mse']
        np.random.seed(0)
        model = Sequential()
        model.add(
            Dense(self.units(), activation='softmax', input_shape=(size_input,),
                  kernel_regularizer=regularizer(reg_val)))
        model.compile(optimizer='adam', loss=f'{self.get_loss_function()}', metrics=metrics)
        model.fit(x_train, y_train, epochs=100, validation_data=(x_valid, y_valid),
                  callbacks=[EarlyStopping(patience=1)], verbose=0)

        if save is True and self._save_directory != "":
            model.save(self._save_directory + f'L1_{self.name.replace(" ", "_").lower()}.h5')

        return model

    def start(self):
        self.split_sets()
        self._train_model()
