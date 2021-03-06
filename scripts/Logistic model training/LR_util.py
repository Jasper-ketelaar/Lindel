import os
import pickle as pkl
from functools import cached_property
from os import sep
from os.path import dirname
from typing import Callable, Tuple, List, Any

import numpy as np
from IPython import display as ids
from ipywidgets import VBox, Button, Layout, HTML, Output
from matplotlib import pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2, l1
from tensorflow.python.keras.regularizers import Regularizer
from tqdm.notebook import tqdm

import selector.file_chooser as fcw


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


class FeatureLoader:

    @property
    def training_chooser(self) -> fcw.FileChooser:
        return self._file_chooser

    @property
    def data_directory(self) -> str:
        return self._file_chooser.selected_path + sep

    @property
    def feature_index(self) -> dict:
        label, rev_index, features = pkl.load(open(self.data_directory + 'feature_index_all.pkl', 'rb'))
        return dict({
            'label': label,
            'rev_index': rev_index,
        })

    @property
    def model_file(self) -> str:
        return self._file_chooser.selected_filename

    @property
    def ready(self) -> bool:
        return self.model_file is not None and self.model_file != ""

    @property
    def feature_size(self) -> int:
        return len(self.features) + 384

    @property
    def features(self):
        return self._features

    @property
    def label(self):
        return self._label

    @property
    def reverse_index(self):
        return self._rev_index

    @property
    def file_path(self) -> str:
        return self.data_directory + self.model_file

    @property
    def is_forecast(self) -> bool:
        return 'ForeCasT' in self.model_file

    @property
    def output(self) -> Output:
        return self._output

    def __init__(self):
        self._file_chooser = fcw.FileChooser(
            title="Select the set to run the linear regression on:",
            path=f"{dirname(dirname(os.getcwd()))}/cwd/",
            select_desc="Choose training/test set",
            change_desc="Choose different file",
            file_ends_with='.txt'
        )
        self._output = Output(layout={'border': '1px solid green'})
        self._file_chooser.register_callback(self.loadFeatureFile)
        self._file_chooser.show_dialog()
        self._features = None
        self._rev_index = None
        self._label = None
        self._data = None

    def loadFeatureFile(self, _):
        self.get_data_np()

        try:
            fia_pkl_same = open(self.data_directory + 'feature_index_all.pkl', 'rb')
            self.loadFeaturePickle(fia_pkl_same)
        except FileNotFoundError:
            pkl_chooser = fcw.FileChooser(
                title="The feature_index_all.pkl file is not in the same directory, please select it:",
                select_desc="Select feature_index_all.pkl",
                change_desc="Choose different file",
                file_ends_with="feature_index_all.pkl"
            )

            def pkl_callback(_):
                fia_pkl = open(pkl_chooser.selected)
                self.loadFeaturePickle(fia_pkl)

            pkl_chooser.register_callback(pkl_callback)

            # https://github.com/ipython/ipython/issues/12182
            # noinspection PyTypeChecker
            ids.display(pkl_chooser)

    def loadFeaturePickle(self, fia_pkl):
        label, rev_index, features = pkl.load(fia_pkl)
        self._label = label
        self._rev_index = rev_index
        self._features = features
        fia_pkl.close()
        with self.output:
            self.output.clear_output()
            self.output.append_stdout("Working directory: {0} \n".format(self.data_directory))
            self.output.append_stdout("Model: {0} \n".format(self.model_file))
            self.output.append_stdout("Is ForeCasT: {0}\n".format("Yes" if self.is_forecast else "No"))
            self.output.append_stdout("Feature size: {0}\n".format(self.feature_size))

    def get_data_np(self):
        if self._data is None:
            self._data = np.loadtxt(self.file_path, delimiter="\t", dtype=str)

        return self._data


class LRModel:
    lr_model_attributes: dict = {
        'Insertion': (21, 'categorical'),
        'Indel': (2, 'binary'),
        'Deletion': (536, 'categorical')
    }

    @property
    def x_train(self):
        return self.x_train

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
    def units(self):
        un, _ = LRModel.lr_model_attributes[self.name]
        return un

    @property
    def cross_entropy_type(self):
        _, ty = LRModel.lr_model_attributes[self.name]
        return ty

    @property
    def should_oh_encode(self):
        return self.name != 'Deletion Model'

    @property
    def observers(self):
        return self._observers

    @cached_property
    def lambdas(self):
        return 10 ** np.arange(-10, -1, 0.1)

    def __init__(
            self,
            feature_loader: FeatureLoader,
            title: str
    ):
        self._feature_loader = feature_loader
        self._name = title
        self._data = feature_loader.get_data_np()
        self._seqs = self._data[:, 0]
        self._data = self._data[:, 1:].astype('float32')
        self._x = self._data[:, :self._feature_loader.feature_size]
        self._y = self._data[:, self._feature_loader.feature_size:]
        self._x_train, self._x_valid = [], []
        self._y_train, self._y_valid = [], []
        self._idx = np.arange(len(self._y))
        self._data_len = len(self._data)
        self._train_size = round(self._data_len * 0.9) if self._feature_loader.is_forecast else 3900
        self._valid_size = round(self._data_len * 0.1) if self._feature_loader.is_forecast else 450

        # Seed with any value for predictable results not sure if 121 was thought about
        np.random.seed(121)
        np.random.shuffle(self._idx)
        self._x = self._x[self._idx]
        self._y = self._y[self._idx]
        self._seq_train = self._seqs[self._idx]
        self._observers = []

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

    def _training_shape(self, i):
        # indel sets dont have condition for being added as it uses binary data
        # their y data shape is split for the last 21 values
        if self.units == 2:
            return sum(self._y[i][:-21]), sum(self._y[i][-21:])

        # For deletion they take the sum of the first 536 values and check if it's between 0 and 1
        # Then use this to normalize the array they use for their model shape
        if "Deletion" in self.name:
            return self._y[i, :536], sum(self._y[i, :536])

        # For insertion they do the same as deletion except they use the last 21
        return self._y[i, -21:], sum(self._y[i, -21:])

    def _add_set_at(self, i, train=True):
        y_val, y_sum = self._training_shape(i)
        x_set = self._x_train if train else self.x_valid
        y_set = self._y_train if train else self._y_valid

        if self.units == 2 or 1 > y_sum > 0:
            y_set.append(y_val / y_sum if self.units > 2 else (y_val, y_sum))

            # Only indels/insertions are one hot encoded.
            if self.should_oh_encode:
                seq_part = self._seq_train[i] if self.units == 2 else self._seq_train[i][-6:]
                x_set.append(onehotencoder(seq_part))
            else:
                x_set.append(self._x[i])

    def _split_sets(self):
        for i in range(self._train_size):
            self._add_set_at(i)
        for i in range(self._train_size, len(self._seq_train)):
            self._add_set_at(i, False)

        x_train, x_valid = np.array(self._x_train), np.array(self._x_valid)
        y_train, y_valid = np.array(self._y_train), np.array(self._y_valid)
        self._train_model(x_train, x_valid, y_train, y_valid)

    def _train_model(self, x_train, x_valid, y_train, y_valid):
        errors_l1 = self._train_model_reg(x_train, x_valid, y_train, y_valid)
        errors_l2 = self._train_model_reg(x_train, x_valid, y_train, y_valid, regularizer=l2)
        self._notify_observers(errors_l1=errors_l1, errors_l2=errors_l2)

    def _train_model_reg(self,
                         x_train,
                         x_valid,
                         y_train,
                         y_valid,
                         save=True,
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

        if save:
            np.save(self._feature_loader.data_directory +
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
            Dense(self.units, activation='softmax', input_shape=(size_input,), kernel_regularizer=regularizer(reg_val)))
        model.compile(optimizer='adam', loss=f'{self.cross_entropy_type}_crossentropy', metrics=metrics)
        model.fit(x_train, y_train, epochs=100, validation_data=(x_valid, y_valid),
                  callbacks=[EarlyStopping(patience=1)], verbose=0)

        if save is True:
            model.save(self._feature_loader.data_directory + f'L1_{self.name.replace(" ", "_").lower()}.h5')

        return model

    def start(self):
        self._split_sets()


class LRWidget(VBox):

    @property
    def model(self) -> Any:
        return self._model

    def __init__(
            self,
            heading,
            fs=False,
            **kwargs
    ):
        self._model = None
        self._title = HTML(
            '<h2>{0}</h2>'.format(heading)
        )
        self._start = Button(
            description='Train Model (loading...)',
            layout=Layout(width='auto'),
            button_style='success'
        )
        if fs:
            self._fc = fcw.FileChooser(f"{dirname(dirname(os.getcwd()))}/cwd/", file_ends_with='npy')
            self._fc.register_callback(self._on_fs_loaded)
            self._model = {
                'lambdas': 10 ** np.arange(-10, -1, 0.1),
                'name': self._fc.selected
            }

        self._start.disable = True
        self._pbars = dict()
        self._children = [
            self._title,
            self.fil
        ] if not fs else [
            self._fc
        ]

        super().__init__(
            children=self._children,
            layout=Layout(width='600px' if fs else '300px'),
            **kwargs
        )

    def _on_fs_loaded(self, _):
        npy = self._fc.selected
        npy_2 = npy.replace("l1", "l2") if "l1" in npy else npy.replace("l2", "l1")
        mse = np.load(npy)
        mse_2 = np.load(npy_2)
        self._render_multi_plot(mse, mse_2)

    def on_model_loaded(self, model: LRModel):
        self._model = model

        self._model.add_observer(self._render_single_plot)
        self._model.add_observer(self._render_progress_bar)
        self._model.add_observer(self._update_progress_bar)

        self._start.on_click(self.on_start_clicked)
        self._start.disabled = False
        self._start.description = 'Train Model'

    def on_start_clicked(self, _):
        self._start.disabled = True
        self._start.description = 'Training...'
        self.model.start()
        # noinspection PyTypeChecker

    def _render_multi_plot(self, errors_l1, errors_l2):
        lambdas = self.model.lambdas if isinstance(self.model, LRModel) else self.model['lambdas']
        plt.title(self.model.name if isinstance(self.model, LRModel) else self.model['name'])
        plt.xscale('log', subs=range(1, 6))
        plt.xlabel('Regularization Strength')
        plt.ylabel('MSE')
        plt.plot(lambdas, errors_l1, label='L1')
        plt.plot(lambdas, errors_l2, label='L2')
        plt.legend()
        plt.show()

    def _render_single_plot(self, lambdas, errors):
        plt.title(f"{self.model.name} - {list(self._pbars.keys())[-1]} regularization" if self.model is not None else
                  f"{self._fc.selected}"
                  )
        plt.xlabel('Regularization Strength')
        plt.ylabel('MSE')
        plt.xscale('log')
        plt.plot(lambdas, errors, linewidth=2)
        plt.show()

    def _render_progress_bar(self, reg_name, lambdas):
        self._pbars[reg_name] = tqdm(lambdas)

    def _update_progress_bar(self, reg_name, reg_val):
        self._pbars[reg_name].update()

    def __repr__(self):
        str_ = """Linear_Regression_Widget(
            model = {0}
        )""".format(self._title)
        return str_
