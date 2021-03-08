import os
import pickle as pkl
from os import sep

import numpy as np
from IPython import display as ids
from ipywidgets import Output

from scripts.logistic_model.notebook.file_chooser import FileChooser


class FeatureLoader:

    @property
    def training_chooser(self) -> FileChooser:
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
        """
        304 features for 2bp insertions, 80 features for 1bp insertions, 2649 binary features for deletion events
        @return:
        """
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
        self._file_chooser = FileChooser(
            title="Select the set to run the linear regression on:",
            path=os.getcwd(),
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
            pkl_chooser = FileChooser(
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
