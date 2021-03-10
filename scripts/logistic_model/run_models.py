import numpy as np

from deletion import DeletionModel
from indel import IndelModel
from insertion import InsertionModel

data = np.loadtxt("../../cwd/Lindel_training.txt", delimiter="\t", dtype=str)


def model_progress_observer(reg_name, lambdas):
    print(f'{reg_name} training started')


def train():
    md = InsertionModel(data, save_directory="../../results/")
    md.add_observer(model_progress_observer)
    md.split_and_train(train_l2=False)

    dm = DeletionModel(data, save_directory="../../results/")
    dm.add_observer(model_progress_observer)
    dm.split_and_train(train_l2=False)

    idm = IndelModel(data, save_directory="../../results/")
    idm.add_observer(model_progress_observer)
    idm.split_and_train(train_l1=False)
