import numpy as np

from deletion import DeletionModel
from indel import IndelModel
from insertion import InsertionModel


def model_progress_observer(reg_name: str, lambdas) -> None:
    """
    Simple observer to check how the training is coming along
    @param reg_name: Name of the type of regularization being used
    @param lambdas: Required parameter for the observer to be recognized
    """
    print(f'{reg_name} training started')


def train(work_dir: str = '../../cwd/') -> None:
    """
    Trains all the models based on the knowledge of which regularizers work best to save on some computation time

    @param work_dir: The directory in which the models should be saved as well as where the training set should be found
    """

    data = np.loadtxt(f"{work_dir}Lindel_training.txt", delimiter="\t", dtype=str)

    md = InsertionModel(data, save_directory=work_dir)
    md.add_observer(model_progress_observer)
    md.split_and_train(train_l2=False)

    dm = DeletionModel(data, save_directory=work_dir)
    dm.add_observer(model_progress_observer)
    dm.split_and_train(train_l2=False)

    idm = IndelModel(data, save_directory=work_dir)
    idm.add_observer(model_progress_observer)
    idm.split_and_train(train_l1=False)


if __name__ == '__main__':
    train()
