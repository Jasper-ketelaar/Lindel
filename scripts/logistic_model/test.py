import numpy as np
from indel import IndelModel
from deletion import DeletionModel
from insertion import InsertionModel


data = np.loadtxt("../../cwd/Lindel_training.txt", delimiter="\t", dtype=str)
md = InsertionModel(data, save_directory="../../assets/")
md.start()
