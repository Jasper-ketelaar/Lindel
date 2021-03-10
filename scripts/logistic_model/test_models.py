import numpy as np
from tensorflow.keras.models import load_model, Sequential

from scripts.logistic_model.model import onehotencoder, mse


def mse_seq(weights, seq, x_in, y_out):
    w1, b1, w2, b2, w3, b3 = weights
    input_indel = onehotencoder(seq)
    input_ins = onehotencoder(seq[-6:])
    input_del = x_in

    dratio, insratio = softmax(np.dot(input_indel, w1) + b1)
    ds = softmax(np.dot(input_del, w2) + b2)
    ins = softmax(np.dot(input_ins, w3) + b3)
    y_hat = np.concatenate((ds * dratio, ins * insratio), axis=None)
    return mse(y_hat, y_out)


def softmax(weights):
    return np.exp(weights) / sum(np.exp(weights))


def run_test_set(test_file, test_path):
    # Best performing models on the Lindel training set are l1 for insertions and deletions and l2 for indel
    l1_insertion: Sequential = load_model(f'{test_path}/L1_insertion_model.h5')
    l1_deletion: Sequential = load_model(f'{test_path}/L1_deletion_model.h5')
    l2_indel: Sequential = load_model(f'{test_path}/l2_indel_model.h5')

    test_data = np.loadtxt(test_file, delimiter="\t", dtype=str)
    seqs = test_data[:, 0]
    float_data = test_data[:, 1:].astype('float32')
    x = float_data[:, :3033]
    y = float_data[:, 3033:]

    w1, b1 = l2_indel.get_weights()
    w2, b2 = l1_deletion.get_weights()
    w3, b3 = l1_insertion.get_weights()
    weights = w1, b1, w2, b2, w3, b3

    errors = []
    for i in range(len(seqs)):
        error = mse_seq(weights, seqs[i], x[i], y[i])
        errors.append(error)
    return errors
