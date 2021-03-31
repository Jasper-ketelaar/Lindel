import numpy as np
import pickle as pkl
from tensorflow.keras.models import load_model, Sequential

from scripts.generation.gen_mh_features import gen_cmatrix, gen_indel
from scripts.generation.matrix_load import ProfileGenerator
from scripts.logistic_model.model import onehotencoder, mse


def mse_seq(weights, seq, x_in, y_out, cmax):
    w1, b1, w2, b2, w3, b3 = weights
    input_indel = onehotencoder(seq)
    input_ins = onehotencoder(seq[-6:])
    input_del = x_in

    dratio, insratio = softmax(np.dot(input_indel, w1) + b1)
    ds = softmax(np.dot(input_del, w2) + b2)
    ins = softmax(np.dot(input_ins, w3) + b3)
    y_hat = np.concatenate((ds * dratio, ins * insratio), axis=None)
    return mse(y_hat, y_out), y_hat


def softmax(weights):
    return np.exp(weights) / sum(np.exp(weights))


def get_model(test_path):
    l1_insertion: Sequential = load_model(f'{test_path}/l1_insertion_model.h5')
    l1_deletion: Sequential = load_model(f'{test_path}/l1_deletion_model.h5')
    l2_indel: Sequential = load_model(f'{test_path}/l2_indel_model.h5')

    w1, b1 = l2_indel.get_weights()
    w2, b2 = l1_deletion.get_weights()
    w3, b3 = l1_insertion.get_weights()
    weights = w1, b1, w2, b2, w3, b3
    wf = open('../../results/weights.pkl', 'wb')
    pkl.dump(weights, wf)
    wf.close()

def run_test_set(test_file, test_path, generator: ProfileGenerator):
    # Best performing models on the Lindel training set are l1 for insertions and deletions and l2 for indel
    l1_insertion: Sequential = load_model(f'{test_path}/l1_insertion_model.h5')
    l1_deletion: Sequential = load_model(f'{test_path}/l1_deletion_model.h5')
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
    ys = []
    y_hats = []

    for i in range(len(seqs)):
        if seqs[i] not in generator.lindel_profiles:
            continue

        profile = generator.lindel_profiles[seqs[i]]
        seq = profile.get_mh_input()
        indels = gen_indel(seq, 30)
        cm = gen_cmatrix(indels, generator.labels_to_index)

        error, y_hat = mse_seq(weights, seqs[i], x[i], y[i], cm)
        errors.append(error)
        ys.append(y[i])
        y_hats.append(y_hat)

    return errors, ys, y_hats


def run_aggregate(test_file, generator: ProfileGenerator):
    test_data = np.loadtxt(test_file, delimiter="\t", dtype=str)
    seqs = test_data[:, 0]
    float_data = test_data[:, 1:].astype('float32')
    y = float_data[:, 3033:]

    errors = []
    y_sum = np.zeros(557)
    for profile in generator.lindel_profiles.values():
        y_sum += profile.labels_together()
    y_avg = y_sum / len(generator.lindel_profiles)

    for i in range(len(seqs)):
        y_out = y[i]
        errors.append(mse(y_avg, y_out))
    return errors
