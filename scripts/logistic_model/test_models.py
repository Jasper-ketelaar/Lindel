import pickle as pkl

import numpy as np
from tensorflow.keras.models import load_model, Sequential

from scripts.generation.generate_profiles import ProfileGenerator
from scripts.logistic_model.model import one_hot_encoder, mse, softmax


def mse_seq(weights: tuple, seq: str, x_in: np.ndarray, y_out: np.ndarray):
    """
    This function takes the weights of all of the models and transforms the input to the correct format. It then
    computes the ratio of insertions and deletions by onehotencoding the sequence and taking the indel model output.

    Afterwards the deletion and insertion events are predicted using the other weights and biases and the models
    computed output becomes a concatenation of these two lists where the deletions are multiplied by the
    deletion ratio and the insertions are multiplied by the insertion ratio; just like in figure 6.A

    @param weights:
    @param seq:
    @param x_in:
    @param y_out:
    @return:
    """
    w1, b1, w2, b2, w3, b3 = weights
    input_indel = one_hot_encoder(seq)
    input_ins = one_hot_encoder(seq[-6:])
    input_del = x_in

    dratio, insratio = softmax(np.dot(input_indel, w1) + b1)
    ds = softmax(np.dot(input_del, w2) + b2)
    ins = softmax(np.dot(input_ins, w3) + b3)
    y_hat = np.concatenate((ds * dratio, ins * insratio), axis=None)
    return mse(y_hat, y_out), y_hat


def dump_model_wb(test_path: str) -> None:
    """
    Pickles the weights and biases of the models since these are the only valuable metrics once the models are trained,
    validated and tested.

    @param test_path: The path where the models are stored
    """
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


def run_test_set(test_file: str, test_path: str, generator: ProfileGenerator) -> (list, list, list):
    f"""
    Runs a test set on the serialized models combined. Best performing models on the Lindel training set are l1 for insertions
    and deletions and l2 for indel so these are the models that are loaded. All the sequences in the test set
    are evaluated and their error is computed by calling L{mse_seq} which performs the steps of the combined model from 
    the paper figure 6.A except for the combination of redundant classes.
    
    @param test_file: The test file path for which the sequences need to be evaluated.
    @param test_path: The path containing the serialized versions of the models
    @param generator: The ProfileGenerator instance that is used to check if the sequence is even in our generated 
        set of sequences
    @return: Three lists: 1) The mses for each sequence. 2) The expected output values. 3) The predicted output values
    """
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

        error, y_hat = mse_seq(weights, seqs[i], x[i], y[i])
        errors.append(error)
        ys.append(y[i])
        y_hats.append(y_hat)

    return errors, ys, y_hats


def run_aggregate(test_file: str, generator: ProfileGenerator) -> list:
    """
    Runs the aggregate model that simply takes the frequencies of each of the classes and normalizes them
    @param test_file: The test file that the aggregate model runs over
    @param generator: The ProfileGenerator that already contains the generated lindel profiles
    @return: The errors for each sequence in the test file aggregated in a list
    """
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
