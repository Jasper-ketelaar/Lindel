#!/usr/bin/env python

# System tools

# System tools
import pickle as pkl
import sys

import numpy as np
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2, l1
from tqdm import tqdm_notebook as tqdm

from LR_util import *


def lr_indel(datadir, fname):
    label, rev_index, features = pkl.load(open(datadir + 'feature_index_all.pkl', 'rb'))
    feature_size = len(features) + 384
    data = np.loadtxt(datadir + fname, delimiter="\t", dtype=str)
    Seqs = data[:, 0]
    data = data[:, 1:].astype('float32')

    # Sum up deletions and insertions to
    X = data[:, :feature_size]
    y = data[:, feature_size:]

    np.random.seed(121)
    idx = np.arange(len(y))
    np.random.shuffle(idx)
    X, y = X[idx], y[idx]
    train_size = round(len(data) * 0.9) if 'ForeCasT' in fname else 3900
    valid_size = round(len(data) * 0.1) if 'ForeCasT' in fname else 450

    Seq_train = Seqs[idx]
    x_train, x_valid = [], []
    y_train, y_valid = [], []
    for i in range(train_size):
        x_train.append(onehotencoder(Seq_train[i]))
        y_train.append((sum(y[i][:-21]), sum(y[i][-21:])))
    for i in range(train_size, len(Seq_train)):
        x_valid.append(onehotencoder(Seq_train[i]))
        y_valid.append((sum(y[i][:-21]), sum(y[i][-21:])))

    x_train, x_valid = np.array(x_train), np.array(x_valid)
    y_train, y_valid = np.array(y_train), np.array(y_valid)

    # Train model
    lambdas = 10 ** np.arange(-10, -1, 0.1)
    errors_l1, errors_l2 = [], []
    for l in tqdm(lambdas):
        np.random.seed(0)
        model = Sequential()
        model.add(Dense(2, activation='softmax', input_shape=(384,), kernel_regularizer=l2(l)))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['mse'])
        model.fit(x_train, y_train, epochs=100, validation_data=(x_valid, y_valid),
                  callbacks=[EarlyStopping(patience=1)], verbose=0)
        y_hat = model.predict(x_valid)
        errors_l2.append(mse(y_hat, y_valid))

    for l in tqdm(lambdas):
        np.random.seed(0)
        model = Sequential()
        model.add(Dense(2, activation='softmax', input_shape=(384,), kernel_regularizer=l1(l)))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['mse'])
        model.fit(x_train, y_train, epochs=100, validation_data=(x_valid, y_valid),
                  callbacks=[EarlyStopping(patience=1)], verbose=0)
        y_hat = model.predict(x_valid)
        errors_l1.append(mse(y_hat, y_valid))

    np.save(datadir + 'mse_l1_indel.npy', errors_l1)
    np.save(datadir + 'mse_l2_indel.npy', errors_l2)

    # final model
    l = lambdas[np.argmin(errors_l1)]
    np.random.seed(0)
    model = Sequential()
    model.add(Dense(2, activation='softmax', input_shape=(384,), kernel_regularizer=l1(l)))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['mse'])
    history = model.fit(x_train, y_train, epochs=100, validation_data=(x_valid, y_valid),
                        callbacks=[EarlyStopping(patience=1)], verbose=0)

    model.save(datadir + 'L1_indel.h5')

    l = lambdas[np.argmin(errors_l2)]
    np.random.seed(0)
    model = Sequential()
    model.add(Dense(2, activation='softmax', input_shape=(384,), kernel_regularizer=l2(l)))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['mse'])
    history = model.fit(x_train, y_train, epochs=100, validation_data=(x_valid, y_valid),
                        callbacks=[EarlyStopping(patience=1)], verbose=0)

    model.save(datadir + 'L2_indel.h5')
