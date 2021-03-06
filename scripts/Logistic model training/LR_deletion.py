#!/usr/bin/env python

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



# Load data

def lr_deletion(datadir, fname):
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
        if 1 > sum(y[i, :536]) > 0:
            y_train.append(y[i, :536] / sum(y[i, :536]))
            x_train.append(X[i])
    for i in range(train_size, len(Seq_train)):
        if 1 > sum(y[i, :536]) > 0:
            y_valid.append(y[i, :536] / sum(y[i, :536]))
            x_valid.append(X[i])
    x_train, x_valid = np.array(x_train), np.array(x_valid)
    y_train, y_valid = np.array(y_train), np.array(y_valid)
    size_input = x_train.shape[1]
    
    # Train model
    lambdas = 10 ** np.arange(-10, -1, 0.1)
    errors_l1, errors_l2 = [], []
    for l in tqdm(lambdas):
        np.random.seed(0)
        model = Sequential()
        model.add(Dense(536, activation='softmax', input_shape=(size_input,), kernel_regularizer=l2(l)))
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['mse'])
        model.fit(x_train, y_train, epochs=100, validation_data=(x_valid, y_valid),
                  callbacks=[EarlyStopping(patience=1)], verbose=0)
        y_hat = model.predict(x_valid)
        errors_l2.append(mse(y_hat, y_valid))
    
    for l in tqdm(lambdas):
        np.random.seed(0)
        model = Sequential()
        model.add(Dense(536, activation='softmax', input_shape=(size_input,), kernel_regularizer=l1(l)))
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['mse'])
        model.fit(x_train, y_train, epochs=100, validation_data=(x_valid, y_valid),
                  callbacks=[EarlyStopping(patience=1)], verbose=0)
        y_hat = model.predict(x_valid)
        errors_l1.append(mse(y_hat, y_valid))
    
    np.save(datadir + 'mse_l1_del.npy', errors_l1)
    np.save(datadir + 'mse_l2_del.npy', errors_l2)
    
    # final model
    l = lambdas[np.argmin(errors_l1)]
    np.random.seed(0)
    model = Sequential()
    model.add(Dense(536, activation='softmax', input_shape=(size_input,), kernel_regularizer=l1(l)))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['mse'])
    history = model.fit(x_train, y_train, epochs=100, validation_data=(x_valid, y_valid),
                        callbacks=[EarlyStopping(patience=1)], verbose=0)
    
    model.save(datadir + 'L1_del.h5')
    
    l = lambdas[np.argmin(errors_l2)]
    np.random.seed(0)
    model = Sequential()
    model.add(Dense(536, activation='softmax', input_shape=(size_input,), kernel_regularizer=l2(l)))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['mse'])
    history = model.fit(x_train, y_train, epochs=100, validation_data=(x_valid, y_valid),
                        callbacks=[EarlyStopping(patience=1)], verbose=0)
    
    model.save(datadir + 'L2_del.h5')
