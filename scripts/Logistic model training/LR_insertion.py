#!/usr/bin/env python

#System tools 
import pickle as pkl
import os,sys,csv,re

from tqdm.notebook import tqdm
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Input, Flatten
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.regularizers import l2, l1

# Ignore KMP Duplicate errors due to OSX v
os.environ['KMP_DUPLICATE_LIB_OK']='True'
# Define useful functions
def mse(x, y):
    return ((x-y)**2).mean()

def corr(x, y):
    return np.corrcoef(x, y)[0, 1] ** 2

def onehotencoder(seq):
    nt= ['A','T','C','G']
    head = []
    l = len(seq)
    for k in range(l):
        for i in range(4):
            head.append(nt[i]+str(k))

    for k in range(l-1):
        for i in range(4):
            for j in range(4):
                head.append(nt[i]+nt[j]+str(k))
    head_idx = {}
    for idx,key in enumerate(head):
        head_idx[key] = idx
    encode = np.zeros(len(head_idx))
    for j in range(l):
        encode[head_idx[seq[j]+str(j)]] =1.
    for k in range(l-1):
        encode[head_idx[seq[k:k+2]+str(k)]] =1.
    return encode
# Load data
#threshold = sys.argv[1]


datadir  = sys.argv[1]
fname    = sys.argv[2]

label,rev_index,features = pkl.load(open(datadir + 'feature_index_all.pkl', 'rb'))
feature_size = len(features) + 384
data     = np.loadtxt(datadir + fname, delimiter="\t", dtype=str)
Seqs = data[:,0]
data = data[:,1:].astype('float32')

# Sum up deletions and insertions to
X = data[:,:feature_size]
y = data[:, feature_size:]

np.random.seed(121)
idx = np.arange(len(y))
np.random.shuffle(idx)
X, y = X[idx], y[idx]
train_size = round(len(data) * 0.9) if 'ForeCasT' in fname else 3900
valid_size = round(len(data) * 0.1) if 'ForeCasT' in fname else 450

print(train_size)
Seq_train = Seqs[idx]
x_train,x_valid = [],[]
y_train,y_valid = [],[]
for i in range(train_size):
    if 1> sum(y[i,-21:])> 0 :# 5 is a random number i picked if i use pred_size here it will be -21:0 it will just generate empty array
        y_train.append(y[i,-21:]/sum(y[i,-21:]))
        x_train.append(onehotencoder(Seq_train[i][-6:]))
for i in range(train_size,len(Seq_train)):
    if 1> sum(y[i,-21:])>0 :
        y_valid.append(y[i,-21:]/sum(y[i,-21:]))
        x_valid.append(onehotencoder(Seq_train[i][-6:]))

x_train,x_valid = np.array(x_train),np.array(x_valid)
y_train,y_valid = np.array(y_train),np.array(y_valid)
size_input = x_train.shape[1]


# Train model
lambdas = 10 ** np.arange(-10, -1, 0.1) # for activation function test
errors_l1, errors_l2 = [], []
for l in tqdm(lambdas):
    np.random.seed(0)
    model = Sequential()
    model.add(Dense(21,  activation='softmax', input_shape=(size_input,), kernel_regularizer=l2(l)))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['mse'])
    model.fit(x_train, y_train, epochs=100, validation_data=(x_valid, y_valid),
              callbacks=[EarlyStopping(patience=1)], verbose=0)
    y_hat = model.predict(x_valid)
    errors_l2.append(mse(y_hat, y_valid))

for l in tqdm(lambdas):
    np.random.seed(0)
    model = Sequential()
    model.add(Dense(21,  activation='softmax', input_shape=(size_input,), kernel_regularizer=l1(l)))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['mse'])
    model.fit(x_train, y_train, epochs=100, validation_data=(x_valid, y_valid),
              callbacks=[EarlyStopping(patience=1)], verbose=0)
    y_hat = model.predict(x_valid)
    errors_l1.append(mse(y_hat, y_valid))


np.save(datadir + 'mse_l1_ins.npy', errors_l1)
np.save(datadir + 'mse_l2_ins.npy', errors_l2)

# final model
l = lambdas[np.argmin(errors_l1)]
np.random.seed(0)
model = Sequential()
model.add(Dense(21, activation='softmax', input_shape=(size_input,), kernel_regularizer=l1(l)))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['mse'])
history = model.fit(x_train, y_train, epochs=100, validation_data=(x_valid, y_valid),
          callbacks=[EarlyStopping(patience=1)], verbose=0)

model.save(datadir + 'L1_ins.h5')


l = lambdas[np.argmin(errors_l2)]
np.random.seed(0)
model = Sequential()
model.add(Dense(21, activation='softmax', input_shape=(size_input,), kernel_regularizer=l2(l)))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['mse'])
history = model.fit(x_train, y_train, epochs=100, validation_data=(x_valid, y_valid),
          callbacks=[EarlyStopping(patience=1)], verbose=0)

model.save(datadir + 'L2_ins.h5')
