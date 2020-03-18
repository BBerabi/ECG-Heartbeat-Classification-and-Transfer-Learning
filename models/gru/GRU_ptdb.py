import numpy as np
from numpy.random import seed
from sklearn.model_selection import train_test_split

seed(1)
import tensorflow as tf
tf.random.set_seed(2)

import pandas as pd
from keras.models import Sequential, load_model
from keras.layers import Dense, SimpleRNN, BatchNormalization, Dropout, GRU
from keras.layers import Dropout
from keras.layers import LSTM
from keras.utils import np_utils
from keras.optimizers import Adam, RMSprop
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.utils import class_weight
from sklearn.metrics import accuracy_score
import os 
import yaml
import argparse 

with open("paths.yaml",'r') as f :
    paths = yaml.load(f, Loader=yaml.FullLoader)

parser = argparse.ArgumentParser()
parser.add_argument('--epoch',default=100,type=int)
args = parser.parse_args()
nr_epoch = args.epoch

df_1 = pd.read_csv("./data/ptbdb_normal.csv", header=None)
df_2 = pd.read_csv("./data/ptbdb_abnormal.csv", header=None)
df = pd.concat([df_1, df_2])

df_train, df_test = train_test_split(df, test_size=0.2, random_state=1337, stratify=df[187])

Y = np.array(df_train[187].values).astype(np.int8)
X = np.array(df_train[list(range(187))].values)[..., np.newaxis]

Y_test = np.array(df_test[187].values).astype(np.int8)
X_test = np.array(df_test[list(range(187))].values)[..., np.newaxis]


def get_model():
    model = Sequential()
    model.add(GRU(64, input_shape=(187, 1), return_sequences=True))
    model.add(BatchNormalization())
    model.add(GRU(64, return_sequences=True))
    model.add(BatchNormalization())
    model.add(GRU(64, return_sequences=False))
    model.add(BatchNormalization())

    model.add(Dense(64, activation='sigmoid'))
    model.add(BatchNormalization())
    model.add(Dense(32, activation='sigmoid'))
    model.add(BatchNormalization())
    model.add(Dense(16, activation='sigmoid'))
    model.add(BatchNormalization())
    model.add(Dense(8, activation='sigmoid'))
    model.add(BatchNormalization())
    model.add(Dense(4, activation='sigmoid'))
    model.add(BatchNormalization())

    model.add(Dense(1, activation='sigmoid'))


    model.summary()

    return model


model = get_model()
opt = Adam(lr=0.001)

path_model = os.path.join(os.path.dirname(paths['PTDB']['Models']['GRU']),'GRU_ptdb.h5')

check = ModelCheckpoint(path_model, monitor='val_acc', save_best_only=True, mode='max', verbose=2)
early = EarlyStopping(monitor='val_acc', mode='max', patience=10, verbose=2)
reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.2, patience=3, verbose=2)

model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['acc'])
model.fit(X, Y, epochs=nr_epoch, batch_size=64, callbacks=[check, early, reduce_lr],
          validation_split=0.1)

predictions = model.predict(X_test)
predictions = (predictions > 0.5).astype(np.int8)



acc = accuracy_score(predictions, Y_test)
print("Test accuracy score : %s " % acc)
