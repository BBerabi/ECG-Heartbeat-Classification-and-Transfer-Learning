import numpy as np
from numpy.random import seed
seed(1)
import tensorflow
from tensorflow import set_random_seed
set_random_seed(2)


import pandas as pd
from keras.models import Sequential, load_model
from keras.layers import Dense, SimpleRNN, GRU
from keras.layers import Dropout
from keras.layers import LSTM
from keras.utils import np_utils
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.utils import class_weight
from sklearn.metrics import accuracy_score




df_train = pd.read_csv("./data/mitbih_train.csv", header=None)
df_train = df_train.sample(frac=1)
df_test = pd.read_csv("./data/mitbih_test.csv", header=None)

Y = np.array(df_train[187].values).astype(np.int8)
X = np.array(df_train[list(range(187))].values)[..., np.newaxis]

Y_test = np.array(df_test[187].values).astype(np.int8)
X_test = np.array(df_test[list(range(187))].values)[..., np.newaxis]

X = np.reshape(X, (len(X), 187, 1))

weights = class_weight.compute_class_weight('balanced', np.unique(Y), Y)
weights = dict(zip(list(range(5)), weights))

print(weights)
Y = np_utils.to_categorical(Y)

print('Data shape: ', X.shape)
print('Y shape: ', Y.shape)

def get_model():
    model = Sequential()
    model.add(GRU(64, input_shape=(187, 1), return_sequences=True))
    #model.add(Dropout(0.2))

    model.add(GRU(64, return_sequences=True))
    #model.add(Dropout(0.2))

    model.add(GRU(64, return_sequences=False))
    #model.add(Dropout(0.2))

    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))

    model.add(Dense(5, activation='softmax'))
    model.summary()

    return model





model = get_model()
opt = Adam(lr=0.001)

check = ModelCheckpoint('best_model_GRU_mitbih.h5', monitor='val_acc', save_best_only=True, mode='max', verbose=2)
early = EarlyStopping(monitor='val_acc', mode='max', patience=6, verbose=2)
reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.2, patience=3, verbose=2)

model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['acc'])
model.fit(X, Y, epochs=100, batch_size=64, callbacks=[check, early, reduce_lr],
          validation_split=0.1)

predictions = model.predict(X_test)
print(X_test.shape)
print(predictions.shape)
predictions = np.argmax(predictions, axis=-1)


acc = accuracy_score(predictions, Y_test)
print("Test accuracy score : %s "% acc)

