import numpy as np
from numpy.random import seed
from sklearn.model_selection import train_test_split

seed(1)
import tensorflow
from tensorflow import set_random_seed
set_random_seed(2)

import pandas as pd
from keras.models import Sequential, load_model
from keras.layers import Dense, BatchNormalization
from keras.layers import Dropout
from keras.layers import LSTM
from keras.utils import np_utils
from keras.optimizers import Adam, Adagrad, Adadelta, RMSprop
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.utils import class_weight
from sklearn.metrics import accuracy_score


df_1 = pd.read_csv("ptbdb_normal.csv", header=None)
df_2 = pd.read_csv("ptbdb_abnormal.csv", header=None)
df = pd.concat([df_1, df_2])

df_train, df_test = train_test_split(df, test_size=0.2, random_state=1337, stratify=df[187])


Y = np.array(df_train[187].values).astype(np.int8)
X = np.array(df_train[list(range(187))].values)[..., np.newaxis]

Y_test = np.array(df_test[187].values).astype(np.int8)
X_test = np.array(df_test[list(range(187))].values)[..., np.newaxis]


X = np.reshape(X, (len(X), 187, 1))
X_test = np.reshape(X_test, (len(X_test), 187, 1))

weights = class_weight.compute_class_weight('balanced', np.unique(Y), Y)
weights_dict = {}
for i in range(2):
    weights_dict[i] = weights[i]
print(weights_dict)

# Y = np_utils.to_categorical(Y)
# Y_test = np_utils.to_categorical(Y_test)

print('Data shape: ', X.shape)
print('Y shape: ', Y.shape)

def get_model():
    model = Sequential()
    model.add(LSTM(64, input_shape=(187, 1), return_sequences=True))
    model.add(BatchNormalization())
    # model.add(Dropout(0.5))

    model.add(LSTM(64, return_sequences=True))
    model.add(BatchNormalization())
    # model.add(Dropout(0.5))

    model.add(LSTM(64, return_sequences=False))
    model.add(BatchNormalization())
    # model.add(Dropout(0.2))

    model.add(Dense(64, activation='sigmoid'))
    model.add(BatchNormalization())
    # model.add(Dropout(0.5))
    model.add(Dense(64, activation='sigmoid'))
    model.add(BatchNormalization())
    # model.add(Dropout(0.5))
    model.add(Dense(64, activation='sigmoid'))
    model.add(BatchNormalization())

    model.add(Dense(64, activation='sigmoid'))
    model.add(BatchNormalization())

    model.add(Dense(1, activation='sigmoid'))
    model.summary()

    return model



model = get_model()
opt = RMSprop(lr=0.001)

check = ModelCheckpoint('best_model.h5', monitor='val_acc', save_best_only=True, mode='max', verbose=2)
early = EarlyStopping(monitor='val_acc', mode='max', patience=10, verbose=2)
reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.2, patience=3, verbose=2)

model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['acc'])
model.fit(X, Y, epochs=100, batch_size=64, callbacks=[check, early, reduce_lr], validation_split=0.1)

predictions = model.predict(X_test)
predictions = (predictions > 0.5).astype(np.int8)

print(X_test.shape)
print(predictions.shape)
# predictions = np.argmax(predictions, axis=-1)


acc = accuracy_score(predictions, Y_test)
print("Test accuracy score : %s "% acc)

