import pandas as pd
import numpy as np

from keras import optimizers, losses, activations, models
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler, ReduceLROnPlateau
from keras.layers import Dense, Input, Dropout, Convolution1D, MaxPool1D, GlobalMaxPool1D, GlobalAveragePooling1D, \
    concatenate, LSTM, BatchNormalization
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

df_1 = pd.read_csv("ptbdb_normal.csv", header=None)
df_2 = pd.read_csv("ptbdb_abnormal.csv", header=None)
df = pd.concat([df_1, df_2])

df_train, df_test = train_test_split(df, test_size=0.2, random_state=1337, stratify=df[187])


Y = np.array(df_train[187].values).astype(np.int8)
X = np.array(df_train[list(range(187))].values)[..., np.newaxis]

Y_test = np.array(df_test[187].values).astype(np.int8)
X_test = np.array(df_test[list(range(187))].values)[..., np.newaxis]



def get_model():
    number_labels = 1
    inp = Input(shape=(187, 1))

    forward = Convolution1D(filters=32, kernel_size=3, strides=1, padding='same', activation=activations.relu)(inp)
    forward = Convolution1D(filters=64, kernel_size=3, strides=1, padding='same', activation=activations.relu)(forward)
    forward = MaxPool1D(pool_size=2)(forward)
    forward = Dropout(rate=0.2)(forward)

    forward = LSTM(units=32, activation='tanh', return_sequences=True)(forward)
    forward = BatchNormalization()(forward)
    forward = LSTM(units=32, activation='tanh', return_sequences=False)(forward)
    forward = BatchNormalization()(forward)

    dense_1 = Dense(32, activation=activations.sigmoid, name="dense_1")(forward)
    dense_1 = Dense(32, activation=activations.sigmoid, name="dense_2")(dense_1)
    dense_1 = Dense(32, activation=activations.sigmoid, name="dense_3")(dense_1)
    dense_1 = Dense(32, activation=activations.sigmoid, name="dense_4")(dense_1)

    dense_1 = Dense(number_labels, activation=activations.sigmoid, name="dense_3_mitbih")(dense_1)

    model = models.Model(inputs=inp, outputs=dense_1)
    opt = optimizers.Adam(0.001)

    model.compile(optimizer=opt, loss=losses.binary_crossentropy, metrics=['acc'])
    model.summary()
    return model


model = get_model()

checkpoint = ModelCheckpoint('best_clf_ptbdb.h5', monitor='val_acc', verbose=1, save_best_only=True, mode='max')
early = EarlyStopping(monitor="val_acc", mode="max", patience=5, verbose=1)
redonplat = ReduceLROnPlateau(monitor="val_acc", mode="max", patience=3, verbose=2)

model.fit(X, Y, epochs=100, verbose=2, callbacks=[checkpoint, early, redonplat], validation_split=0.1)

pred_test = model.predict(X_test)
pred_test = (pred_test>0.5).astype(np.int8)

acc = accuracy_score(Y_test, pred_test)
print("Test accuracy score : %s "% acc)