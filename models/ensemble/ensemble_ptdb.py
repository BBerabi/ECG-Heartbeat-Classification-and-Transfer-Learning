import tensorflow as tf
import numpy as np
import h5py
from keras.utils import np_utils
from sklearn.metrics import accuracy_score, precision_recall_curve, roc_curve, auc
from tensorflow.keras.models import load_model
import pandas as pd


df_test = pd.read_csv("./data/mitbih_test.csv", header=None)
Y_test = np.array(df_test[187].values).astype(np.int8)
Y = Y_test
Y_test = np_utils.to_categorical(Y_test)

X_test = np.array(df_test[list(range(187))].values)[..., np.newaxis]
del df_test


paths = []
models = []


for p in paths :
    models.append(load_model(filepath=p))
print(models)


probabilities = np.zeros(Y_test.shape)

for m in models:
    probabilities += m.predict(X_test)
probabilities /= len(models)
predictions = np.argmax(probabilities, axis=-1)

acc = accuracy_score(predictions, Y)
precision, recall, thresholds = precision_recall_curve(Y_test,probabilities)
fpr, tpr, thresholds = roc_curve(Y_test,probabilities,pos_label=1)

aucroc = auc(fpr, tpr)
aucprc = auc(recall, precision)


print("=====ENSEMBLE MODEL=====")
print("Accuracy: ", acc)
print("AUCROC: ", aucroc)
print("AUCPRC: ", aucprc)