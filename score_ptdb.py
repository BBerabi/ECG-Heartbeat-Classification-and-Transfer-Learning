import tensorflow as tf
import numpy as np
import h5py
from keras.utils import np_utils
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

import pandas as pd

import os 
import yaml 

path_csv = "./scores_ptdb.csv"
with open("paths.yaml",'r') as f :
    paths = yaml.load(f, Loader=yaml.FullLoader)

path_normal = os.path.join(paths["PTDB"]["Data"], "ptbdb_normal.csv")
path_abnormal = os.path.join(paths["PTDB"]["Data"], "ptbdb_abnormal.csv")


df_1 = pd.read_csv(path_normal, header=None)
df_2 = pd.read_csv(path_abnormal, header=None)
df = pd.concat([df_1, df_2])

df_train, df_test = train_test_split(df, test_size=0.2, random_state=1337, stratify=df[187])
Y_test = np.array(df_test[187].values).astype(np.int8)
X_test = np.array(df_test[list(range(187))].values)[..., np.newaxis]


names = list(paths["PTDB"]["Models"].keys())
paths = list(paths["PTDB"]["Models"].values())
print("NAMES ",names)
print(paths)

models = []
d = dict()

for p in paths :
    models.append(load_model(filepath=p))
print(models)
for i in range(len(models)):

    probabilities = models[i].predict(X_test)
    predictions = (probabilities > 0.5).astype(np.int8)

    acc = accuracy_score(predictions, Y_test)

    precision, recall, thresholds = precision_recall_curve(Y_test,probabilities)
    fpr, tpr, thresholds = roc_curve(Y_test,probabilities,pos_label=1)

    aucroc = auc(fpr, tpr)
    aucprc = auc(recall, precision)

    d[names[i]] = {"Accuracy":acc,"AUCROC":aucroc,"AUCPRC":aucprc}
    print(d)


d = pd.DataFrame(d)
d.to_csv(path_csv)



