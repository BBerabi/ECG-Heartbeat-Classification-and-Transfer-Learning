import tensorflow as tf
import numpy as np
import h5py
from keras.utils import np_utils
from sklearn.metrics import accuracy_score, precision_recall_curve, roc_curve, auc
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
import pandas as pd
import os 
import yaml 

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--yaml',type=str)
args = parser.parse_args()

path_yaml = args.yaml

#Get paths dictionary 
with open(path_yaml,'r') as f :
    paths = yaml.load(f, Loader=yaml.FullLoader)

path_csv = "./models/ensemble/ENS2_ptdb.csv"
path_normal = os.path.join(paths["PTDB"]["Data"], "ptbdb_normal.csv")
path_abnormal = os.path.join(paths["PTDB"]["Data"], "ptbdb_abnormal.csv")

#Get data
df_1 = pd.read_csv(path_normal, header=None)
df_2 = pd.read_csv(path_abnormal, header=None)
df = pd.concat([df_1, df_2])

df_train, df_test = train_test_split(df, test_size=0.2, random_state=1337, stratify=df[187])
Y_test = np.array(df_test[187].values).astype(np.int8)
X_test = np.array(df_test[list(range(187))].values)[..., np.newaxis]

#Get names and paths of models to be ensembled 
names = ["BRNN","CNN_LSTM","LSTM"]

paths = [paths["PTDB"]["Models"][n] for n in names ] 


models = []


for p in paths :
    models.append(load_model(filepath=p))


probabilities = np.zeros((Y_test.shape[0],1))

#Get predictions for test dataset 
for m in models:
    probabilities += m.predict(X_test)
probabilities /= len(models)
predictions = (probabilities > 0.5).astype(np.int8)

#Compute Accuracy, AUROC, AUPRC
acc = accuracy_score(predictions, Y_test)
precision, recall, thresholds = precision_recall_curve(Y_test,probabilities)
fpr, tpr, thresholds = roc_curve(Y_test,probabilities,pos_label=1)

aucroc = auc(fpr, tpr)
aucprc = auc(recall, precision)


print("=====ENSEMBLE MODEL=====")
print("Accuracy: ", acc)
print("AUCROC: ", aucroc)
print("AUCPRC: ", aucprc)

d = {"Ensemble":{"Accuracy":acc,"AUCROC":aucroc,"AUCPRC":aucprc}}
d = pd.DataFrame(d)
#print(d)
#d.to_csv(path_csv)