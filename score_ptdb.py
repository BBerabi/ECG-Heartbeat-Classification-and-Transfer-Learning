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

from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout

import pandas as pd
import os 
import yaml 

path_csv = "./scores_ptdb.csv"
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--yaml',type=str)
args = parser.parse_args()

path_yaml = args.yaml

#Get paths dictionary 
with open(path_yaml,'r') as f :
    paths = yaml.load(f, Loader=yaml.FullLoader)


path_normal = os.path.join(paths["PTDB"]["Data"], "ptbdb_normal.csv")
path_abnormal = os.path.join(paths["PTDB"]["Data"], "ptbdb_abnormal.csv")

#Get data
df_1 = pd.read_csv(path_normal, header=None)
df_2 = pd.read_csv(path_abnormal, header=None)
df = pd.concat([df_1, df_2])

df_train, df_test = train_test_split(df, test_size=0.2, random_state=1337, stratify=df[187])
Y_test = np.array(df_test[187].values).astype(np.int8)
X_test = np.array(df_test[list(range(187))].values)[..., np.newaxis]

#Get models' paths and names 
names = list(paths["PTDB"]["Models"].keys())
names.extend(list(paths["Optionals"]["Optional_1"].keys()))
names.extend(list(paths["Optionals"]["Optional_2"].keys()))
names.extend(list(paths["Optionals"]["Optional_3"].keys()))
#print("NAMES ",names)


path = list(paths["PTDB"]["Models"].values())
path.extend(list(paths["Optionals"]["Optional_1"].values()))
path.extend(list(paths["Optionals"]["Optional_2"].values()))
path.extend(list(paths["Optionals"]["Optional_3"].values()))

models = []
d = dict()

for p in path :
    models.append(load_model(filepath=p))

#Get predictions for test dataset 
for i in range(len(models)):
    #print("Getting scores of ",names[i])

    #Get representations for optional 1 
    if "Optional1" in names[i] : 
        #print("Proceeding with optional 1 ")

        #print("Getting base model")

        if "GRU" in names[i] :
            base_model = load_model(paths["MITBIH"]["Models"]["GRU"])
        elif "RNN" in names[i] :
            base_model = load_model(paths["MITBIH"]["Models"]["RNN"])
        elif "LSTM" in names[i] :
            base_model = load_model(paths["MITBIH"]["Models"]["LSTM"])
        
        representation_model = Sequential()
        for layer in base_model.layers[:-3]: # go through until last lstm layer
            representation_model.add(layer)
        
        X_test_representations = representation_model.predict(X_test)
        probabilities = models[i].predict(X_test_representations)
    else :
        probabilities = models[i].predict(X_test)

    predictions = (probabilities > 0.5).astype(np.int8)

    acc = accuracy_score(predictions, Y_test)

    precision, recall, thresholds = precision_recall_curve(Y_test,probabilities)
    fpr, tpr, thresholds = roc_curve(Y_test,probabilities,pos_label=1)

    aucroc = auc(fpr, tpr)
    aucprc = auc(recall, precision)
    #Save accuracy, AUROC, AUPRC
    d[names[i]] = {"Accuracy":acc,"AUCROC":aucroc,"AUCPRC":aucprc}
    


d = pd.DataFrame(d)
print(d)
#d.to_csv(path_csv)



