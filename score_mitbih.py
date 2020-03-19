import tensorflow as tf
import numpy as np
import h5py
from keras.utils import np_utils
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import load_model
import pandas as pd
import os 
import yaml 
import argparse

path_csv = "./scores_mitbih.csv"
#Get paths dictionary 
parser = argparse.ArgumentParser()
parser.add_argument('--yaml',type=str)
args = parser.parse_args()
path_yaml = args.yaml

with open(path_yaml,'r') as f :
    paths = yaml.load(f, Loader=yaml.FullLoader)


path_mitbih = os.path.join(paths["MITBIH"]["Data"], "mitbih_test.csv")


#Get data
df_test = pd.read_csv(path_mitbih, header=None)
Y_test = np.array(df_test[187].values).astype(np.int8)
Y = Y_test
Y_test = np_utils.to_categorical(Y_test)

X_test = np.array(df_test[list(range(187))].values)[..., np.newaxis]
del df_test

names = list(paths["MITBIH"]["Models"].keys())
paths = list(paths["MITBIH"]["Models"].values())

models = []
accs = []
d = dict()


#Get models 
for p in paths :
    models.append(load_model(filepath=p))

#Get predictions for test dataset for every model 
for i in range(len(models)):
    predictions = models[i].predict(X_test)
    predictions = np.argmax(predictions,axis=-1)
    #Save accuracy
    d[names[i]] = accuracy_score(predictions, Y)

d = pd.DataFrame(d,index=["Accuracy"])
print(d)
#d.to_csv(path_csv)