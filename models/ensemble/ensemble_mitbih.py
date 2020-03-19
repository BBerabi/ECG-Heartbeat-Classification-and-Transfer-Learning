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

parser = argparse.ArgumentParser()
parser.add_argument('--yaml',type=str)
args = parser.parse_args()

path_yaml = args.yaml

#Get paths dictionary 
with open(path_yaml,'r') as f :
    paths = yaml.load(f, Loader=yaml.FullLoader)

path_csv = "./models/ensemble/ENS_mitbih.csv"
path_test = os.path.join(paths["MITBIH"]["Data"], "mitbih_test.csv")

#Get data
df_test = pd.read_csv(path_test, header=None)
Y_test = np.array(df_test[187].values).astype(np.int8)
Y = Y_test
Y_test = np_utils.to_categorical(Y_test)

X_test = np.array(df_test[list(range(187))].values)[..., np.newaxis]
del df_test
#Get names and paths of models to be ensembled 
names = list(paths["MITBIH"]["Models"].keys())
paths = list(paths["MITBIH"]["Models"].values())

models = []

for p in paths :
    models.append(load_model(filepath=p))


predictions = np.zeros(Y_test.shape)

#Get predictions for test dataset 
for m in models:
    predictions += m.predict(X_test)
predictions /= len(models)
predictions = np.argmax(predictions, axis=-1)

#Compute Accuracy 
acc = accuracy_score(predictions, Y)

print("=====ENSEMBLE MODEL=====")
print("Accuracy: ", acc)
d = {"Ensemble":{"Accuracy":acc}}
d = pd.DataFrame(d)
#d.to_csv(path_csv)
#print(d)