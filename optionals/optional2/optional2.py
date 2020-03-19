
import pandas as pd
import numpy as np
#Add seed for reproducibility
from numpy.random import seed
seed(1)
import tensorflow as tf
tf.random.set_seed(2)

from keras import Sequential
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.layers import Dense, BatchNormalization, Dropout

from keras.models import load_model
from keras.optimizers import Adam, RMSprop
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import os 
import yaml 

#Get paths dictionary 
with open("new_paths.yaml",'r') as f :
    paths = yaml.load(f, Loader=yaml.FullLoader)

import argparse 
parser = argparse.ArgumentParser()
parser.add_argument('--model',type=str)
parser.add_argument('--epoch',default=100,type=int)
args = parser.parse_args()
nr_epoch = args.epoch

model_name = paths['MITBIH']['Models'][args.model]
print("Loading Model :",model_name)
base_model = load_model(model_name)
base_model.summary()

# load the data
path_normal = os.path.join(paths["PTDB"]["Data"], "ptbdb_normal.csv")
path_abnormal = os.path.join(paths["PTDB"]["Data"], "ptbdb_abnormal.csv")


df_1 = pd.read_csv(path_normal, header=None)
df_2 = pd.read_csv(path_abnormal, header=None)
df = pd.concat([df_1, df_2])

df_train, df_test = train_test_split(df, test_size=0.2, random_state=1337, stratify=df[187])

Y = np.array(df_train[187].values).astype(np.int8)
X = np.array(df_train[list(range(187))].values)[..., np.newaxis]

Y_test = np.array(df_test[187].values).astype(np.int8)
X_test = np.array(df_test[list(range(187))].values)[..., np.newaxis]



# Remove the last layers
model = Sequential()
for layer in base_model.layers[:-3]: # go through until last lstm layer
    model.add(layer)
model.summary()

#Define model 
# Add new output layers for ptbdb
model.add(Dense(units=64, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(units=64, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(units=64, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(units=64, activation='relu'))
model.add(BatchNormalization())

model.add(Dense(1, activation='sigmoid'))

model.summary()

#Configure model 
opt = Adam(lr=0.001)
model.compile(loss='binary_crossentropy', metrics=['acc'], optimizer=opt)

#Create string for path to save the model 
file_name = "OPT2_"+args.model+".h5"
path_model = os.path.join(os.path.dirname(paths['Optionals']['Optional_2']['Optional2_RNN']),file_name)

print("Best model will be saved in ",file_name)
checkpoint = ModelCheckpoint(path_model, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
early = EarlyStopping(monitor="val_acc", mode="max", patience=10, verbose=1)
redonplat = ReduceLROnPlateau(monitor="val_acc", mode="max", patience=7, verbose=2, factor=0.5)

model.fit(X, Y, epochs=nr_epoch, verbose=2, callbacks=[checkpoint, early, redonplat], validation_split=0.1,
          batch_size=64)

#Get predictions for test dataset 
pred_test = model.predict(X_test)
pred_test = (pred_test>0.5).astype(np.int8)

#Compute Accuracy 
acc = accuracy_score(Y_test, pred_test)
print("Test accuracy score : %s "% acc)
