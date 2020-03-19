import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from random import randint
import os 
import yaml
#Get paths dictionary 
with open("paths.yaml",'r') as f :
    paths = yaml.load(f, Loader=yaml.FullLoader)

def plot_mean_signals_and_one_random(data):
    number_of_samples = data.shape[0]
    number_of_features = data.shape[1]

    labels = data[:number_of_samples, -1]
    number_of_labels = np.unique(labels).shape[0]
    data = data[:, :number_of_features - 1]
    #Compute mean of signals for every class 
    mean_signals = []
    for label in range(number_of_labels):
        indices = np.where(labels == label)[0]
        number_of_class_samples = indices.shape[0]
        mean = np.sum(data[indices, :], axis=0) / number_of_class_samples
        mean_signals.append(mean)

    mean_signals = np.array(mean_signals)


    f, axarr = plt.subplots(2, number_of_labels, 'none')
    for i in range(number_of_labels):
        axarr[0, i].plot(pd.DataFrame(mean_signals[i]))
        axarr[0, i].title.set_text("Mean of " + str(i))

        indices = np.where(labels == i)[0]
        random_index = randint(0, indices.shape[0])
        signal_random = data[random_index, :]

        axarr[1, i].plot(pd.DataFrame(signal_random))
        axarr[1, i].title.set_text("Signal of " + str(i))

    plt.show()


path_normal = os.path.join(paths["PTDB"]["Data"], "ptbdb_normal.csv")
path_abnormal = os.path.join(paths["PTDB"]["Data"], "ptbdb_abnormal.csv")
#Get data
data_normal = pd.read_csv(path_normal).values
data_abnormal = pd.read_csv(path_abnormal).values
data_train = np.concatenate((data_normal, data_abnormal), axis=0)

path_mitbih = os.path.join(paths["MITBIH"]["Data"], "mitbih_train.csv")

print("Plot for PTBDB Dataset")
plot_mean_signals_and_one_random(data_train)
data_train = pd.read_csv(path_mitbih).values
print("Plot for MITBIH Dataset")
plot_mean_signals_and_one_random(data_train)


