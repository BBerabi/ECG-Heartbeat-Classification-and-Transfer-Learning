import numpy as np
import pandas as pd
import os.path
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from keras.models import load_model, Sequential



import os 
import yaml
#Get paths dictionary  
with open("paths.yaml",'r') as f :
    paths = yaml.load(f, Loader=yaml.FullLoader)

path_mitbih = os.path.join(paths["MITBIH"]["Data"], "mitbih_train.csv")
df_train = pd.read_csv(path_mitbih, header=None)
df_train = df_train.sample(frac=1)

#Load model 
model = load_model(paths["Baseline"]["MITBIH"])

Y = np.array(df_train[187].values).astype(np.int8)
X = np.array(df_train[list(range(187))].values)[..., np.newaxis]


representation_model = Sequential()
for layer in model.layers[:-1]: # go through until last layer
    representation_model.add(layer)
representation_model.summary()

path_mitbih = paths["Baseline"]["MITBIH_Representation"]

if os.path.isfile(path_mitbih):
    #Load the representations
    print('loading representations')
    embeddings = np.load(path_mitbih)
else:
    #Find the representations
    embeddings = representation_model.predict(X)
    np.save(path_mitbih, embeddings)

N = 40000
p = np.random.permutation(len(embeddings))
embeddings = embeddings[p]
Y = Y[p]
embeddings = embeddings[:N]
Y = Y[:N]

#Cluster representations
clustering = KMeans(n_clusters=5, random_state=42).fit(embeddings)

dimension = 2
#Visualize with tSNE
transformer = TSNE(n_components=dimension, random_state=11, method='barnes_hut')
embeddings2d = transformer.fit_transform(embeddings)
print("shape after embedding: ", embeddings2d.shape)

# v_score = v_measure_score(labels, clustering.predict(data_embedded))
# print("V score: ", v_score)
# nmi = normalized_mutual_info_score(labels, clustering.predict(data_embedded), average_method='geometric')
# print("NMI score: ", nmi)

# Plot
if dimension == 2:
    plt.scatter(embeddings2d[:, 0], embeddings2d[:, 1], c=clustering.labels_, alpha=0.5)
    plt.title('Visualization of learned representations')
    plt.legend(loc=2)

    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
    #plt.savefig('vis_cluster.png')
else:
    fig = plt.figure()
    ax = Axes3D(fig)

    ax.scatter(embeddings2d[:, 0], embeddings2d[:, 1], embeddings2d[:, 2], c=clustering.labels_)
    plt.show()