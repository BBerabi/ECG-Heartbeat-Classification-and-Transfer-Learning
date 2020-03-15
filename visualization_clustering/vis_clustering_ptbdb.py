import numpy as np
import pandas as pd
import os.path
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from keras.models import load_model, Sequential


model = load_model('baseline_cnn_ptbdb.h5')

df_1 = pd.read_csv("ptbdb_normal.csv", header=None)
df_2 = pd.read_csv("ptbdb_abnormal.csv", header=None)
df = pd.concat([df_1, df_2])


Y = np.array(df[187].values).astype(np.int8)
X = np.array(df[list(range(187))].values)[..., np.newaxis]
print(X.shape)


representation_model = Sequential()
for layer in model.layers[:-1]:  # go through until last layer
    representation_model.add(layer)
representation_model.summary()

if os.path.isfile('ptbdb_representations.npy'):
    print('loading representations')
    embeddings = np.load('ptbdb_representations.npy')
else:
    embeddings = representation_model.predict(X)
    np.save('ptbdb_representations.npy', embeddings)

N = 5000
p = np.random.permutation(len(embeddings))
embeddings = embeddings[p]
Y = Y[p]
embeddings = embeddings[:N]
Y = Y[:N]

clustering = KMeans(n_clusters=2, random_state=42).fit(embeddings)

dimension = 2
transformer = TSNE(n_components=dimension, random_state=11, method='barnes_hut')
embeddings2d = transformer.fit_transform(embeddings)
print("shape after embedding: ", embeddings2d.shape)


# Plot
if dimension == 2:
    plt.scatter(embeddings2d[:, 0], embeddings2d[:, 1], c=clustering.labels_, alpha=0.5)
    plt.title('Visualization of learned representations')
    plt.legend(loc=2)

    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
else:
    fig = plt.figure()
    ax = Axes3D(fig)

    ax.scatter(embeddings2d[:, 0], embeddings2d[:, 1], embeddings2d[:, 2], c=clustering.labels_)
    plt.show()