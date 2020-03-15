import numpy as np
import pandas as pd
import os.path
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from keras.models import load_model, Sequential


model = load_model('baseline_cnn_mitbih.h5')

df_train = pd.read_csv("mitbih_train.csv", header=None)
df_train = df_train.sample(frac=1)

Y = np.array(df_train[187].values).astype(np.int8)
X = np.array(df_train[list(range(187))].values)[..., np.newaxis]

representation_model = Sequential()
for layer in model.layers[:-1]: # go through until last layer
    representation_model.add(layer)
representation_model.summary()

if os.path.isfile('mitbih_representations.npy'):
    print('loading representations')
    embeddings = np.load('mitbih_representations.npy')
else:
    embeddings = representation_model.predict(X)
    np.save('mitbih_representations.npy', embeddings)

N = 40000
p = np.random.permutation(len(embeddings))
embeddings = embeddings[p]
Y = Y[p]
embeddings = embeddings[:N]
Y = Y[:N]

clustering = KMeans(n_clusters=5, random_state=42).fit(embeddings)

dimension = 2
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
    # plt.show()
    plt.savefig('vis_cluster.png')
else:
    fig = plt.figure()
    ax = Axes3D(fig)

    ax.scatter(embeddings2d[:, 0], embeddings2d[:, 1], embeddings2d[:, 2], c=clustering.labels_)
    plt.show()