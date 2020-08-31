import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import MDS
from collections import defaultdict

data = np.load("mds-population.npz")
clusters = defaultdict(list)
d = data['D']
populations = data['population_list']
mds = MDS(n_components=2, dissimilarity='precomputed', metric=True).fit_transform(d)
km = KMeans(n_clusters=5)
km.fit(mds)
labels = km.labels_
plt.scatter(mds[:, 0], mds[:, 1], c=labels, cmap='rainbow')
# print cluster
for idx, label in enumerate(labels):
    clusters[label].append(populations[idx])

for label, cluster in clusters.items():
    for population in cluster:
        print label, population

plt.savefig("cluster_result")
