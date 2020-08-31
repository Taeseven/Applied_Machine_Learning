import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from sklearn.cluster import KMeans
from pyclustering.cluster.kmedoids import kmedoids


# pip install pyclustering

data = np.load("mds-population.npz")
mds = MDS(n_components=2, dissimilarity='precomputed', metric=True).fit_transform(data['D'])
km_mds = KMeans(n_clusters=5).fit(mds)
labels = km_mds.labels_

plt.scatter(mds[:, 0], mds[:, 1], c=labels, cmap='rainbow')
plt.savefig("K-Means_cluster_ii")

kd = kmedoids(data['D'], range(10, 50, 10), data_type='distance_matrix')
kd.process()
kd_clusters = kd.get_clusters()
n = len(kd_clusters)

kmedoids_cluster = []
for j in xrange(n):
    kmedoids_cluster += [j]*len(kd_clusters[j])

for j in xrange(n):
    for i in kd_clusters[j]:
        kmedoids_cluster[i] = j

plt.scatter(mds[:, 0], mds[:, 1], c=kmedoids_cluster, cmap='rainbow')
plt.savefig("K-Medoids_cluster_ii")
