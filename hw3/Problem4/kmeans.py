import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial.distance as ssd
from sklearn.manifold import MDS
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import fcluster
from scipy.cluster.hierarchy import linkage


data = np.load("mds-population.npz")
mds = MDS(n_components=2, dissimilarity='precomputed', metric=True).fit_transform(data['D'])
km_mds = KMeans(n_clusters=5).fit(mds)
populations = np.array([p.decode('UTF-8') for p in data['population_list']])
km = linkage(ssd.squareform(data['D']), 'ward', metric='precomputed')
assignments = fcluster(km, 300, 'distance')
plt.scatter(mds[:, 0], mds[:, 1], c=assignments, cmap='rainbow')
plt.savefig("Hierarchy_Cluster")
plt.scatter(mds[:, 0], mds[:, 1], c=km_mds.labels_, cmap='rainbow')
plt.savefig("K-Means_Cluster")
