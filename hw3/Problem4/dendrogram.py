import scipy.spatial.distance as ssd
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import linkage



data = np.load("mds-population.npz")
populations = np.array([p.decode('UTF-8') for p in data['population_list']])
km = linkage(ssd.squareform(data['D']), 'ward', metric='precomputed')
dendrogram(km, labels=populations)
plt.ylabel("Nei Dist")
plt.savefig('_dendrogram')
