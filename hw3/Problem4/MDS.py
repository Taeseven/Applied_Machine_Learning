import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS

data = np.load("mds-population.npz")
populations = data['population_list']
mds = MDS(n_components=2, dissimilarity='precomputed', metric=True).fit_transform(data['D'])
_, plot = plt.subplots()
plot.scatter(mds[:, 0], mds[:, 1], cmap="jet", alpha=0.5)
for index, value in enumerate(populations):
    plot.annotate(str(value),(mds[:, 0][index], mds[:, 1][index]))
plt.savefig("scatter_plot")
