import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

variances = []
res = []
data = np.load("mds-population.npz")

for i in range(1,21):
    res.append(i)
    pca = PCA(n_components=i)
    pca.fit(data['D'])
    variances.append(sum(pca.explained_variance_ratio_))
for index, value in enumerate(variances):
    print "m = %d: sum of variance: %f" % (index+1, value)
plt.plot(res, variances)
plt.savefig('sum_of_variance')
