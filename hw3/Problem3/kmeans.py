import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

num = 50

if __name__ == '__main__':
    data = np.loadtxt("Old.txt")[:, 1:3]
    label = KMeans(n_clusters=2).fit(data, 2).labels_
    arr1, arr2 = [], []
    for l, d in zip(label, data):
        if label[l] == 0:
            arr1.append(d)
        else:
            arr2.append(d)
    arr1, arr2 = np.array(arr1), np.array(arr2)
    plt.scatter(arr1[:, 0], arr1[:, 1], s=[num]*len(data), c='g', linewidth=0.1)
    plt.scatter(arr2[:, 0], arr2[:, 1], s=[num]*len(data), c='r', linewidth=0.1)
    plt.xlabel('eruptions length')
    plt.ylabel('waiting time')
    plt.savefig('eruption_waiting2')
