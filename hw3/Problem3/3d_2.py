import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from sklearn.cluster import KMeans

num = 50

class EM:
    def __init__(self, data, k=2):
        self.k, self.data, self.avg, self.p = k, data, [], [1.0/2, 1.0/2]
        co = np.dot(self.data.T, self.data) / self.data.shape[0]
        self.co1 = np.array([co] * k)
        tmp = np.zeros((k, self.data.shape[1]))
        self.miui = np.mat(tmp)
        for i in range(self.data.shape[1]):
            self.miui[:, i] = (max(self.data[:, i]) - min(self.data[:, i])) * np.random.rand(k, 1)
        self.miui = np.array(self.miui)
    def plot(self):
        plt.scatter(self.data[:, 0], self.data[:, 1],linewidth=0.2, c='green', s=[50] * len(self.data))
        colors = ("b", "r")
        for i in range(2):
            trajectory = np.array([u[i, :] for u in self.avg])
            prev1, prev2 = None, None
            for now1, now2 in trajectory:
                if prev1 and prev2:
                    plt.quiver(prev1, prev2, now1-prev1, now2-prev2, color=colors[i], scale_units='xy',
                               scale=1, angles='xy')
                prev1, prev2 = now1, now2
    def get_count(self):
        cnt = 0
        gg = np.zeros((self.data.shape[0], self.k))
        for a in range(50):
            tmp = np.copy(self.miui)
            self.avg.append(tmp)
            gg1 = np.copy(gg)
            for i in range(self.data.shape[0]):
                de = 0
                for j in range(self.k):
                    py = multivariate_normal.pdf(self.data[i, :], mean=self.miui[j, :], cov=self.co1[j])
                    de += self.p[j] * py
                for j in range(self.k):
                    py = multivariate_normal.pdf(self.data[i, :], mean=self.miui[j, :], cov=self.co1[j])
                    gg[i, j] = self.p[j] * py / de
            if np.allclose(gg, gg1):
                break
            for j in range(self.k):
                miu0 = miu1 = co2 = co3 = 0
                for i in range(self.data.shape[0]):
                    miu0, miu1 = gg[i, j] + miu0, gg[i, j] * self.data[i, :] + miu1
                self.miui[j] = miu1/miu0
                for i in range(self.data.shape[0]):
                    co3, co2 = gg[i, j] + co3, gg[i, j] * (self.data[i, :] - self.miui[j]) ** 2 + co2
                self.co1[j] = co2/co3
                self.p[j] = (np.sum(gg[:, j])) / self.data.shape[0]
            cnt += 1
        return cnt

    def guess(self, avg1, avg2, co1, co2):
        self.miui, self.co1 = np.vstack((avg1, avg2)), np.array([co1, co2])


if __name__ == '__main__':
    data = np.loadtxt("Old.txt")[:, 1:3]
    cnts = []
    for i in range(num):
        label = KMeans(n_clusters=2).fit(data, 2).labels_
        arr1, arr2 = [], []
        for l, d in zip(label, data):
            if label[l] == 0:
                arr1.append(d)
            else:
                arr2.append(d)
        arr1, arr2 = np.array(arr1), np.array(arr2)
        e = EM(data)
        e.guess(np.mean(arr1, axis=0), np.mean(arr2, axis=0), np.dot((arr1 - np.mean(arr1, axis=0)).T, arr1 -
                np.mean(arr1, axis=0)) / arr1.shape[0], np.dot((arr2 - np.mean(arr2, axis=0)).T, arr2 -
                                                               np.mean(arr2, axis=0)) / arr2.shape[0])
        cnts.append(e.get_count())
    plt.hist(cnts)
    plt.savefig('iterations_hist2')

