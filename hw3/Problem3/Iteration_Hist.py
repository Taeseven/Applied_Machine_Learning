import 3c_1
import numpy as np
import matplotlib.pyplot as plt

cnts = []
iteration = 50

if __name__ == '__main__':

    for i in range(iteration):
        e = code2.EM(np.loadtxt("Old.txt")[:, 1:3])
        cnts.append(e.get_count())
    plt.hist(cnts)
    plt.savefig('iterations_hist')
