import numpy as np
import matplotlib.pyplot as plt


data = np.loadtxt("Old.txt")
plt.scatter(data[:, 1], data[:, 2])
plt.xlabel('eruptions length')
plt.ylabel('waiting time')
plt.savefig('eruption_3b')
