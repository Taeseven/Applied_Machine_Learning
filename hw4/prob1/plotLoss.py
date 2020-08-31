import csv
import matplotlib.pyplot as plt

loss = []
with open('download.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        loss.append(float(row[0]))

plt.plot(loss)
plt.ylabel('Loss')
plt.xlabel('Iteration')
plt.savefig('loss1')
