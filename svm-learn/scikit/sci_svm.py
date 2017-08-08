from sklearn.svm import SVC, NuSVC
# import sys
# sys.path.append('..')
# sys.path.append('../../')

import random
import numpy as np
from matplotlib import pyplot as plt
# from simple_smo import loadDataSet
# from svmMLiA import smoP, smoPK, optStruct, kernelTrans


def loadDataSet(fileName):
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat, labelMat


def show(data, label, predict):
    for point, lb, p in zip(data, label, predict):
        style = 'r' if lb > 0 else 'g'
        if lb == p:
            style += '.'
        else:
            style += 'x'
        plt.plot(point[0], point[1], style)
    plt.show()

# ldata, llabel = loadDataSet('simple_data')
ldata, llabel = loadDataSet('circle_data')
tdata, tlabel = loadDataSet('circle_data.1')
# X= np.array([[-1,-1],[-2,-1],[1,1],[2,1]])
# y = np.array([1,1,2,2])

svc = SVC(kernel='rbf')
svc.fit(ldata, llabel)

print(tlabel, tdata)
show(tdata, tlabel, (svc.predict(tdata)))
