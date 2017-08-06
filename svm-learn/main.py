"""
一个简单的svm
"""
import numpy as np
from time import sleep
import random


def loadDataSet(fileName):
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat, labelMat


def selectJrand(i, m):
    j = i  # we want to select any J not equal to i
    while (j == i):
        j = int(random.uniform(0, m))
    return j


def clipAlpha(aj, H, L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj


def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
    dataMatrix = np.mat(dataMatIn)
    labelMat = np.mat(classLabels).transpose()
    b = 0
    m, n = np.shape(dataMatrix)
    alphas = np.mat(np.zeros((m, 1)))
    iter = 0
    while (iter < maxIter):
        alphaPairsChanged = 0
        for i in range(m):
            fXi = float(np.multiply(alphas, labelMat).T *
                        (dataMatrix * dataMatrix[i, :].T)) + b
            # if checks if an example violates KKT conditions
            Ei = fXi - float(labelMat[i])
            if ((labelMat[i] * Ei < -toler) and (alphas[i] < C)) or ((labelMat[i] * Ei > toler) and (alphas[i] > 0)):
                j = selectJrand(i, m)
                fXj = float(np.multiply(alphas, labelMat).T *
                            (dataMatrix * dataMatrix[j, :].T)) + b
                Ej = fXj - float(labelMat[j])
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()
                if (labelMat[i] != labelMat[j]):
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if L == H:
                    print("L==H")
                    continue
                eta = 2.0 * dataMatrix[i, :] * dataMatrix[j, :].T \
                    - dataMatrix[i, :] * dataMatrix[i, :].T \
                    - dataMatrix[j, :] * dataMatrix[j, :].T
                if eta >= 0:
                    print("eta>=0")
                    continue
                alphas[j] -= labelMat[j] * (Ei - Ej) / eta
                alphas[j] = clipAlpha(alphas[j], H, L)
                if (abs(alphas[j] - alphaJold) < 0.00001):
                    print("j not moving enough")
                    continue
                # update i by the same amount as j
                alphas[i] += labelMat[j] * \
                    labelMat[i] * (alphaJold - alphas[j])
                # the update is in the oppostie direction
                b1 = b - Ei - labelMat[i] * (alphas[i] - alphaIold) * dataMatrix[i, :] * dataMatrix[i, :].T \
                    - labelMat[j] * (alphas[j] - alphaJold) * \
                    dataMatrix[i, :] * dataMatrix[j, :].T
                b2 = b - Ej - labelMat[i] * (alphas[i] - alphaIold) * dataMatrix[i, :] * dataMatrix[j, :].T \
                    - labelMat[j] * (alphas[j] - alphaJold) * \
                    dataMatrix[j, :] * dataMatrix[j, :].T
                if (0 < alphas[i]) and (C > alphas[i]):
                    b = b1
                elif (0 < alphas[j]) and (C > alphas[j]):
                    b = b2
                else:
                    b = (b1 + b2) / 2.0
                alphaPairsChanged += 1
                print("iter: %d i:%d, pairs changed %d" %
                      (iter, i, alphaPairsChanged))
        if (alphaPairsChanged == 0):
            iter += 1
        else:
            iter = 0
        print("iteration number: %d" % iter)
    return b, alphas

def calc_w(data, label, alpha):
    data = np.mat(data)
    label = np.mat(label).transpose()
    alpha = np.mat(alpha)
    # print(alpha)
    # print(label)
    # print(np.multiply(alpha, label))
    return np.multiply(alpha[0], label[0]) * data


g_data, g_label = loadDataSet('simple_data')
g_b, g_alphas = smoSimple(g_data, g_label, 3, 0, 100)
g_alphas = g_alphas.T
g_w = calc_w(g_data, g_label, g_alphas[0])
print('b is', g_b)
print('alpha is', g_alphas)
print('w is', g_w)
# print(g_data)

def show_data(data, label, alpha, w, b):
    from matplotlib import pyplot

    supa = []
    supb = []

    for point, lb, a in zip(data, label, alpha.getA1()):
        style = 'g' if lb > 0 else 'r'
        if abs(a) > 0.000000001:
            style += 'o'
            if lb > 0:
                supa.append(point)
            else:
                supb.append(point)
        else:
            style += '.'
        pyplot.plot(point[0], point[1], style)

    def get_fx(_w, pa):
        # _w = (pb[1] - pa[1])/(pb[0] - pa[0])
        _b = pa[1] - _w*pa[0]
        return lambda x: _w*x + _b

    line_x = list(range(20))
    pyplot.plot(line_x, [x*w[0, 0]/w[0, 1]+b[0, 0] for x in line_x], 'b')
    pyplot.plot(line_x, [get_fx(w[0, 0]/w[0, 1], supa[0])(x) for x in line_x], 'g')
    pyplot.plot(line_x, [get_fx(w[0, 0]/w[0, 1], supb[0])(x) for x in line_x], 'r')
    pyplot.show()


show_data(g_data, g_label, g_alphas, g_w, g_b)

