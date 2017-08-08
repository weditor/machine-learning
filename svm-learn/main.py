"""
一个简单的svm
"""
import random
import numpy as np
from simple_smo import loadDataSet, smoSimple
from svmMLiA import smoP, smoPK, optStruct, kernelTrans

def calc_w(data, label, alpha):
    """
    w = alphai*yi*xi
    """
    data = np.mat(data)
    label = np.mat(label).transpose()
    alpha = np.mat(alpha)
    return np.multiply(alpha[0], label[0]) * data



def show_data(data, label, alpha, w, b):
    from matplotlib import pyplot

    supa = []
    supb = []
    print('a is', data.getA())
    for point, lb, a in zip(data.getA(), label, alpha.getA1()):
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
        _b = pa[1] - _w*pa[0]
        return lambda x: _w*x + _b

    line_x = list(range(20))
    try:
        tmp_w = w[0, 1]/w[0, 0]
        pyplot.plot(line_x, [x*tmp_w+b[0, 0] for x in line_x], 'k')
        pyplot.plot(line_x, [get_fx(tmp_w, supa[0])(x) for x in line_x], 'g')
        pyplot.plot(line_x, [get_fx(tmp_w, supb[0])(x) for x in line_x], 'r')
    except:
        pass
    pyplot.show()



# g_data, g_label = loadDataSet('simple_data')
g_data, g_label = loadDataSet('simple_data')
g_b, g_alphas = smoSimple(g_data, g_label, 3, 0.0, 100)
g_alphas = g_alphas.T
g_w = calc_w(g_data, g_label, g_alphas[0])
print('b is', g_b)
print('alpha is', g_alphas)
print('w is', g_w)
show_data(np.mat(g_data), g_label, g_alphas, g_w, g_b)
# os = optStruct(np.mat(g_data), np.mat(g_label).transpose(), 3, 0.15, kTup=('rbf', 0.5))
# g_data_mat = np.mat(g_data)
# g_data_m = np.shape(g_data)[0]
# g_K = np.mat(np.zeros((g_data_m, g_data_m)))
# for i in range(g_data_m):
#     g_K[:, i] = kernelTrans(g_data_mat, g_data_mat[i,:], ('rbf', 4))

# print('k is', g_K)
# print('data is', g_data)
# print('kdata is', g_K*g_data_mat)
# show_data(g_data_mat, g_label, g_alphas, g_w, g_b)

# g_w2 = g_alphas*np.mat(g_label).transpose()
# g_w2 = np.multiply(g_alphas[0], np.mat(g_label).transpose()[0])
# for point, pl in zip(g_data_mat.getA(), g_label):
#     print(point, (pl*(g_w2*g_K*(g_data_mat*np.mat(point).T) + g_b)) > 0 )
