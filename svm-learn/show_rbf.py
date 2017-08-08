import numpy as np
from simple_smo import loadDataSet, smoSimple
from svmMLiA import smoP, smoPK, optStruct, kernelTrans


g_data, g_label = loadDataSet('circle_data')
# g_b, g_alphas = smoP(g_data, g_label, 3, 0.1, 1000, kTup=('rbf', 0.6))
# g_alphas = g_alphas.T
# g_w = calc_w(g_data, g_label, g_alphas[0])
# print('b is', g_b)
# print('alpha is', g_alphas)
# print('w is', g_w)

# os = optStruct(np.mat(g_data), np.mat(g_label).transpose(), 3, 0.15, kTup=('rbf', 0.5))
def show_data(data, label):
    from matplotlib import pyplot

    supa = []
    supb = []
    # print('a is', data.getA())
    for point, lb in zip(data.getA(), label):
        style = 'g.' if lb > 0 else 'r.'
        pyplot.plot(point[0], point[1], style)

    def get_fx(_w, pa):
        _b = pa[1] - _w*pa[0]
        return lambda x: _w*x + _b

    line_x = list(range(20))
    # try:
    #     pyplot.plot(line_x, [x*tmp_w+b[0, 0] for x in line_x], 'k')
    # except:
    #     pass
    pyplot.show()

g_data_mat = np.mat(g_data)
g_data_m = np.shape(g_data)[0]

# print('k is', g_K)
print('data is', g_data)
# print('kdata is', g_K*g_data_mat)
# for rbfk in range(30):
# print(rbfk)
g_K = np.mat(np.zeros((g_data_m, g_data_m)))
for i in range(g_data_m):
    g_K[:, i] = kernelTrans(g_data_mat, g_data_mat[i,:], ('rbf', 300))
show_data(g_K*g_data_mat, g_label)
