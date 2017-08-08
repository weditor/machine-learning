from sklearn.svm import SVC, NuSVC
import numpy as np

# 构造数据
X= np.array([[-1,-1],[-2,-1],[1,1],[2,1]])
y = np.array([-1,-1, 1, 1])

# 数据集线性可分，使用最简单的线性核
svc = SVC(kernel='linear')
svc.fit(X, y)

testX = [[-5, -7], [-3, -1], [-7, -89], [5, 7], [3, 1], [7, 89]]
# 输出 testY = [-1, -1, -1, 1, 1, 1]
print(svc.predict(testX))

testY = [-1, -1, -1, 1, 1, 1]
def show(data, label, predict):
    from matplotlib import pyplot as plt
    for point, lb, p in zip(data, label, predict):
        style = 'r' if lb > 0 else 'g'
        if lb == p:
            style += '.'
        else:
            style += 'x'
        plt.plot(point[0], point[1], style)
    plt.show()
show(testX, testY, svc.predict(testX))
