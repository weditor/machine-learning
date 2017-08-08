from read_data import read_doc, doc_to_vec
from sklearn.svm import SVC, NuSVC
from sklearn.externals import joblib
import numpy as np
import os


def create_svc():
    svc = SVC(kernel='linear', C=1.0, gamma='auto', tol=0.001)
    train_X = []
    train_Y = []
    # 构造两个分类的训练集。
    for doc in read_doc('corpus/jisuanji'):
        vec = doc_to_vec(doc)
        if vec:
            train_X.append(vec)
            train_Y.append(1)
    for doc in read_doc('corpus/huanjing'):
        vec = doc_to_vec(doc)
        if vec:
            train_X.append(vec)
            train_Y.append(-1)
    # print(len(train_X), len(train_X[0]))
    print('read doc finish')
    # print(len(train_X), train_Y)
    svc.fit(np.array(train_X), np.array(train_Y))
    print('fit finish')
    joblib.dump(svc, 'data/svc.model')
    return svc

#如果模型不存在，就训练一个模型，如果模型已经存在就直接读取。
if os.path.exists('data/svc.model'):
    print('reading svc model')
    svc = joblib.load('data/svc.model')
else:
    svc = create_svc()


# 使用训练好的模型进行预测
hj_test = []
for doc in read_doc('corpus/hj_test'):
    vec = doc_to_vec(doc)
    if vec and sum(vec) != 0:
        hj_test.append(vec)
print('hj  test, should all be -1', svc.predict(hj_test))

jsj_test = []
for doc in read_doc('corpus/jsj_test'):
    vec = doc_to_vec(doc)
    if vec and sum(vec) != 0:
        jsj_test.append(vec)
print('jsj test, should all be  1', svc.predict(jsj_test))