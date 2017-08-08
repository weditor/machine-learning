"""
创建一个样例数据文件
"""
from random import randint, random


def randfloat(start, end):
    return random() * (end-start) + start


limit = 20
# 错误率
toler = 0.0
fp = open('simple_data', 'w')


for _ in range(30):
    i = randfloat(0, limit)
    j = randfloat(0, limit)
    need_wrong = (random() <= toler)
    print(need_wrong)
    if i - j >= 2 or (j - i >= 2 and need_wrong):
        fp.write('%s\t%s\t1\n' % (i, j))
    elif j - i >= 2 or ( i - j >= 2 and need_wrong):
        fp.write('%s\t%s\t-1\n' % (i, j))

fp.close()
