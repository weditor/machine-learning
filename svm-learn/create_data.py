"""
创建一个样例数据文件
"""
from random import randint, random


def randfloat(start, end):
    return random() * (end-start) + start


limit = 20
fp = open('simple_data', 'w')
for _ in range(20):
    i = randfloat(0, limit)
    j = randfloat(0, limit)
    if i - j >= 2:
        fp.write('%s\t%s\t1\n' % (i, j))
    elif j - i >= 2:
        fp.write('%s\t%s\t-1\n' % (i, j))

fp.close()
