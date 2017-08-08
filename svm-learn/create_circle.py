"""
创建一个样例数据文件
"""
from random import randint, random


def randfloat(start, end):
    return random() * (end-start) + start


limit = 20
# 错误率
toler = 0.0
fp = open('circle_data', 'w')

for _ in range(30):
    i = randfloat(-limit, limit)
    j = randfloat(-limit, limit)
    need_wrong = (random() <= toler)
    print(need_wrong)
    if i**2 + j**2 >= 14.5**2 or ( i**2 + j**2 <= 9.5**2 and need_wrong):
        fp.write('%s\t%s\t1\n' % (i, j))
    elif i**2 + j**2 <= 13.5**2 or ( i**2 + j**2 >= 10.5**2 and need_wrong):
        fp.write('%s\t%s\t-1\n' % (i, j))

fp.close()

