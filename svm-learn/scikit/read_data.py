
import os
import jieba
from weight import Weight
from collections import defaultdict
import jieba
import pickle


# word_info = {word1: {cat1: n1, cat2: n2}}
word_info = defaultdict(lambda: defaultdict(lambda: 0))
# cat_info = {cat1: num1}
cat_info = defaultdict(lambda: 0)

# def read_cat()
def read_doc(path):
    for info in os.walk(path):
        for filename in info[2]:
            # print(filename)
            lines = [line.strip() for line in open(os.path.join(path, filename), 'r', encoding='gb2312', errors='ignore')]
            yield ''.join(lines)


def calc_info():
    for doc in read_doc('corpus/jisuanji'):
        cat_info['jsj'] += 1
        for word in jieba.cut(doc, cut_all=False):
            word_info[word]['jsj'] += 1

    for doc in read_doc('corpus/huanjing'):
        cat_info['hj'] += 1
        for word in jieba.cut(doc, cut_all=False):
            word_info[word]['hj'] += 1

    wt = Weight(cat_info, word_info)

    word_list = []
    for word, score in sorted(wt.ECE_ID4().items(), key=lambda x: x[1], reverse=True):
        word_list.append(word)

    word_dict = {w: i for i, w in enumerate(word_list)}
    fp = open('data/info.pk', 'wb')
    pickle.dump(word_list, fp)
    pickle.dump(word_dict, fp)
    return word_list, word_dict


def read_word():
    if os.path.exists('data/info.pk'):
        fp = open('data/info.pk', 'rb')
        word_list = pickle.load(fp)
        word_dict = pickle.load(fp)
        return word_list, word_dict
    else:
        print('read info.pk')
        return calc_info()


def doc_to_vec(doc):
    winfo = defaultdict(lambda: 0)
    for word in jieba.cut(doc, cut_all=False):
        winfo[word] += 1
    cnt = sum(winfo.values())
    if cnt == 0:
        return None
    vec = [winfo[word]/float(cnt) for word in word_list]
    H = max(vec)
    return [n/H for n in vec]


word_list, word_dict = read_word()
word_list = word_list[:10000]
# print(word_list)
# for doc in read_doc('corpus/jisuanji'):
#     print(doc)

