# -*- encoding: utf-8 -*-
import math


class Weight:
    def __init__(self, cflag, globword):
        '''
        cflag记录所有分类的文章数: {cat1: num1, cat2: num2.....}
        globword记录每个词在每个分类出现的次数: {word1: {cat1: num1.1, cat2: num1.2}, word2: {cat1: num2.1, cat2: num2.2}}
        '''
        self.globword = globword
        self.cflag = cflag
        self.dnum = sum(self.cflag.values())

    def DF_ID1(self):
        df = {}
        for word in self.globword:
            wnum = sum(self.globword[word].values())
            df[word] = float(wnum) / self.dnum
        return df

    # 信息增益
    def IG_ID2(self):
        ig = {}
        for word, winfo in self.globword.items():
            wnum = sum(winfo.values())
            pt = sum([n / float(self.dnum) * math.log(n / float(wnum))
                      for cat, n in winfo.items()])
            pc = 0.0
            pt_ = 0.0
            for d in self.cflag:
                pc += self.cflag[d] * math.log(self.cflag[d])
                t_ = self.cflag[d] - winfo.get(d, 0)
                pt_ += (0 if t_ == 0 else (t_ / self.dnum) *
                        math.log(t_ / float(self.dnum - wnum)))
            ig[word] = pt + pt_ - pc
        return ig

    # 互信息
    def MI_ID3(self):
        mi = {}
        for word, winfo in self.globword.items():
            wnum = sum(winfo.values())
            mi[word] = sum(
                [math.log(self.dnum * winfo[c] / float(wnum * self.cflag[c])) for c in winfo])
        return mi

    # 期望交叉熵
    def ECE_ID4(self):
        ece = {}
        for word, winfo in self.globword.items():
            wnum = sum(winfo.values())
            ece[word] = sum([n * math.log(n * self.dnum) / float(self.cflag[c] * wnum) for c, n in winfo.items()])
        return ece

    # 文本证据权
    def WET_ID5(self):
        wet = {}
        for word, winfo in self.globword.items():
            wnum = sum(winfo.values())
            pt = wnum / float(self.dnum)
            score = 0.0
            for c, n in winfo.items():
                pci = self.cflag[c] / float(self.dnum)
                pci_t = pci / float(wnum)
                score += abs(pci * math.log((pci_t * (1 - pci) + 1) / (pci * (1 - pci_t) + 1)))
            wet[word] = pt*score
        return wet
