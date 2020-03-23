# coding=utf-8

r"""
使用bm25算法，对给定的诊断文本，查询k个候选概念
"""

from gensim.summarization import bm25  # 请运行在3.8.1及以上版本
from gensim.summarization.bm25 import get_bm25_weights
import math
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import normalize
from sklearn.neighbors import KDTree
import time
import heapq
import random
import numpy as np


class BM25(object):
    def __init__(self, docs):
        self.D = len(docs)
        self.avgdl = sum([len(doc) + 0.0 for doc in docs]) / self.D
        self.docs = docs
        self.f = []  # 列表的每一个元素是一个dict，dict存储着一个文档中每个词的出现次数
        self.df = {}  # 存储每个词及出现了该词的文档数量
        self.idf = {}  # 存储每个词的idf值
        self.k1 = 1.5
        self.b = 0.75
        self.init()

    def init(self):
        for doc in self.docs:
            tmp = {}
            for word in doc:
                tmp[word] = tmp.get(word, 0) + 1  # 存储每个文档中每个词的出现次数
            self.f.append(tmp)
            for k in tmp.keys():
                self.df[k] = self.df.get(k, 0) + 1
        for k, v in self.df.items():
            self.idf[k] = math.log(self.D - v + 0.5) - math.log(v + 0.5)

    def sim(self, doc, index):
        score = 0
        for word in doc:
            if word not in self.f[index]:
                continue
            d = len(self.docs[index])
            score += (self.idf[word] * self.f[index][word] * (self.k1 + 1)
                      / (self.f[index][word] + self.k1 * (1 - self.b + self.b * d
                                                          / self.avgdl)))
        return score

    def simall(self, doc):
        scores = []
        for index in range(self.D):
            score = self.sim(doc, index)
            scores.append(score)
        return scores


def bm25_query_k_con_4_dia(train_dia, train_con, val_dia, val_con, infer_dia, infer_con, k_set, out_path):
    """
    使用BM25算法，对给定的诊断文本，查询k个候选概念
    :param train_dia:
    :param train_con:
    :param val_dia:
    :param val_con:
    :param infer_dia:
    :param infer_con:
    :param k_set:
    :param out_path:
    :return:
    """
    start_time = time.time()
    train_dia_f = open(train_dia, 'r', encoding='utf-8').readlines()
    train_dia_f = [i.split('\t')[0] for i in train_dia_f]  # # 不使用病例信息，经过验证，加入会变差
    val_dia_f = open(val_dia, 'r', encoding='utf-8').readlines()
    val_dia_f = [i.split('\t')[0] for i in val_dia_f]  # 不使用病例信息，经过验证，加入会变差
    infer_dia_f = open(infer_dia, 'r', encoding='utf-8').readlines()
    infer_dia_f = [i.split('\t')[0] for i in infer_dia_f]  # 不使用病例信息，经过验证，加入会变差

    train_con_f = open(train_con, 'r', encoding='utf-8').readlines()
    train_con_f = [i.rstrip('\n') for i in train_con_f]
    val_con_f = open(val_con, 'r', encoding='utf-8').readlines()
    val_con_f = [i.rstrip('\n') for i in val_con_f]
    infer_con_f = open(infer_con, 'r', encoding='utf-8').readlines()
    infer_con_f = [i.rstrip('\n') for i in infer_con_f]

    all_dia = train_dia_f + val_dia_f + infer_dia_f
    # all_dia = list(set(all_dia))

    infer_dia_f_num = len(infer_dia_f)
    all_dia_num = len(all_dia)
    all_con = train_con_f + val_con_f + infer_con_f
    all_con = list(set(all_con))
    all_con_num = len(all_con)
    all_con_dict_idx2txt = {}
    for idx, txt in enumerate(all_con):
        all_con_dict_idx2txt[idx] = txt

    corpus = all_dia + all_con
    corpus = [i.split(' ') for i in corpus]
    corpus_dict = {}
    for idx, line in enumerate(corpus):
        corpus_dict[idx] = line

    # bm25_model = bm25.BM25(corpus)
    bm25_model_w = get_bm25_weights(corpus, n_jobs=2)
    # bm25_model = BM25(corpus)

    result_log = open(output_path + 'result-log.txt', 'w', encoding='utf-8')

    all_con_weight = bm25_model_w[-all_con_num:]
    assert len(all_con_weight) == all_con_num, \
        '概念权重的个数{}和全体概念的个数{}不一致！'.format(len(all_con_weight), all_con_num)
    all_con_w_kdtree = KDTree(normalize(np.mat(all_con_weight)), metric='euclidean')

    infer_dia_weight = bm25_model_w[-(infer_dia_f_num+all_con_num):-all_con_num]
    assert len(infer_dia_weight) == infer_dia_f_num, \
        '推理诊断权重个数{}与推理诊断个数{}不一致！'.format(len(infer_dia_weight), infer_dia_f_num)
    infer_dia_weight = normalize(infer_dia_weight)

    for k in k_set:
        output_pro_txt = open(output_path + 'k=' + str(k) + '.txt', 'w', encoding='utf-8')

        y_pred = [0 for _ in range(len(infer_con_f))]
        y_true = [1 for _ in range(len(infer_con_f))]

        for idx, query_dia in enumerate(infer_dia_weight):
            query_dia = np.mat(query_dia)
            query_pro, query_idx = all_con_w_kdtree.query(query_dia.reshape(1, -1), k)

            candidate_pro = [round(i, 5) for i in query_pro[0]]
            candidate_txt = [all_con_dict_idx2txt[i] for i in query_idx[0]]

            assert len(candidate_pro) == len(candidate_txt), '候选的概率和个数不一致！'

            if infer_con_f[idx] in candidate_txt:
                y_pred[idx] = 1

        acc = round(accuracy_score(y_true, y_pred), 3)
        end_time = time.time()
        use_time = round(end_time - start_time, 3)
        each_time = round(use_time / len(infer_con_f), 3)

        result_log.write('当k={}时，Cov值为{}，总消耗的时间为{}，每条消耗时间为{}。\n'.
                         format(k, acc, use_time, each_time))

        print('当k={}时，Cov值为{}，总消耗的时间为{}，每条消耗时间为{}。'.format(k, acc, use_time, each_time))

        output_pro_txt.close()
    result_log.close()


if __name__ == "__main__":
    data_set_dict = [['dd50', 50],
                     ['dd200', 100],
                     ['dp50', 50],
                     ['dp200', 100]]

    for set_k in data_set_dict:
        data_set = set_k[0]
        print('数据集为{}'.format(set_k[0]))
        k_set = [1] + [i for i in range(5, set_k[1], 5)]

        train_dia = '..\\use_data\\' + data_set + '\\train-dia.txt'
        train_con = '..\\use_data\\' + data_set + '\\train-con.txt'

        val_dia = '..\\use_data\\' + data_set + '\\val-dia.txt'
        val_con = '..\\use_data\\' + data_set + '\\val-con.txt'

        infer_dia = '..\\use_data\\' + data_set + '\\infer-dia.txt'
        infer_con = '..\\use_data\\' + data_set + '\\infer-con.txt'

        output_path = '.\\save\\' + data_set + '\\'

        bm25_query_k_con_4_dia(
            train_dia, train_con, val_dia, val_con, infer_dia, infer_con, k_set, output_path)

        print('\n')


