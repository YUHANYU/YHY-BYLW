# coding=utf-8

r"""
R1模型的代码实现，总体的思路是先单词重写机制，后WR句子建模算法，对给定的诊断文本，粗筛选出k个候选概念文本
"""

import os
from Levenshtein import distance
import gensim
from sklearn.neighbors import KDTree
import numpy as np
import sys
sys.path.append('.\\')
from matching import MatchingModel
from sklearn.metrics import accuracy_score
import time


class RetrievalModel():
    """
    抽取子模型R1代码实现类，ReW+WR，对给定的诊断文本，粗筛选出k个候选概念
    """
    def __init__(self, k, bio_nlp, use_edit, use_word_emb, use_info=0):
        """
        初始化函数
        :param k: 候选概念个数
        """
        self.k = k
        self.bionlp = gensim.models.KeyedVectors.load_word2vec_format(bio_nlp, binary=True)
        self.cons = None
        use_edit_distance = use_edit
        use_word_embedding = use_word_emb
        use_info = use_info

    def read_file(self, train_dia_f, train_con_f, val_dia_f, val_con_f, infer_dia_f, infer_con_f):
        """
        读取文件
        :param train_dia_f:
        :param train_con_f:
        :param val_dia_f:
        :param val_con_f:
        :param infer_dia_f:
        :param infer_con_f:
        :return:
        """
        train_dia = open(train_dia_f, 'r', encoding='utf-8').readlines()
        train_con = open(train_con_f, 'r', encoding='utf-8').readlines()

        val_dia = open(val_dia_f, 'r', encoding='utf-8').readlines()
        val_con = open(val_con_f, 'r', encoding='utf-8').readlines()

        infer_dia = open(infer_dia_f, 'r', encoding='utf-8').readlines()
        infer_con = open(infer_con_f, 'r', encoding='utf-8').readlines()

        if use_info:  # 使用病人信息
            train_dia = [dia.replace('\t', ' ').rstrip('\n') for dia in train_dia]
            val_dia = [dia.replace('\t', ' ').rstrip('\n') for dia in val_dia]
            infer_dia = [dia.replace('\t', ' ').rstrip('\n') for dia in infer_dia]
        else:  # 不使用病人信息
            train_dia = [dia.split('\t')[0] for dia in train_dia]
            val_dia = [dia.split('\t')[0] for dia in val_dia]
            infer_dia = [dia.split('\t')[0] for dia in infer_dia]

        train_con = [con.rstrip('\n') for con in train_con]
        val_con = [con.rstrip('\n') for con in val_con]
        infer_con = [con.rstrip('\n') for con in infer_con]

        return (train_dia, train_con), (val_dia, val_con), (infer_dia, infer_con)

    def word_rewrite_mechanism(self, train, val, infer, edit_distance, query_word_len,
                               use_edit, use_word_emb):
        """
        单词重写机制，对诊断文本中的单词，如果不在概念文本形成的单词字典中，则替换；先是用编辑距离，不行在用词向量
        :param train:
        :param val:
        :param infer:
        :param edit_distance:
        :param query_word_len: 诊断文本中查询单词的长度限制
        :return: 经过清洗后的推理诊断文本集合
        """
        cons = train[1] + val[1] + infer[1]
        self.cons = list(set(cons))
        cons_word = []
        for con in cons:
            for word in con.rstrip('\n').split(' '):
                cons_word.append(word)
        cons_word = list(set(cons_word))[1:]
        cons_word_embedding = []
        idx_2_word = {}
        for idx, word in enumerate(cons_word):
            if len(word) > 0 and word in self.bionlp.vocab:
                cons_word_embedding.append(self.bionlp.wv[word])
                idx_2_word[idx] = word
        # print(idx_2_word)

        cons_word_kdtree = KDTree(np.mat(cons_word_embedding), metric='euclidean')

        infer_dia = [dia.rstrip('\n') for dia in infer[0]]  # 推理的诊断文本
        new_infer_dia = []
        for dia in infer_dia:
            new_dia = []
            for word in dia.split(' '):
                if word not in cons_word:  # 诊断文本的单词不在概念文本形成的单词字典中
                    every_distance = [distance(word, i) for i in cons_word]
                    if max(every_distance) <= edit_distance \
                            and len(word) > query_word_len\
                            and use_edit:  # 以编辑距离替换
                        new_dia.append(cons_word[every_distance.index([min(every_distance)])])

                    elif word in self.bionlp.vocab \
                            and len(word) > query_word_len\
                            and use_word_emb:  # 以词向量替换
                        word_embedding = np.mat(self.bionlp.wv[word]).reshape(1, -1)
                        _, query_result = cons_word_kdtree.query(word_embedding, k=1)
                        # print(query_result[0][0])
                        new_dia.append(idx_2_word[query_result[0][0]])

                    else:
                        new_dia.append(word)

                else:  # 单词在概念文本形成的单词字典中
                    new_dia.append(word)

            new_infer_dia.append(' '.join(new_dia))

        return new_infer_dia

    def wr(self, train, val, infer, new_infer_dias, output_path, k_set):
        """
        WR算法，给定一个推理诊断文本，返回k个候选的概念文本
        :return:
        """
        all_cons = train[1] + val[1] + infer[1]
        all_cons = list(set(all_cons))
        all_cons = [line.rstrip('\n').split(' ') for line in all_cons]
        wr_model = MatchingModel(all_cons, self.bionlp, False)

        y_pred = [0 for _ in range(len(infer[1]))]
        y_true = [1 for _ in range(len(infer[1]))]

        result_log = open(output_path + 'result_log.txt', 'w', encoding='utf-8')
        for k in k_set:
            output_file = open(output_path + 'k=' + str(k) + '.txt', 'w', encoding='utf-8')
            for idx, dia in enumerate(new_infer_dias):
                k_candidate_cons = wr_model.query(dia.split(' '), k)  # 得出k个候选概念及对应的余弦相似度

                k_txt = []
                for txt_p in k_candidate_cons:
                    k_txt.append(' '.join(txt_p[0]))

                    output_file.write(str(round(txt_p[1], 5)))
                    output_file.write('\t')
                    output_file.write(' '.join(txt_p[0]))
                    output_file.write('\n')

                if infer[1][idx] in k_txt:
                    y_pred[idx] = 1

            end_time = time.time()
            take_time = end_time - start_time
            each_items_time = take_time / len(infer[0])
            result_log.write('当k={}时，Cov值为{}，从头到尾的消耗时间为{}，每条耗时为{}.'.
                             format(k, round(accuracy_score(y_true, y_pred), 3),
                                    take_time, each_items_time))
            result_log.write('\n')
            print('当k={}时，Cov值为{}，从头到尾的消耗时间为{}，每条耗时为{}.'.
                             format(k, round(accuracy_score(y_true, y_pred), 3),
                                    take_time, each_items_time))

            output_file.close()
        result_log.close()


if __name__ == '__main__':
    data_set = [['dd50', 50],
                ['dd200', 100],
                ['dp50', 50],
                ['dp200', 100]]

    for set_k in data_set:
        print('数据集{}'.format(set_k[0]))

        k_set = [1] + [i for i in range(5, set_k[1], 5)]  # k的范围为[1, 5, 10, 15,...,45]
        data_set = '..\\use_data\\' + set_k[0] + '\\'
        output_path = '.\\save\\' + set_k[0] + '\\'
        train_dia_f = data_set + 'train-dia.txt'
        train_con_f = data_set + 'train-con.txt'
        val_dia_f = data_set + 'val-dia.txt'
        val_con_f = data_set + 'val-con.txt'
        infer_dia_f = data_set + 'infer-dia.txt'
        infer_con_f = data_set + 'infer-con.txt'
        edit_distance = 1  # 编辑距离
        bio_nlp = '..\\use_data\\bionlp.bin'
        dia_word_len = 5  # 诊断文本要被替换单词的长度限制
        use_edit = 1
        use_word_emb = 1
        use_info = 0  # 是否使用病人的个人信息

        start_time = time.time()
        r1 = RetrievalModel(set_k[1], bio_nlp, use_edit, use_word_emb, use_info)
        train, val, infer = r1.read_file(train_dia_f, train_con_f,
                                         val_dia_f, val_con_f,
                                         infer_dia_f, infer_con_f)
        new_infer_dia = r1.word_rewrite_mechanism(
            train, val, infer, edit_distance, dia_word_len, use_edit, use_word_emb)
        r1.wr(train, val, infer, new_infer_dia, output_path, k_set)
