import gensim
import csv
from fse.inputs import IndexedList
from fse.models import SIF
import math
import numpy as np
from sklearn.preprocessing import normalize,scale,MinMaxScaler
from sklearn.neighbors import KDTree
import logging
import pandas as pd
import os


class MatchingModel():

    def __init__(self, all_concepts, bionlp_model,rewrite=False):
        self.corpus = []
        self.sens = []
        self.w2v_model = None
        self.se = None
        # self.file_path = file_path
        self.concepts = all_concepts
        # self.model_path = model_path
        self.w2v_model = bionlp_model
        self.rewrite_mode = rewrite

        self._read()
        self._train()

    def _read(self):
        # txt_or_csv = 'csv'
        # if txt_or_csv == 'txt':
        #     with open(self.file_path, 'r', encoding='utf-8') as file:
        #         for row in file:
        #             self.concepts.append(row.lstrip(' ').rstrip('\n'))
        # elif txt_or_csv == 'csv':
        #     with open(self.file_path, 'r', encoding='utf-8') as file:
        #         reader = csv.reader(file)
        #         next(reader)
        #         for row in reader:
        #             self.corpus.append(row[0].split())
        #             self.concepts.append(row[1])

        # self.concepts = [c.split() for c in list(set(self.concepts))]
        if self.rewrite_mode == True:
            # for procedure data set, the following 2 file paths should be replaced
            self.con_dic = dict(pd.read_csv('top200-con-dic-pro.csv'))
            self.dic = dict(pd.read_csv('top200-dic-pro.csv'))
            for key, value in self.con_dic.items():
                self.con_dic[key] = np.array(value)
            for key, value in self.dic.items():
                self.dic[key] = np.array(value)
            length = len(self.con_dic)
            X = np.ndarray(shape=(length, 200))
            self.words = []
            for key, i in zip(self.con_dic.keys(), range(length)):
                self.words.append(key)
                X[i] = self.con_dic[key]  # shape --> (length, 200)
            self.tree = KDTree(X)

        print('loading model...')
        # self.w2v_model = gensim.models.KeyedVectors.load_word2vec_format(self.model_path, binary=True)
        print('loading data...')

    def _train(self):
        self.sens = IndexedList(self.concepts)
        print('training SIF...')
        self.se = SIF(self.w2v_model)
        self.se.train(self.sens)

    def query(self, query_sen, topk=25):
        new_sen = query_sen

        if self.rewrite_mode == True:
            new_sen = []
            for w in query_sen:
                if w in self.dic and w not in self.con_dic:
                    q_emb = self.dic[w].reshape(1, 200)
                    dist, ind = self.tree.query(q_emb, k=1)
                    if dist[0][0] < 3.6:
                        # experiments shows that rewriting within a distance of 3.6 performs better than no-rewriting
                        index = ind.tolist()[0]
                        new_sen.append(self.words[index[0]])
                    else:
                        new_sen.append(w)
                else:
                    new_sen.append(w)

        # print(new_sen)
        cands = self.se.sv.similar_by_sentence(new_sen, model=self.se, topn=topk, indexable=self.sens.items)
        most_sim = [[x[0], x[2]] for x in cands]
        return most_sim


# src_path = os.path.abspath('..') + '\\data\\original_data\\split_data\\'
# train_con = src_path + 'dia-desc--pro\\full\\train-con.txt'  # 训练的concept文件
# infer_dia = src_path + '\\dia-desc--pro\\full\\infer-dia.txt'  # 测试的diagnosis文件
# bio_nlp = os.path.abspath('.') + '\\BioNLP-word-embedding.bin'  # 预训练的BioNLP词向量
#
# match_model = MatchingModel(train_con, bio_nlp, rewrite=True)
# with open(infer_dia, 'r', encoding='utf-8') as f:
#     for dia in f:
#         candi = match_model.query(dia.lstrip(' ').rstrip('\n').split(' '))
#         cons = [i[0] for i in candi]
#         cons = [' '.join(con) for con in cons]
#         print(len(cons), len(set(cons)))
#         if len(set(cons)) != len(cons):
#             for con_pro in candi:
#                 print(con_pro[0], con_pro[1])
#             break


