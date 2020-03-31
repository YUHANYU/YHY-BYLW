# coding=utf-8

r"""
使用Doc2Vec的方法，对给定的诊断文本，连接对应的概念文本
"""

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import numpy as np
from sklearn.neighbors import KDTree
from sklearn.metrics import accuracy_score, f1_score
import time
from tqdm import tqdm


def load_data(train_dia, train_con, val_dia, val_con, infer_dia, infer_con):
    train_dia = open(train_dia, 'r', encoding='utf-8').readlines()
    # train_dia = [line.split('\t')[0].rstrip('\n') for line in train_dia]
    train_dia = [line.rstrip('\n') for line in train_dia]
    train_con = open(train_con, 'r', encoding='utf-8').readlines()
    train_con = [line.rstrip('\n') for line in train_con]

    val_dia = open(val_dia, 'r', encoding='utf-8').readlines()
    # val_dia = [line.split('\t')[0].rstrip('\n') for line in val_dia]
    val_dia = [line.rstrip('\n') for line in val_dia]
    val_con = open(val_con, 'r', encoding='utf-8').readlines()
    val_con = [line.rstrip('\n') for line in val_con]

    infer_dia = open(infer_dia, 'r', encoding='utf-8').readlines()
    # infer_dia = [line.split('\t')[0].rstrip('\n') for line in infer_dia]
    infer_dia = [line.rstrip('\n') for line in infer_dia]
    infer_con = open(infer_con, 'r', encoding='utf-8').readlines()
    infer_con = [line.rstrip('\n') for line in infer_con]

    return (train_dia, train_con), (val_dia, val_con), (infer_dia, infer_con)


def use_doc2vec(train, val, infer, dim):
    document = train[0] + train[1] + val[0] + val[1] + infer[0] + infer[1]
    document = list(set(document))
    document = [line.split(' ') for line in document]
    doc_idx = [TaggedDocument(doc, [idx]) for idx, doc in enumerate(document)]
    doc2vec_model = Doc2Vec(doc_idx, dm=0, vector_size=dim, window=1, min_count=1, workers=2, epochs=1000)

    infer_dia_vec = np.zeros((len(infer[0]), dim))
    for idx, dia in enumerate(infer[0]):
        dia = dia.split(' ')
        dia_vec = doc2vec_model.infer_vector(dia, steps=100)
        dia_vec = dia_vec.reshape(1, dia_vec.shape[0])
        infer_dia_vec[idx] = dia_vec

    all_con = train[1] + val[1] + infer[1]
    all_con = list(set(all_con))
    all_con_dict = {}
    for idx, con in enumerate(all_con):
        all_con_dict[idx] = con

    all_con_vec = np.zeros((len(all_con), dim))
    for idx, con in enumerate(all_con):
        con = con.split(' ')
        con_vec = doc2vec_model.infer_vector(con, steps=10)
        all_con_vec[idx] = con_vec
    all_con_vec_kdtree = KDTree(all_con_vec, metric='euclidean')

    y_true = [1 for _ in range(len(infer[1]))]
    y_pred = [0 for _ in range(len(infer[1]))]
    y_pred_5 = [0 for _ in range(len(infer[1]))]
    y_pred_10 = [0 for _ in range(len(infer[1]))]
    for idx in tqdm(range(len(infer[0])), desc='推理中...', leave=False):
        dia_vec = infer_dia_vec[idx].reshape(1, -1)
        _, query_con = all_con_vec_kdtree.query(dia_vec, k=10)
        query_con = [i for i in query_con[0]]
        query_con_txt = [all_con_dict[i] for i in query_con]

        if infer[1][idx] == query_con_txt[0]:
            y_pred[idx] = 1
        if infer[1][idx] in query_con_txt[:5]:
            y_pred_5[idx] = 1
        if infer[1][idx] in query_con_txt:
            y_pred_10[idx] = 1

    acc = accuracy_score(y_true, y_pred)
    acc_5 = accuracy_score(y_true, y_pred_5)
    acc_10 = accuracy_score(y_true, y_pred_10)
    f1 = f1_score(y_true, y_pred)
    end_time = time.time()
    take_time = end_time - start_time
    each_item_time = take_time / len(infer[0])

    print('Acc={},F-1={},A@5={},A@10={}，总时间={}，每条耗时={}'.
          format(round(acc, 3), round(f1, 3), round(acc_5, 3), round(acc_10, 3),
                 round(take_time, 2), round(each_item_time, 2)))


if __name__ == "__main__":
    data_set = [['dd50'],
                ['dd200'],
                ['dp50'],
                ['dp200']]

    for data in data_set:
        start_time = time.time()
        print('数据集{}'.format(data))
        data_set_path = '..\\use_data\\' + data[0] + '\\'
        out_path = '.\\save\\' + data[0] + '\\'

        train_dia = data_set_path + 'train-dia.txt'
        train_con = data_set_path + 'train-con.txt'

        val_dia = data_set_path + 'val-dia.txt'
        val_con = data_set_path + 'val-con.txt'

        infer_dia = data_set_path + 'infer-dia.txt'
        infer_con = data_set_path + 'infer-con.txt'

        train, val, infer = load_data(train_dia, train_con, val_dia, val_con, infer_dia, infer_con)
        use_doc2vec(train, val, infer, dim=512)


