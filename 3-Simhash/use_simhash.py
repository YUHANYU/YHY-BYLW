# coding=utf-8

r"""
使用Simhash算法，给诊断文本，查找k个概念文本
"""

from simhash import Simhash
import re
import copy


import re
from simhash import Simhash
import time
import heapq
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import normalize


def get_features(s):
    width = 3
    s = s.lower()
    s = re.sub(r'[^\w]+', '', s)
    return [s[i:i + width] for i in range(max(len(s) - width + 1, 1))]


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


def use_simhash(infer, k_set):
    infer_dia, infer_con = infer[0], infer[1]
    infer_con = list(set(infer_con))
    infer_con_dict = {}
    for idx, con in enumerate(infer_con):
        infer_con_dict[idx] = con

    all_dia_con_simhash = []
    for dia in infer_dia:
        dia_con_simhash = []
        for con in infer_con:
            # dia_con_distance = Simhash(dia.split(' ')).distance(Simhash(con.split(' ')))
            dia_con_simhash = Simhash(get_features(dia)).distance(Simhash(get_features(con)))
            dia_con_simhash.append(dia_con_distance)
        all_dia_con_simhash.append(dia_con_simhash)
    new_all_dia_con_simhash = copy.deepcopy(all_dia_con_simhash)

    y_true = [1 for _ in range(len(infer[1]))]
    for k in k_set:
        y_pred = [0 for _ in range(len(infer[1]))]
        for idx in range(len(infer[0])):
            min_simhash_idx = []
            all_dia_con_simhash = copy.deepcopy(new_all_dia_con_simhash)
            for _ in range(k):
                min_value = min(all_dia_con_simhash[idx])
                min_simhash_idx.append(all_dia_con_simhash[idx].index(min_value))
                all_dia_con_simhash[idx].remove(min_value)

            candidate_txt = [infer_con_dict[i] for i in min_simhash_idx]

            if infer[1][idx] in candidate_txt:
                y_pred[idx] = 1

        acc = round(accuracy_score(y_true, y_pred), 3)
        f1 = round(f1_score(y_true, y_pred), 3)
        end_time = round(time.time(), 3)
        take_time = round(end_time - start_time, 3)
        each_item_time = round(take_time / len(infer[0]), 3)

        print('当k={}时，acc={}，f1={}，耗时={}，每条耗时={}'.format(k, acc, f1, take_time, each_item_time))




if __name__ == "__main__":
    data_set = [['dd50', 50],
                ['dd200', 100],
                ['dp50', 50],
                ['dp200', 100]]

    for data in data_set:
        k_set = [1] + [i for i in range(5, data[1], 5)]
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
        use_simhash(infer, k_set)
