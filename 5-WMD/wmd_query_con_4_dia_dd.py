# coding=utf-8

r"""
使用WMD算法，给dia匹配con
"""


import warnings
warnings.filterwarnings(action='ignore',category=UserWarning,module='gensim')
from gensim.models import Word2Vec
from gensim.similarities import WmdSimilarity
import time
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

def wmd_query_k_con_4_dia(train_dia, train_con, val_dia, val_con, infer_dia, infer_con, k_set, out_path):
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
    train_dia_f = [i.split('\t')[0].rstrip('\n') for i in train_dia_f]
    val_dia_f = open(val_dia, 'r', encoding='utf-8').readlines()
    val_dia_f = [i.split('\t')[0].rstrip('\n') for i in val_dia_f]
    infer_dia_f = open(infer_dia, 'r', encoding='utf-8').readlines()
    infer_dia_f = [i.split('\t')[0].rstrip('\n') for i in infer_dia_f]

    train_con_f = open(train_con, 'r', encoding='utf-8').readlines()
    train_con_f = [i.rstrip('\n') for i in train_con_f]
    val_con_f = open(val_con, 'r', encoding='utf-8').readlines()
    val_con_f = [i.rstrip('\n') for i in val_con_f]
    infer_con_f = open(infer_con, 'r', encoding='utf-8').readlines()
    infer_con_f = [i.rstrip('\n') for i in infer_con_f]

    all_dia = train_dia_f + val_dia_f + infer_dia_f
    all_dia = list(set(all_dia))
    all_dia_num = len(all_dia)
    all_con = train_con_f + val_con_f + infer_con_f
    all_con = list(set(all_con))
    all_con_num = len(all_con)
    all_infer_con = list(set(infer_con_f))
    all_infer_con_num = len(all_infer_con)

    corpus = all_dia + all_con
    corpus = [i.split(' ') for i in corpus]  # TODO 加载停用词
    corpus_dict = {}
    for idx, line in enumerate(corpus):
        corpus_dict[idx] = line

    w2v_model = Word2Vec(corpus, size=512, min_count=1, window=3, sg=0)  # sg=0--CBOW;1-skip-gram
    wmd_model = WmdSimilarity(corpus=corpus, w2v_model=w2v_model, num_best=len(corpus))

    y_pred_acc = [0 for _ in range(len(infer_con_f))]
    y_pred_acc_5 = [0 for _ in range(len(infer_con_f))]
    y_pred_acc_10 = [0 for _ in range(len(infer_con_f))]
    y_true = [1 for _ in range(len(infer_con_f))]
    result = open(out_path + 'result.txt', 'w', encoding='utf-8')
    for idx, query_dia in enumerate(tqdm(infer_dia_f, desc='推理中...', leave=False)):
        query_result = wmd_model[query_dia.split(' ')]

        candidate_con = {}
        for idx_value in query_result:
            if idx_value[0] > all_dia_num - 1:
                candidate_con[idx_value[0]] = idx_value[1]

        assert len(candidate_con) == all_con_num, '抽取出来的概念文本个数不是{}个！'.format(all_con_num)

        candidate_txt_value = {}
        for key, value in candidate_con.items():
            candidate_txt_value[' '.join(corpus_dict[key])] = value

        assert len(candidate_con) == all_con_num, '查询出来的候选概念数量和实际的候选概率数量不一致！！！'
        sort_candidate_txt_value = sorted(candidate_txt_value.items(), key=lambda x:x[1], reverse=True)

        max_10_con = [txt_value[0] for txt_value in sort_candidate_txt_value]

        if infer_con_f[idx] == max_10_con[0]:
            y_pred_acc[idx] = 1
        if infer_con_f[idx] in max_10_con[:5]:
            y_pred_acc_5[idx] = 1
        if infer_con_f[idx] in max_10_con[:10]:
            y_pred_acc_10[idx] = 1

    acc = accuracy_score(y_true, y_pred_acc)
    acc_5 = accuracy_score(y_true, y_pred_acc_5)
    acc_10 = accuracy_score(y_true, y_pred_acc_10)
    f1 = f1_score(y_true, y_pred_acc)

    end_time = time.time()
    take_time = end_time - start_time
    per_item_time = take_time / len(infer_con_f)

    print('{}数据集下，Acc={}，Acc@5={}，Acc@10={}，F1={}，总耗时{}，平均每条耗时{}'.
          format(data_set, round(acc, 3), round(acc_5, 3), round(acc_10, 3), round(f1, 3),
                 take_time, round(per_item_time, 2)))

    result.write('{}数据集下，Acc={}，Acc@5={}，Acc@10={}，F1={}，总耗时{}，平均每条耗时{}'.
          format(data_set, acc, acc_5, acc_10, f1, take_time, per_item_time))


if __name__ == "__main__":
    # data_set_dict = [['dd50', 50],
    #                  ['dd200', 100],
    #                  ['dp50', 50],
    #                  ['dp200', 100]]

    data_set_dict = [['dd50', 50],
                     ['dd200', 100]
                     ]

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

        wmd_query_k_con_4_dia(train_dia, train_con, val_dia, val_con, infer_dia, infer_con, k_set, output_path)
