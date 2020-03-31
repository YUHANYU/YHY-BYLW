# coding=utf-8

r"""
使用TF-IDF的方法，对诊断文本，查询k个候选概念文本
"""

from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import accuracy_score
from sklearn.neighbors import BallTree
import time


def dia_query_k_con(train_dia, train_con, val_dia, val_con, infer_dia, infer_con, k_set, out_path):
    """
    给推理的诊断文本查询k个候选的推理概念文本
    :param train_dia: 训练的诊断文本
    :param train_con: 训练的概念文本
    :param val_dia: 验证的诊断文本
    :param val_con: 验证的概念文本
    :param infer_dia: 推理的诊断文本
    :param infer_con: 推理的概念文本
    :param k_set: k值集合
    :param out_path: 输出路径
    :return:
    """
    start_time = time.time()

    train_dia_f = open(train_dia, 'r', encoding='utf-8').readlines()
    train_dia_f = [i.split('\t')[0] for i in train_dia_f]  # 不使用病例信息，经过验证，加入会变差
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
    all_dia = list(set(all_dia))  # 去重后的所有诊断文本
    all_con = train_con_f + val_con_f + infer_con_f
    all_con = list(set(all_con))  # 去重后的所有概念文本
    all_dia_con = all_dia + all_con
    all_txt_len = len(all_dia_con)
    all_txt_word_num = []
    for line in all_dia_con:
        words = line.split(' ')
        for word in words:
            all_txt_word_num.append(word)
    all_txt_word_num = len(list(set(all_txt_word_num)))
    print('该数据集一共有{}条文本，{}个单词。'.format(all_txt_len, all_txt_word_num))

    vectorizer = CountVectorizer()
    transformer = TfidfTransformer()
    all_dia_con_tfidf = transformer.fit_transform(vectorizer.fit_transform(all_dia_con))  # 用所有的文本训练TF-IDF模型
    all_dia_con_w = all_dia_con_tfidf.toarray()  # 获取TF-IDF权重矩阵
    all_dia_con_w = normalize(all_dia_con_w)  # 归一化

    all_con_w = vectorizer.transform(all_con).toarray()  # 对所有的概念文本构建TF-IDF矩阵
    all_con_w = normalize(all_con_w)
    infer_dia_w = vectorizer.transform(infer_dia_f).toarray()  # 对推理的诊断文本构建TF-IDF矩阵
    infer_dia_w = normalize(infer_dia_w)

    y_pred = [0 for _ in range(len(infer_con_f))]
    y_true = [1 for _ in range(len(infer_con_f))]

    # all_con_w_balltree = BallTree(all_con_w, metric='euclidean')  # 概念树 欧式距离
    all_con_w_balltree = BallTree(all_con_w, metric='cosine')  # 概念树，余弦相似度

    result_log = open(out_path + 'result-log.txt', 'w', encoding='utf-8')
    for k in k_set:
        candidate_pro_txt = open(out_path + 'k=' + str(k) + '.txt', 'w', encoding='utf-8')
        for idx, query_dia in enumerate(infer_dia_w):
            query_pro, query_idx = all_con_w_balltree.query(query_dia.reshape(1, -1), k)

            candidate_pro = [round(i, 5) for i in query_pro[0]]
            candidate_txt = [all_con[i] for i in query_idx[0]]

            assert len(candidate_pro) == len(candidate_txt), '候选的概率和个数不一致！'

            for i in range(len(candidate_pro)):
                candidate_pro_txt.write(str(candidate_pro[i]))
                candidate_pro_txt.write('\t')
                candidate_pro_txt.write(str(candidate_txt[i]))
                candidate_pro_txt.write('\n')

            if infer_con_f[idx] in candidate_txt:
                y_pred[idx] = 1

        acc = round(accuracy_score(y_true, y_pred), 3)

        end_time = time.time()
        use_time = round(end_time - start_time, 3)
        each_time = round(use_time / len(infer_con_f), 3)

        result_log.write('当k={}时，Cov值为{}，总消耗的时间为{}，每条消耗时间为{}。\n'.format(k, acc, use_time, each_time))

        print('当k={}时，Cov值为{}，总消耗的时间为{}，每条消耗时间为{}。'.format(k, acc, use_time, each_time))

        candidate_pro_txt.close()
    result_log.close()


if __name__ == "__main__":
    data_set = [['dd50', 50],
                ['dd200', 100],
                ['dp50', 50],
                ['dp200', 100]]

    for set_k in data_set:
        print('数据集{}'.format(set_k[0]))
        data_set = '..\\use_data\\' + set_k[0] + '\\'
        output_path = '.\\save\\' + set_k[0] + '\\'
        k_set = [1] + [i for i in range(5, set_k[1], 5)]

        train_dia = data_set + 'train-dia.txt'
        train_con = data_set + 'train-con.txt'

        val_dia = data_set + 'val-dia.txt'
        val_con = data_set + 'val-con.txt'

        infer_dia = data_set + 'infer-dia.txt'
        infer_con = data_set + 'infer-con.txt'

        dia_query_k_con(train_dia, train_con, val_dia, val_con, infer_dia, infer_con, k_set, output_path)






