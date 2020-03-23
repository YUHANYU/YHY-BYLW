# coding=utf-8

r"""
融合R1和R2模型
"""
from sklearn.metrics import accuracy_score, f1_score
import time


def merge(r1_result, r2_result, output_file, real_infer_con, k, data_set, max_alpha=100, max_beta=100):
    start_time = time.time()
    r1 = open(r1_result, 'r', encoding='utf-8').readlines()
    r2 = open(r2_result, 'r', encoding='utf-8').readlines()
    real = open(real_infer_con, 'r', encoding='utf-8').readlines()

    assert len(r1) == len(r2), 'R1预测结果与R2预测结果条数不一样！'
    assert len(r1) == len(real) * k, 'R1和R2的预测结果不是实际的{}倍！'.format(k)

    # 对R1和R2的数据做归一化
    new_r1 = []
    for line in r1:
        value, txt = line.split('\t')
        new_r1.append([float(value), txt.split('\n')[0]])
    r1_max = max([i[0] for i in new_r1])
    r1_min = min([i[0] for i in new_r1])
    for line in new_r1:
        line[0] = (line[0] - r1_min) / (r1_max - r1_min)

    new_r2 = []
    for line in r2:
        value, txt = line.split('\t')
        new_r2.append([float(value), txt.split('\n')[0]])
    r2_max = max([i[0] for i in new_r2])
    r2_min = min([i[0] for i in new_r2])
    for line in new_r2:
        line[0] = (line[0] - r2_min) / (r2_max - r2_min)

    txt_r1_r2_merge = [[] for _ in range(len(r1))]
    for idx in range(len(real)):
        r1_k = new_r1[idx*k:idx*k+k]
        r2_k = new_r2[idx*k:idx*k+k]

        txt_r1_dict = {}
        txt_r2_dict = {}

        for idx_k in range(k):
            txt_r1_dict[r1_k[idx_k][1]] = r1_k[idx_k][0]
            txt_r2_dict[r2_k[idx_k][1]] = r2_k[idx_k][0]

        for key_idx, key in enumerate(sorted(txt_r1_dict.keys())):
            txt_r1_r2_merge[idx*k+key_idx].append(key)
            txt_r1_r2_merge[idx*k+key_idx].append(txt_r1_dict[key])

        for key_idx, key in enumerate(sorted(txt_r2_dict.keys())):
            if key == txt_r1_r2_merge[idx*k+key_idx][0]:
                txt_r1_r2_merge[idx*k+key_idx].append(txt_r2_dict[key])
                # txt_r1_r2_merge[idx * k + key_idx].append(None)
            else:
                print('无法匹配同一个文本下的两个值！')
                exit()

    alpha_beta = []
    max_acc = []
    for alpha in range(1, 100, 1):
        for beta in range(1, 100, 1):
            for idx, i in enumerate(txt_r1_r2_merge):
                i.append(float((alpha/100)*i[1] + (beta/100)/(i[2]+1)))

            y_pred = [0 for _ in range(len(real))]
            y_true = [1 for _ in range(len(real))]
            for idx_2 in range(len(real)):
                merge_pro = txt_r1_r2_merge[idx_2*k:idx_2*k+k]
                sort_merge_pro = sorted(merge_pro, key=lambda x: x[3], reverse=True)  # 从大到小排列

                if real[idx_2].split('\n')[0] == sort_merge_pro[0][0]:
                    y_pred[idx_2] = 1

            acc = accuracy_score(y_true, y_pred)
            max_acc.append(acc)
            alpha_beta.append([alpha, beta])

            for j in txt_r1_r2_merge:
                j.remove(j[-1])

    a_b = alpha_beta[max_acc.index(max(max_acc))]
    print('动态加权投票法下最好性能的alpha={}，beta={}，acc={}'.format(a_b[0], a_b[1], max(max_acc)))
    for idx, i in enumerate(txt_r1_r2_merge):
        i.append(float((a_b[0] / 100) * i[1] + (a_b[1] / 100) / (i[2] + 1)))
    y_pred_acc = [0 for _ in range(len(real))]
    y_pred_acc_5 = [0 for _ in range(len(real))]
    y_pred_acc_10 = [0 for _ in range(len(real))]
    y_true = [1 for _ in range(len(real))]
    for idx_2 in range(len(real)):
        merge_pro = txt_r1_r2_merge[idx_2 * k:idx_2 * k + k]
        sort_merge_pro = sorted(merge_pro, key=lambda x: x[3], reverse=True)  # 从大到小排列

        _1_5_10 = [i[0] for i in sort_merge_pro[:10]]

        if real[idx_2].split('\n')[0] == _1_5_10[0]:
            y_pred_acc[idx_2] = 1
        if real[idx_2].split('\n')[0] in _1_5_10[:5]:
            y_pred_acc_5[idx_2] = 1
        if real[idx_2].split('\n')[0] in _1_5_10:
            y_pred_acc_10[idx_2] = 1

    acc = round(accuracy_score(y_true, y_pred_acc),5)
    acc_5 = round(accuracy_score(y_true, y_pred_acc_5), 5)
    acc_10 = round(accuracy_score(y_true, y_pred_acc_10), 5)
    f1 = round(f1_score(y_true, y_pred_acc), 5)

    end_time = time.time()
    take_time = end_time - start_time
    per_item_time = take_time / len(real)

    print('{}数据集下，当alpha={}， beta={}，Acc={}，Acc@5={}，Acc@10={}，F1={}，总耗时{}，平均每条耗时{}'.
          format(data_set, a_b[0], a_b[1], acc, acc_5, acc_10, f1, take_time, per_item_time))

    with open(output_file, 'w', encoding='utf-8') as file:
        file.write('{}数据集下，当alpha={}， beta={}，Acc={}，Acc@5={}，Acc@10={}，F1={}，总耗时{}，平均每条耗时{}'.
          format(data_set, a_b[0], a_b[1], acc, acc_5, acc_10, f1, take_time, per_item_time))

data_set = [['dd50', 25],
            ['dd200', 60],
            ['dp50', 30],
            ['dp200', 65]]

for set_k in data_set:
    print('数据集:', set_k[0])
    r1_result = '..\\3-R1\\save\\' + set_k[0] + '\\k=' + str(set_k[1]) + '.txt'
    r2_result = '..\\4-R2\\save\\' + set_k[0] + '\\result_loss.txt'
    real_infer_con = '..\\use_data\\' + set_k[0] + '\\infer-con.txt'
    output_file = '.\\' + set_k[0] + '.txt'
    merge(r1_result, r2_result, output_file, real_infer_con, set_k[1], set_k[0])
