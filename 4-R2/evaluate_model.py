# coding=utf-8

r"""
测评R2模型
"""

from sklearn.metrics import accuracy_score, f1_score
import os

def evaluate_model(real_infer_tgt, pred_infer_target, k, output_file):
    real_infer_target = open(real_infer_tgt, 'r', encoding='utf-8').readlines()
    pre_infer_target = open(pred_infer_target, 'r', encoding='utf-8').readlines()

    y_pred_acc = [0 for _ in range(len(real_infer_target))]
    y_pred_acc_5 = [0 for _ in range(len(real_infer_target))]
    y_pred_acc_10 = [0 for _ in range(len(real_infer_target))]
    y_true = [1 for _ in range(len(real_infer_target))]

    assert len(real_infer_target) * k == len(pre_infer_target), '预测的序列不是实际序列的{}倍！'.format(k)

    for i in range(len(real_infer_target)):
        real_con = real_infer_target[i].split('\n')[0]

        pre_con = pre_infer_target[i * k:(i+1) * k]

        # print(i*k, (i+1)*k)

        pro_txt_dict = {}

        for line in pre_con:
            pro_txt_dict[float(line.split('\t')[0])] = line.split('\t')[1].split('\n')[0]

        pred_con_txt = []
        pred_con_pro = []
        for key in sorted(pro_txt_dict.keys()):
            pred_con_pro.append(key)
            pred_con_txt.append(pro_txt_dict[key])

        if real_con == pred_con_txt[0]:
            y_pred_acc[i] = 1
        if real_con in pred_con_txt[:5]:
            y_pred_acc_5[i] = 1
        if real_con in pred_con_txt[:10]:
            y_pred_acc_10[i] = 1

    acc = round(accuracy_score(y_true, y_pred_acc),5)
    acc_5 = round(accuracy_score(y_true, y_pred_acc_5), 5)
    acc_10 = round(accuracy_score(y_true, y_pred_acc_10), 5)
    f1 = round(f1_score(y_true, y_pred_acc), 5)

    print('Acc={},Acc@5={},Acc@10={},F1={}'.format(acc, acc_5, acc_10, f1))

    with open(output_file, 'w', encoding='utf-8') as out_f:
        out_f.write('Acc={},Acc@5={},Acc@10={},F1={}'.format(acc, acc_5, acc_10, f1))


data_set = [['dd50', 25],
            ['dd200', 60],
            ['dp50', 30],
            ['dp200', 65]]

for set_k in data_set:
    print('数据集:', set_k[0])
    real_infer_tgt = '..\\use_data\\' + set_k[0] + '\\infer-con.txt'
    pred_infer_tgt = '.\\save\\' + set_k[0] + '\\result_loss.txt'
    out_file = '.\\save\\' + set_k[0] + '\\r2-result.txt'
    evaluate_model(real_infer_tgt, pred_infer_tgt, set_k[1], out_file)


