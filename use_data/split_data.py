# coding=utf-8

r"""
切分数据集为训练集、验证集和推理集，按照0.9:0.3:0.7的比例
"""

import csv
import os
from sklearn.model_selection import train_test_split


def split_data(file, out_path, max_len):
    """
    切分数据
    :param file:数据集
    :param out_path:输出路径
    :param max_len:序列限制最大长度
    :return:切分数据为训练集、验证集和推理集
    """
    f = csv.reader(open(file, 'r'))
    next(f)
    dias, cons = [],[]
    for line in f:
        dia = line[0] + ' ' + \
              line[1] + '\t' + \
              line[3] + ' ' + \
              line[4] + ' ' + \
              line[5] + ' ' + \
              line[6] + ' ' + \
              line[7] + ' ' + \
              line[8]  # 诊断文本和后面的个人信息用制表符隔开，以供不同模块使用
        if len(dia.split(' ')) > max_len:
            new_dia = ''
            for i in dia.split(' ')[:max_len]:
                new_dia = new_dia + ' ' + i
            dias.append(new_dia)
        else:
            dias.append(dia)

        con = line[12]
        cons.append(con)

    assert len(dias) == len(cons), '诊断文本和概念文本数量不一致！'

    print('诊断文本{}条=概念文本{}条'.format(len(dias), len(cons)))

    train_x, val_infer_x, train_y, val_infer_y = train_test_split(
        dias, cons, train_size=0.9, test_size=0.1, shuffle=True, random_state=2019)
    val_x, infer_x, val_y, infer_y = train_test_split(
        val_infer_x, val_infer_y, train_size=0.3, test_size=0.7, shuffle=True, random_state=2019)

    print('训练诊断{}条=训练概念{}条'.format(len(train_x), len(train_y)))
    print('验证诊断{}条=验证概念{}条'.format(len(val_x), len(val_y)))
    print('推理诊断{}条=概念{}条'.format(len(infer_x), len(infer_y)))

    with open(out_path + '\\train-dia.txt', mode='w', encoding='utf-8') as train_dia:
        for idx, i in enumerate(train_x):
            train_dia.write(i)
            if idx != len(train_x) - 1:
                train_dia.write('\n')
    with open(out_path + '\\train-con.txt', mode='w', encoding='utf-8') as train_con:
        for idx, i in enumerate(train_y):
            train_con.write(i)
            if idx != len(train_y) - 1:
                train_con.write('\n')

    with open(out_path + '\\val-dia.txt', mode='w', encoding='utf-8') as val_dia:
        for idx, i in enumerate(val_x):
            val_dia.write(i)
            if idx != len(val_x) - 1:
                val_dia.write('\n')
    with open(out_path + '\\val-con.txt', mode='w', encoding='utf-8') as val_con:
        for idx, i in enumerate(val_y):
            val_con.write(i)
            if idx != len(val_y) - 1:
                val_con.write('\n')

    with open(out_path + '\\infer-dia.txt', mode='w', encoding='utf-8') as infer_dia:
        for idx, i in enumerate(infer_x):
            infer_dia.write(i)
            if idx != len(infer_x) - 1:
                infer_dia.write('\n')
    with open(out_path + '\\infer-con.txt', mode='w', encoding='utf-8') as infer_con:
        for idx, i in enumerate(infer_y):
            infer_con.write(i)
            if idx != len(infer_y) - 1:
                infer_con.write('\n')

    print('该数据集切分完毕！')


if __name__ == '__main__':
    file = '.\\dd50\\dd50.csv'
    out_path = '.\\dd50'
    max_len = 50
    split_data(file, out_path, max_len)

    file = '.\\dd200\\dd200.csv'
    out_path = '.\\dd200'
    max_len = 50
    split_data(file, out_path, max_len)

    file = '.\\dp50\\dp50.csv'
    out_path = '.\\dp50'
    max_len = 50
    split_data(file, out_path, max_len)

    file = '.\\dp200\\dp200.csv'
    out_path = '.\\dp200'
    max_len = 50
    split_data(file, out_path, max_len)

