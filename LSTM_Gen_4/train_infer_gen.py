# coding=utf-8

r"""
联合LSTM-XXX模型的训练和预测阶段
"""

import torch
import os
import time
from tqdm import tqdm

from LSTM_Dot_4.train import train
from LSTM_Dot_4.language_loss_strategy import infer


class Config(object):
    def __init__(self, data_k):
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备选择
        self.attn_model = data_k[2]
        self.k = data_k[1]

        self.data_set = '..\\use_data\\' + data_k[0] + '\\'
        self.out_path = '.\\save\\' + data_k[0] + '\\'

        self.train_input = self.data_set + 'train-dia.txt'
        self.train_target = self.data_set + 'train-con.txt'
        self.val_input = self.data_set + 'val-dia.txt'
        self.val_target = self.data_set + 'val-con.txt'

        self.infer_input_k = self.out_path + 'infer-dia-' + str(data_k[1]) + '.txt'
        self.infer_target_k = self.out_path + 'infer-con-' + str(data_k[1]) + '.txt'
        self.infer_target = self.data_set + 'infer-con.txt'

        self.batch_size = 16
        self.epoch = 3

        self.save_word = self.out_path + 'word-dict.pt'

        self.min_len = 0  # 序列最短长度
        self.max_len = 50  # 序列最大长度
        self.min_word_count = 3  # 输入和目标字典中词频最少限制数
        self.pad = 0  # 序列填充符及位置
        self.unk = 1  # 序列unk未知token符及位置
        self.sos = 2  # 序列开始符及位置
        self.eos = 3  # 序列终止符及位置

        self.enc_dec_layer = 2
        self.emb_dim = 512
        self.dropout = 0.5

        self.lr = 0.00001
        self.grad_clip = 50.0


if __name__ == '__main__':
    attn_model = 'general'
    data_set_k = [['dd50', 25, attn_model],
                  ['dd200', 60, attn_model],
                  ['dp50', 30, attn_model],
                  ['dp200', 65, attn_model]]

    for data_k in data_set_k:
        config = Config(data_k)

        enc_checkpoint = config.out_path + 'enc.chkpt'
        if os.path.exists(enc_checkpoint):
            print('该路径文件下存在已有的模型保存文件，请检查是否需要重新训练！！！')
            print('如无终止，10s后自动开始进入训练-推理！')
            for _ in tqdm(range(10), desc='倒计时...'):
                time.sleep(1)
            train(config)
            infer(config)
        else:
            train(config)
            infer(config)
