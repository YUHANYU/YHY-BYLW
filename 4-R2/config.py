#-*-coding:utf-8-*-

"""
Transformer的内置参数类
注释中使用的字母代表的意思
B batch_size 批大小
L max_len 序列最大长度
D dimension = d_mode 模型维度
d dimension = d_q, d_k, d_v，三个变量query，key和value的维度 
H heads h次线性映射
"""

import torch

import os

class Config():

    def __init__(self, d_mode=512, d_q=64, heads=8, d_ff=2048, dropout=0.1, layers=6):
        """
        Transformer模型基本参数
        :param d_mode: 模型维度
        :param d_q: query维度
        :param heads: 计算头数
        :param d_ff: 中间层大小
        :param dropout: dropout大小
        :param layers: 层数
        """
        data_set = [['dd50', 25],
                    ['dd200', 60],
                    ['dp50', 30],
                    ['dp200', 65]]
        self.k = data_set[3][1]  # 候选概念个数，暂时dd50=25，dd200=60
        self.data_path = '..\\use_data\\' + data_set[3][0] + '\\'
        self.save_path = '.\\save\\' + data_set[3][0] + '\\'
        self.r1_path = os.path.abspath('..') + '\\3-R1\\save\\' + data_set[3][0] + '\\'

        self.train_input = self.data_path + 'train-dia.txt'  # 训练输入序列
        self.train_target = self.data_path + 'train-con.txt'  # 训练目标序列
        self.val_input = self.data_path + 'val-dia.txt'  # 验证输入序列
        self.val_target = self.data_path + 'val-con.txt'  # 验证目标序列

        self.infer_input = self.data_path + 'infer-dia.txt'
        self.infer_target = self.data_path + 'infer-con.txt'
        self.infer_input_k = self.save_path + 'infer-dia-' + str(self.k) + '.txt'
        self.infer_target_k = self.save_path + 'infer-con-' + str(self.k) + '.txt'
        self.candidate_k = self.r1_path + 'k=' + str(self.k) + '.txt'

        self.result_loss = self.save_path + 'result_loss.txt'

        self.d_model = d_mode  # 模型维度
        self.d_q = d_q  # query变量维度
        self.d_k = d_q  # key变量维度
        self.d_v = d_q  # value变量维度
        self.heads = heads  # 多头注意力中h次进行query，key和value的线性映射
        self.d_ff = d_ff  # 按位置前馈层的内层维度
        self.layers = layers  # 编码器和解码迭代计算层数
        self.dropout = dropout  # dropout大小

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备选择
        self.use_gpu = True if torch.cuda.is_available() else False  # 设备选择
        self.multi_gpu = False  # 多GPU

        self.min_len = 0  # 序列最短长度
        self.max_len = 50  # 序列最大长度
        self.min_word_count = 3  # 输入和目标字典中词频最少限制数
        self.pad = 0  # 序列填充符及位置
        self.unk = 1  # 序列unk未知token符及位置
        self.sos = 2  # 序列开始符及位置
        self.eos = 3  # 序列终止符及位置

        # batch_size和warmup_step的对应关系，按此表，才可使损失正常下降 32-16,000; 64-8,000; 128-4,000;
        self.batch_size = 32 * 2  # 训练批大小
        self.warmup_step = 8000 * 1 # 模型预热步
        self.epochs = 50  # 训练轮次
        self.loss_cal = 'sum'  # 损失是采用总和sum还是平均elementwise_mean，默认后者

        self.beam_search_left = 1  # beam search后留下top句子的数量
        self.beam_size = 5  # beam search搜索宽度
        self.infer_batch = 25  # 推理测试的批大小

        self.log = True  # 是否对训练、验证过程写日志
        self.save_trained_model = True  # 是否保存训练模型最佳点
        self.save_trained_model_type = 'best'  # 保存模型最佳点的方式
        self.save_model_checkpoint = self.save_path + 'best_model.chkpt'  # 模型最佳保存点
        self.save_data = self.save_path + 'words_data.pt'  # 输入、目标序列token数据保存点
        self.train_log = self.save_path + 'train-log.txt'  # 训练日志
        self.val_log = self.save_path + 'val-log.txt'  # 验证日志

        self.visual = True  # 训练&验证的loss，ppl，acc三个值的可视化
        self.use_visdom = False

        self.pic_train_loss = self.save_path + 'train_loss.png'  # 训练loss保存图片位置 eps/pdf
        self.pic_val_loss = self.save_path + 'val_loss.png'  # 验证loss保存图片位置 eps/pdf
        self.pic_train_val_ppl = self.save_path + 'train_val_ppl.png'  # 训练-验证PPL图片保存位置 eps/pdf
        self.pic_train_val_acc = self.save_path + 'train_val_acc.png'  # 训练-验证ACC图片保存位置 eps/pdf

        self.__check()  # 参数检查

    def __check(self):
        """
        检查d_model，d_q，d_k，d_v，heads，d_ff，这几者之间关系
        :return: True/False
        """
        assert self.d_ff / self.d_model == 4, '模型维度d_model和前馈层内层d_ff不匹配！'

        assert self.d_q == self.d_k or self.d_q == self.d_v or self.d_k == self.d_v, \
            '模型d_q，d_k，d_v三个变量维度维度不相等！'
