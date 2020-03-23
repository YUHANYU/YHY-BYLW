# -*-coding:utf-8-*-

r"""Transformer模型训练和验证程序
"""

import argparse
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
from apex import amp, optimizers

from transformer.data_process import DataProcess
from transformer.utils import Criterion, SpecialOptimizer
from transformer.modules import Transformer
from transformer.model import Sequence2Sequence

import os
import sys
sys.path.append(os.path.abspath('.'))
from config import Config
config = Config()


def main():
    show_keyword(config)  # 显示关键参数

    data_obj = DataProcess(config.train_input, config.train_target, config.val_input, config.val_target)  # 读入数据

    src_tgt, src_lang, tgt_lang = data_obj.get_src_tgt_data()  # 输入序列、目标序列字符token处理
    *_, src_tgt_seq_train = data_obj.word_2_index(
        config.train_input, config.train_target, src_lang, tgt_lang)  # 训练输入-目标序列

    *_, src_tgt_seq_val = data_obj.word_2_index(
        config.val_input, config.val_target, src_lang, tgt_lang)  # 验证输入-目标序列

    words_data = {
        'src_lang':{
            'name': src_lang.name,
            'trimmed': src_lang.trimmed,
            'word2index': src_lang.word2index,
            'word2count': src_lang.word2count,
            'index2word': src_lang.index2word,
            'n_words': src_lang.n_words,
            'seq_max_len': src_lang.seq_max_len},
        'tgt_lang':{
            'name': tgt_lang.name,
            'trimmed': tgt_lang.trimmed,
            'word2index': tgt_lang.word2index,
            'word2count': tgt_lang.word2count,
            'index2word': tgt_lang.index2word,
            'n_words': tgt_lang.n_words,
            'seq_max_len': tgt_lang.seq_max_len},
        'data_obj':{
            'src_max_len': data_obj.src_max_len,
            'tgt_max_len': data_obj.tgt_max_len}}  # 保存训练、验证的输入-目标序列的token数据
    torch.save(words_data, config.save_data)

    train_data_loader = DataLoader(src_tgt_seq_train, config.batch_size, True, drop_last=False)  # 打包训练批次数据
    val_data_loader = DataLoader(src_tgt_seq_val, config.batch_size, False, drop_last=False)  # 打包验证批次数据

    transformer = Transformer(  # 定义transformer模型
        input_vocab_num=src_lang.n_words,
        target_vocab_num=tgt_lang.n_words,
        src_max_len=data_obj.src_max_len,
        tgt_max_len=data_obj.tgt_max_len).to(config.device)

    optimizer = SpecialOptimizer(  # 定义优化器
        optimizer=torch.optim.Adam(
            filter(lambda x: x.requires_grad, transformer.parameters()),
            betas=(0.9, 0.98),
            eps=1e-09),
        warmup_steps=config.warmup_step,
        d_model=config.d_model)

    criterion = Criterion()  # 定义损失函数

    seq2seq = Sequence2Sequence(
        transformer=transformer,
        optimizer=optimizer,
        criterion=criterion)  # 定义计算模型

    seq2seq.train_val(train_data_loader, val_data_loader)  # 模型训练-验证


def show_keyword(config):
    """
    根据参数器，输入一些关键字
    :param param: 参数类对象
    :return: 输入关键参数信息
    """
    print('数据集路径：', config.data_path)  # 打印保存根路径
    print('词向量维度为{}维，自注意力计算头数为{}头，编码器、解码器计算重复层数为{}层!'.
          format(config.d_model, config.heads, config.layers))
    print('词频最小出现次数限制为{}次，序列最大长度限制为{}个token，训练批大小为{}条，预热步为{}步，训练轮次为{}轮!\n'.
          format(config.min_word_count, config.max_len, config.batch_size, config.warmup_step, config.epochs))

if __name__ == '__main__':
    main()
