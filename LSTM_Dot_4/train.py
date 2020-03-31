# coding=utf-8

r"""
Seq2seq模型训练-验证实现
"""

import torch
from torch.utils.data import DataLoader
from torch import optim
from torch import nn
# from apex import amp

from LSTM_Dot_4.seq2seq.encoder import EncoderRNN
from LSTM_Dot_4.seq2seq.decoder import LuongAttnDecoderRNN
from LSTM_Dot_4.seq2seq.data_process import DataProcess
from LSTM_Dot_4.seq2seq.seq2seq import Seq2SeqModel


def train(config):
    print('数据集为{}, 注意力计算方式为{}，学习率为{}'.format(config.data_set, config.attn_model, config.lr))

    data_obj = DataProcess(
        config, config.train_input, config.train_target, config.val_input, config.val_target)  # 读入数据

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
    torch.save(words_data, config.save_word)

    train_data_loader = DataLoader(src_tgt_seq_train, config.batch_size, True, drop_last=False,
                                   pin_memory=True)  # 打包训练批次数据
    val_data_loader = DataLoader(src_tgt_seq_val, config.batch_size, False, drop_last=False,
                                 pin_memory=True)  # 打包验证批次数据

    encoder = EncoderRNN(
        src_lang.n_words,
        config.emb_dim,
        config.enc_dec_layer,
        config.dropout).to(config.device)
    decoder = LuongAttnDecoderRNN(
        config.attn_model,
        config.emb_dim,
        tgt_lang.n_words,
        config.enc_dec_layer,
        config.dropout).to(config.device)

    encoder_optimizer = optim.Adam(encoder.parameters(), config.lr)
    decoder_optimizer = optim.Adam(decoder.parameters(), config.lr * 5)

    seq2seq = Seq2SeqModel(encoder, decoder, encoder_optimizer, decoder_optimizer, config)
    seq2seq.train(train_data_loader, val_data_loader)