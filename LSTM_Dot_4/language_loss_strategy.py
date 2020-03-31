# coding=utf-8

r"""
让LSTM-Add预训练好的模型执行语义损失策略
"""

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable

import time
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score

from LSTM_Dot_4.seq2seq.encoder import EncoderRNN
from LSTM_Dot_4.seq2seq.decoder import LuongAttnDecoderRNN
from LSTM_Dot_4.seq2seq.data_process import DataProcess
from LSTM_Dot_4.seq2seq.seq2seq import Seq2SeqModel
from LSTM_Dot_4.seq2seq.loss_func import masked_cross_entropy
from LSTM_Dot_4.seq2seq.data_process import Tools


def infer(config):
    print('数据集{}，输出路径{}。'.format(config.data_set, config.out_path))

    word_data = torch.load(config.save_word)  # 加载保存的训练-验证的输入、目标序列的token信息
    src_lang = word_data['src_lang']  # 输入序列token信息
    tgt_lang = word_data['tgt_lang']  # 目标序列token信息
    # data_obj = word_data['data_obj']  # 数据对象

    write_info = []
    start_time = time.time()

    enc_checkpoint = torch.load(  # 载入encoder保存点
        config.out_path + 'enc.chkpt',
        map_location=lambda storage, loc:storage.cuda(0))
    encoder = EncoderRNN(
        src_lang['n_words'],
        config.emb_dim,
        config.enc_dec_layer,
        config.dropout).to(config.device)
    encoder.load_state_dict(enc_checkpoint['model'])
    encoder.eval()

    dec_checkpoint = torch.load(  # 载入decoder保存点
        config.out_path + 'dec.chkpt',
        map_location=lambda storage, loc:storage.cuda(0))
    decoder = LuongAttnDecoderRNN(
        config.attn_model,
        config.emb_dim,
        tgt_lang['n_words'],
        config.enc_dec_layer,
        config.dropout).to(config.device)
    decoder.load_state_dict(dec_checkpoint['model'])
    decoder.eval()

    print('加载编码器和解码器的参数完成!')

    data_obj = DataProcess(config)
    *_, src_tgt_seq_infer = data_obj.word_2_index(
        config.infer_input_k, config.infer_target_k, src_lang, tgt_lang)  # 推理输入-目标序列

    infer_data_loader = DataLoader(
        src_tgt_seq_infer, 16, False, drop_last=False, pin_memory=1)
    tool = Tools(config)

    all_loss = []
    with torch.no_grad():
        for batch_data in tqdm(infer_data_loader, desc='推理中...', ncols=10, leave=False):
            input_seq, input_len, target_seq, tgt_len = tool.batch_2_tensor(batch_data)
            target_seq = target_seq.transpose(0, 1).to(config.device)
            enc_out, enc_hidden = encoder(input_seq, input_len, None)
            this_batch_size = input_seq.shape[0]

            dec_input = Variable(torch.LongTensor([config.sos] * this_batch_size)).to(config.device)
            dec_hidden = (enc_hidden[0][:config.enc_dec_layer],
                          enc_hidden[1][:config.enc_dec_layer])
            max_tgt_len = max(tgt_len)
            all_dec_out = Variable(torch.zeros(
                max_tgt_len, this_batch_size, decoder.output_size)).to(config.device)

            for t in range(max_tgt_len):
                dec_out, dec_hidden, dec_attn = decoder(dec_input, dec_hidden, enc_out, config)

                all_dec_out[t] = dec_out.to(config.device)
                dec_input = target_seq[t].to(config.device)

            _, loss_batch = masked_cross_entropy(
                all_dec_out.transpose(0, 1).contiguous().to(config.device),
                target_seq.transpose(0, 1).contiguous().to(config.device),
                tgt_len, item=True)

            all_loss += loss_batch

    info = evaluate_model(all_loss, config, start_time)
    write_info.append(info)

    with open(config.out_path + 'epoch_result.txt', 'w', encoding='utf-8') as file:
        for idx, line in enumerate(write_info):
            file.write('第{}个轮次\t'.format(idx) + line + '\n')


def evaluate_model(all_loss, config, start_time):
    infer_con = open(config.infer_target, 'r', encoding='utf-8').readlines()
    infer_con_k = open(config.infer_target_k, 'r', encoding='utf-8').readlines()
    
    assert len(infer_con) * config.k == len(infer_con_k), \
        '预测目标序列条数不是真实目标序列条数的{}倍！'.format(config.k)
    assert len(infer_con_k) == len(all_loss), \
        '预测目标序列条数和预测目标序列损失值条数不一致！'

    pred_loss_txt = [[all_loss[i], infer_con_k[i].split('\n')[0]] for i in range(len(infer_con_k))]
    
    y_pred_acc = [0 for _ in range(len(infer_con))]
    y_pred_acc_5 = [0 for _ in range(len(infer_con))]
    y_pred_acc_10 = [0 for _ in range(len(infer_con))]
    y_true = [1 for _ in range(len(infer_con))]

    for i in range(len(infer_con)):
        k_loss_txt = pred_loss_txt[i * config.k:(i+1) * config.k]

        new_k_loss_txt = sorted(k_loss_txt, key=lambda x:x[0])

        txt = [line[1] for line in new_k_loss_txt]

        if infer_con[i].split('\n')[0] == txt[0]:
            y_pred_acc[i] = 1
        if infer_con[i].split('\n')[0] in txt[:5]:
            y_pred_acc_5[i] = 1
        if infer_con[i].split('\n')[0] in txt[:10]:
            y_pred_acc_10[i] = 1

    acc = round(accuracy_score(y_true, y_pred_acc),5)
    acc_5 = round(accuracy_score(y_true, y_pred_acc_5), 5)
    acc_10 = round(accuracy_score(y_true, y_pred_acc_10), 5)
    f1 = round(f1_score(y_true, y_pred_acc), 5)
    end_time = time.time()
    take_time = end_time - start_time
    item_take_time = take_time / len(infer_con)

    print('数据集{}, Acc={}，Acc@5={}，Acc@10={}，F1={}，耗时{}，每条耗时{}'.
          format(config.data_set, acc, acc_5, acc_10, f1, round(take_time, 5), round(item_take_time, 1)))

    return ('数据集{}, Acc={}，Acc@5={}，Acc@10={}，F1={}，耗时{}，每条耗时{}'.
          format(config.data_set, acc, acc_5, acc_10, f1, take_time, item_take_time))
