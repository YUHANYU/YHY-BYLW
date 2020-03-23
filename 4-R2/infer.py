# -*-coding:utf-8-*-

r""" Transformer模型的beam search 推理测试部分命程序

"""

import torch
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader

from transformer.utils import Translator, Tools
from transformer.modules import Transformer
from transformer.data_process import DataProcess

import os
import sys
sys.path.append(os.path.abspath('.'))
from config import Config
config = Config()


def main(candidate_k):
    show_keyword(config, candidate_k)

    print('===开始推理测试===\n')

    def index_2_word(lang, seq):
        """ 转化索引到单词"""
        seq = [int(idx.detach()) for idx in seq]
        new_seq = []
        for i in seq:
            if i != config.sos and i != config.eos and i != config.pad:
                new_seq.append(i)
        idx_2_word = [lang['index2word'][i] for i in new_seq]

        return idx_2_word

    # 加载训练&验证字符信息
    words_data = torch.load(config.save_data)  # 加载已经保存的输入、目标序列token信息
    source_lang = words_data['src_lang']  # 输入序列token信息
    target_lang = words_data['tgt_lang']  # 目标序列token信息
    data_obj = words_data['data_obj']

    checkpoint = torch.load(
        config.save_model_checkpoint,
        map_location='cuda' if torch.cuda.is_available() else 'cpu')
    transformer = Transformer(  # 定义transformer模型
        input_vocab_num=source_lang['n_words'],
        target_vocab_num=target_lang['n_words'],
        src_max_len=data_obj['src_max_len'],
        tgt_max_len=data_obj['tgt_max_len'])

    transformer.load_state_dict(checkpoint['model'])  # 加载transformer模型预训练参数
    transformer.eval()  # 模型验证模式，不更改参数
    print('加载预训练的模型参数完成！\n')

    infer = Translator(model=transformer, tgt_max_len=data_obj['tgt_max_len'])  # 推理预测模型

    data_obj = DataProcess()
    *_, src_tgt_seq = data_obj.word_2_index(
        config.test_input, config.test_target,
        source_lang, target_lang,
        father=config.infer_father if config.enc_father else None)  # 测试数据
    # 打包批次数据
    data_loader = DataLoader(dataset=src_tgt_seq, batch_size=candidate_k, shuffle=False, drop_last=False)

    all_sent_gene_p = []  # 所有句子的beam生成概率
    all_sent_scores, all_sent_scores_sf = [], []  # 所有生成句子的得分
    with open(config.infer_result, 'w', encoding='utf-8') as f:
        for batch_data in tqdm(data_loader, ncols=1, desc='推理测试中...', leave=True):  # 迭代推理批次数据

            if config.has_father:  # 编码父概念
                src_seq, tgt_seq, father_seq, none_seq = Tools().batch_2_tensor(
                    batch_data, source_lang['word2index']['none'])
                src_pos = Tools().seq_2_pos(src_seq)  # 得到输入序列的pos位置向量
                father_pos = Tools().seq_2_pos(father_seq)  # 得到父概念序列的pos位置向量
                batch_pre_seq, batch_sent_scores, batch_sent_scores_sf, batch_gene_p = infer.translate_batch(
                    src_seq, src_pos, father_seq=father_seq, father_pos=father_pos, none_mask=none_seq)  # 预测和对应概率
            else:  # 不编码父概念
                src_seq, tgt_seq = Tools().batch_2_tensor(batch_data)  # 获得输入序列和实际目标序列
                src_pos = Tools().seq_2_pos(src_seq)  # 得到输入序列的pos位置向量
                batch_pre_seq, batch_sent_scores, batch_sent_scores_sf, batch_gene_p = infer.translate_batch(
                    src_seq, src_pos)  # 获得预测结果和概率

            batch_sent_scores = [round(i.cpu().item(), 3) for i in batch_sent_scores]
            batch_sent_scores_sf = [round(i.cpu().item(), 3) for i in batch_sent_scores_sf]
            all_sent_gene_p += batch_gene_p
            all_sent_scores += batch_sent_scores
            all_sent_scores_sf += batch_sent_scores_sf

            pre_word_seq = None  # 推理预测的单词序列
            for index, pre_seq in enumerate(batch_pre_seq):
                src_word_seq = index_2_word(source_lang, src_seq[index])  # 清洗输入序列并转化为字符
                tgt_word_seq = index_2_word(target_lang, tgt_seq[index])  # 清洗目标序列并转化为字符
                for seq in pre_seq:  # 清洗预测序列并转化为字符
                    new_seq = []
                    for i in seq:
                        if i != config.sos and i != config.eos and i != config.pad:
                            new_seq.append(i)
                    pre_word_seq = [target_lang['index2word'][idx] for idx in new_seq]

                f.write('输入序列->：' + ' '.join(src_word_seq) + '\n')  # 写入输入序列
                f.write('->预测序列：' + ' '.join(pre_word_seq) + '\n')  # 写入预测序列
                f.write('==目标序列：' + ' '.join(tgt_word_seq) + '\n\n')  # 写入实际序列

    with open(config.generate_pro, 'w', encoding='utf-8') as gene_f:  # 写出每一句话的生成概率
        for idx, p in enumerate(all_sent_scores_sf):
            gene_f.write(str(idx))
            gene_f.write(' ')
            gene_f.write(str(p))
            gene_f.write('\n')

    print('推理预测序列完毕！')


def show_keyword(param, candidate_k):
    """
    打印测试关键字信息
    :param param: 参数对象
    :param candidate_k: 由concept生成diagnosis，候选的concept数量
    :return: 检查并答应打印参数
    """
    if config.gen_order:
        print('从Diagnosis序列生成Concept序列!')
    else:
        print('从Concept序列生成Diagnosis序列!')
    assert not config.gen_order, '从concept序列生成diagnosis序列时才采用此测试方法！'
    if config.has_father:
        print('编码父概念!')
    else:
        print('没有编码父概念！')
    print('词频最小出现次数限制为{}，序列最大长度限制为{}，Beam Search大小为{}，推理批次大小为{}!\n'.
          format(config.min_word_count, config.max_len, config.beam_size, candidate_k))

if __name__ == '__main__':
    candidate_cons = 25
    main(candidate_cons)
