# coding=utf-8

r"""
transformer模型针对diagnosis生成concept情况下特殊的验证方式，即计算同一输入diagnosis，
目标序列是不同的concept（包含正确concept），在训练模式下，计算两个序列的loss，即语义损失策略
"""

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn.functional as F

from transformer.utils import Translator, Tools
from transformer.modules import Transformer
from transformer.data_process import DataProcess
import time

from config import Config
config = Config()


def main():
    """
    从Diagnosis序列生成Concept序列的特殊测试（验证）模式，计算输入序列是同一个情况下，目标序列是不同时，
    计算预测序列和目标序列的loss
    :param candidate_k: 候选的concept个数
    :param train: 为集成模型二分类准备数据需要对train数据生成loss
    :param val: 为集成模型二分类准备数据需要对val数据生成loss
    :return: 写出loss文件
    """
    print('数据集路径：', config.data_path)
    print('输出路径：', config.save_path)

    word_data = torch.load(config.save_data)  # 加载保存的训练-验证的输入、目标序列的token信息
    src_lang = word_data['src_lang']  # 输入序列token信息
    tgt_lang = word_data['tgt_lang']  # 目标序列token信息
    data_obj = word_data['data_obj']  # 数据对象

    checkpoint = torch.load(  # 加载保存好的模型训练保存点
        config.save_model_checkpoint,
        map_location=lambda storage, loc: storage.cuda(0))
    transformer = Transformer(  # 定义transformer模型
        input_vocab_num=src_lang['n_words'],
        target_vocab_num=tgt_lang['n_words'],
        src_max_len=data_obj['src_max_len'],
        tgt_max_len=data_obj['tgt_max_len']).cuda()
    transformer.load_state_dict(checkpoint['model'])  # 给transformer模型加载训练好的参数
    transformer.eval()  # 验证模式，保证模型参数不变
    print('加载预训练的模型参数完成！\n')

    data_object = DataProcess()  # 数据处理类对象
    *_, src_tgt_seq = data_object.word_2_index(
        config.infer_input_k,
        config.infer_target_k,
        src_lang, tgt_lang)  # 测试数据

    data_loader = DataLoader(
        dataset=src_tgt_seq,
        batch_size=config.k,  # 根据k的大小，可以乘上2倍，不影响
        shuffle=False,
        drop_last=False)  # 打包批次数据

    all_batch_loss = []  # 所有批次数据中每条数据的loss
    with torch.no_grad():  # 不更新模型梯度情况下测试（验证）
        for candidate_k_data in tqdm(data_loader, mininterval=1, ncols=1, desc='特殊测试中', leave=True):
            src_seq, tgt_seq = Tools().batch_2_tensor(candidate_k_data)  # 获得候选的k个输入、目标序列
            _, pre_seq = transformer.forward(src_seq, tgt_seq)  # 模型预测的序列
            tgt_seq = tgt_seq[:, 1:]  # 构建真实的目标序列
            assert pre_seq.size()[0] == tgt_seq.size()[0], '预测序列和目标序列的条数不一致，无法计算loss！'
            for i in range(pre_seq.size()[0]):  # 循环计算这一批次数据每条的loss
                seq_len = 0
                for j in tgt_seq[i]:
                    if j != config.pad:
                        seq_len += 1  # 计算当前目标序列的实际长度
                loss = F.cross_entropy(  # 计算当前预测序列和实际序列的loss
                    pre_seq[i], tgt_seq[i], ignore_index=config.pad, reduction='elementwise_mean')
                loss = loss.detach().cpu().tolist()
                # if config.loss_cal == 'sum':  # 如果模型的loss是sum模式，要除以实际目标序列的长度
                #     loss /= seq_len  FIXME 这里用mean的效果远比sum好，要找出是什么原因
                all_batch_loss.append(round(loss, 7))  # 计算出的loss要除以实际序列长度

    infer_target_k = open(config.infer_target_k, 'r', encoding='utf-8').readlines()
    assert len(all_batch_loss) == len(infer_target_k), '语义损失数和条数不一致！！！'

    with open(config.result_loss, 'w', encoding='utf-8') as f:  # 写出序列loss文件
        for idx, i in enumerate(all_batch_loss):
            if idx != len(all_batch_loss) - 1:
                f.write(str(i) + '\t' + infer_target_k[idx].split('\n')[0])
                f.write('\n')
            else:
                f.write(str(i) + '\t' + infer_target_k[idx].split('\n')[0])
    end_time = time.time()
    take_time = end_time - start_time
    per_item_time = round(take_time / (len(all_batch_loss)/config.k), 7)

    print('耗时{}，每条耗时{}'.format(take_time, per_item_time))


if __name__ == '__main__':
    start_time = time.time()
    main()  # dd50 k=25;dd200 k=60;dp50 k=30;dp200 k=65
