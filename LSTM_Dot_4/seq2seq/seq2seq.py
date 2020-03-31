# coding=utf-8

r"""
seq2seq模型的训练、验证和推理实现
"""
from torch.autograd import Variable
import torch
# from apex import amp

from LSTM_Dot_4.seq2seq.encoder import EncoderRNN
from LSTM_Dot_4.seq2seq.decoder import BahdanauAttnDecoderRNN, LuongAttnDecoderRNN
from LSTM_Dot_4.seq2seq.loss_func import masked_cross_entropy

from tqdm import tqdm

from LSTM_Dot_4.seq2seq.data_process import Tools


class Seq2SeqModel():
    def __init__(self, encoder, decoder, encoder_optimizer, decoder_optimizer, config):
        self.encoder = encoder
        self.decoder = decoder
        self.encoder_optimizer = encoder_optimizer
        self.decoder_optimizer = decoder_optimizer
        self.config = config
        self.val_loss = [0 for _ in range(self.config.epoch)]
        self.tool = Tools(config)

    def train(self, train_loader, val_loader):
        train_log = open(self.config.out_path + 'train_log.txt', 'w', encoding='utf-8')
        torch.cuda.empty_cache()
        for epoch_idx, epoch in enumerate(range(self.config.epoch)):

            self.encoder.train()
            self.decoder.train()
            self.encoder_optimizer.zero_grad()
            self.decoder_optimizer.zero_grad()

            batch_idx = 0
            for batch_data in tqdm(train_loader, desc='训练中...', ncols=50, leave=False):
                input_seq, input_len, target_seq, tgt_len = self.tool.batch_2_tensor(
                    batch_data)
                target_seq = target_seq.transpose(0, 1).to(self.config.device)
                enc_out, enc_hidden = self.encoder(input_seq, input_len, None)

                this_batch_size = enc_out.shape[1]

                dec_input = Variable(
                    torch.LongTensor([self.config.sos] * this_batch_size)).to(self.config.device)
                dec_hidden = (enc_hidden[0][:self.config.enc_dec_layer],
                              enc_hidden[1][:self.config.enc_dec_layer])
                max_tgt_len = max(tgt_len)
                all_dec_out = Variable(
                    torch.zeros(max_tgt_len, this_batch_size, self.decoder.output_size)).to(self.config.device)

                for t in range(max_tgt_len):
                    dec_out, dec_hidden, dec_attn = self.decoder(dec_input, dec_hidden, enc_out, self.config)

                    all_dec_out[t] = dec_out.to(self.config.device)
                    dec_input = target_seq[t].to(self.config.device)

                loss = masked_cross_entropy(
                    all_dec_out.transpose(0, 1).contiguous().to(self.config.device),
                    target_seq.transpose(0, 1).contiguous().to(self.config.device),
                    tgt_len).to(self.config.device)

                train_log.write('当前轮次{}， 当前批次{}，批大小{}，该批损失{}\n'.
                                format(epoch_idx, batch_idx, this_batch_size, loss.detach().cpu().tolist()))

                # print('当前轮次{}， 当前批次{}，批大小{}，该批损失{}'.
                #                 format(epoch_idx, batch_idx, this_batch_size, loss.detach().cpu().tolist()))

                loss.to(self.config.device).backward()

                self.encoder_optimizer.step()
                self.decoder_optimizer.step()

                batch_idx += 1

                # ec = torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), self.config.grad_clip)
                # dc = torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), self.config.grad_clip)

            self.valid(val_loader, epoch_idx)  # 在验证集上进行验证

            train_log.write('\n\n')

        train_log.close()

    def valid(self, val_loader, epoch_idx):
        self.encoder.eval()
        self.decoder.eval()
        this_epoch_loss = 0

        with torch.no_grad():
            for batch_data in tqdm(val_loader, desc='验证中...', ncols=50, leave=False):
                input_seq, input_len, target_seq, tgt_len = self.tool.batch_2_tensor(
                    batch_data)
                target_seq = target_seq.transpose(0, 1).to(self.config.device)
                enc_out, enc_hidden = self.encoder(input_seq, input_len, None)

                this_batch_size = enc_out.shape[1]

                dec_input = Variable(
                    torch.LongTensor([self.config.sos] * this_batch_size)).to(self.config.device)
                dec_hidden = (enc_hidden[0][:self.config.enc_dec_layer],
                              enc_hidden[1][:self.config.enc_dec_layer])
                max_tgt_len = max(tgt_len)
                all_dec_out = Variable(
                    torch.zeros(max_tgt_len, this_batch_size, self.decoder.output_size)).to(self.config.device)

                for t in range(max_tgt_len):
                    dec_out, dec_hidden, dec_attn = self.decoder(dec_input, dec_hidden, enc_out, self.config)

                    all_dec_out[t] = dec_out.to(self.config.device)
                    dec_input = target_seq[t].to(self.config.device)

                loss = masked_cross_entropy(
                    all_dec_out.transpose(0, 1).contiguous().to(self.config.device),
                    target_seq.transpose(0, 1).contiguous().to(self.config.device),
                    tgt_len).to(self.config.device)

                this_epoch_loss += loss.detach().cpu().tolist()
                self.val_loss[epoch_idx] += loss.detach().cpu().tolist()

            if this_epoch_loss <= min(self.val_loss[:epoch_idx+1]):
                print('保存第{}个轮次的模型参数！'.format(epoch_idx))

                enc_state_dict = self.encoder.state_dict()
                enc_checkpoint = {
                    'model': enc_state_dict,
                    'epoch': epoch_idx
                }
                enc_save_name = self.config.out_path + 'enc.chkpt'
                torch.save(enc_checkpoint, enc_save_name)

                dec_state_dict = self.decoder.state_dict()
                dec_checkpoint = {
                    'model': dec_state_dict,
                    'epoch': epoch_idx
                }
                dec_save_name = self.config.out_path + 'dec.chkpt'
                torch.save(dec_checkpoint, dec_save_name)

