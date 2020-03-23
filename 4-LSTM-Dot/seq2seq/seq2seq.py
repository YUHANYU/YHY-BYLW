# coding=utf-8

r"""
seq2seq模型的训练、验证和推理实现
"""
from torch.autograd import Variable
import torch

from seq2seq.encoder import EncoderRNN
from seq2seq.decoder import BahdanauAttnDecoderRNN, LuongAttnDecoderRNN
from seq2seq.loss_func import masked_cross_entropy

from tqdm import tqdm

from seq2seq.data_process import Tools


class Seq2SeqModel():
    def __init__(self, encoder, decoder, encoder_optimizer, decoder_optimizer, config):
        self.encoder = encoder
        self.decoder = decoder
        self.encoder_optimizer = encoder_optimizer
        self.decoder_optimizer = decoder_optimizer
        self.config = config
        self.tool = Tools(config)

    def train(self, train_loader, val_loader):
        train_log = open(self.config.out_path + 'train_log.txt', 'w', encoding='utf-8')
        for epoch_idx, epoch in enumerate(range(self.config.epoch)):

            self.encoder.train()
            self.decoder.train()
            self.encoder_optimizer.zero_grad()
            self.decoder_optimizer.zero_grad()

            for batch_idx, batch_data in enumerate(tqdm(train_loader, desc='训练中...', leave=True)):
            # for idx, batch_data in enumerate(train_loader):
                input_seq, input_len, target_seq, tgt_len = self.tool.batch_2_tensor(batch_data)
                target_seq = target_seq.transpose(0, 1)
                enc_out, enc_hidden = self.encoder(input_seq, input_len, None)

                this_batch_size = enc_out.shape[1]

                dec_input = Variable(
                    torch.LongTensor([self.config.sos] * this_batch_size)).to(self.config.device)
                dec_hidden = (enc_hidden[0][:self.config.enc_dec_layer],
                              enc_hidden[1][:self.config.enc_dec_layer])
                max_tgt_len = max(tgt_len)
                all_dec_out = Variable(torch.zeros(max_tgt_len, this_batch_size, self.decoder.output_size))

                for t in range(max_tgt_len):
                    dec_out, dec_hidden, dec_attn = self.decoder(dec_input, dec_hidden, enc_out, self.config)

                    all_dec_out[t] = dec_out
                    dec_input = target_seq[t]

                loss = masked_cross_entropy(
                    all_dec_out.transpose(0, 1).contiguous().to(self.config.device),
                    target_seq.transpose(0, 1).contiguous().to(self.config.device),
                    tgt_len)

                train_log.write('当前轮次{}， 当前批次{}，批大小{}，该批损失{}\n'.
                                format(epoch_idx, batch_idx, this_batch_size, loss.detach().cpu().tolist()))

                loss.backward()

                self.encoder_optimizer.step()
                self.decoder_optimizer.step()

                ec = torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), self.config.grad_clip)
                dc = torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), self.config.grad_clip)

            train_log.write('\n\n')

            enc_state_dict = self.encoder.state_dict()
            enc_checkpoint = {
                'model': enc_state_dict,
                'epoch': epoch_idx
            }
            enc_save_name = self.config.out_path + 'enc_' + str(epoch_idx) + '.chkpt'
            torch.save(enc_checkpoint, enc_save_name)

            dec_state_dict = self.decoder.state_dict()
            dec_checkpoint = {
                'model': dec_state_dict,
                'epoch': epoch_idx
            }
            dec_save_name = self.config.out_path + 'dec_' + str(epoch_idx) + '.chkpt'
            torch.save(dec_checkpoint, dec_save_name)

        train_log.close()