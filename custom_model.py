import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from onmt.modules import GlobalAttention
from torch.autograd import Variable

CUDA = True


class AudioEncoder(nn.Module):
    def __init__(self):
        super(AudioEncoder, self).__init__()
        self.gru = nn.GRU(input_size=512, hidden_size=256, num_layers=2, batch_first=True, bidirectional=True)  # 256
        self.conv1 = nn.Conv1d(in_channels=80, out_channels=256, kernel_size=2, stride=2, padding=1)
        self.bn1 = nn.BatchNorm1d(256)
        self.conv2 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=2, stride=2, padding=1)
        self.bn2 = nn.BatchNorm1d(512)
        self.fc1 = nn.Linear(512, 512)

    def forward(self, x, lengths=None):
        # input: batch x timesteps x filters
        x = x.transpose(1, 2)
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        x = x.transpose(1, 2)
        x = self.fc1(x)
        output, hidden = self.gru(x)
        return output, hidden


class LipEncoder(nn.Module):
    def __init__(self):
        super(LipEncoder, self).__init__()
        self.gru = nn.GRU(input_size=512, hidden_size=256, num_layers=2, batch_first=True, bidirectional=True)  # 256
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=64, kernel_size=(2, 3, 3), stride=2, padding=(1, 0, 0))
        self.bn1 = nn.BatchNorm3d(64)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=2)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=2)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), stride=2)
        self.bn4 = nn.BatchNorm2d(512)
        self.conv5 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=2)
        self.bn5 = nn.BatchNorm2d(512)
        self.conv6 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=2)
        self.bn6 = nn.BatchNorm2d(512)
        self.fc1 = nn.Linear(1024, 512)

    def forward(self, x, lengths=None):
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x_shape = x.shape
        x = x.transpose(1, 2).contiguous().view(x_shape[0] * x_shape[2], x_shape[1], x_shape[3], x_shape[4])
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        x = F.leaky_relu(self.bn3(self.conv3(x)))
        x = F.leaky_relu(self.bn4(self.conv4(x)))
        x = F.leaky_relu(self.bn5(self.conv5(x)))
        x = F.leaky_relu(self.bn6(self.conv6(x)))
        x = x.view(x_shape[0], x_shape[2], -1)
        x = self.fc1(x)
        output, hidden = self.gru(x)
        return output, hidden


class Speller(nn.Module):
    def __init__(self, distinct_tokens, encoder_count, sos, eos, max_len=70):
        super(Speller, self).__init__()
        self.vocab_to_embedding = nn.Embedding(distinct_tokens + 1, 8)  # +1 for padding
        self.to_tokens = nn.Linear(encoder_count * 512, distinct_tokens + 1)
        self.initial_hiddens = nn.Linear(encoder_count * 512, 512)
        self.gru = nn.GRU(input_size=8, hidden_size=512, num_layers=1, batch_first=True,
                          bidirectional=False, dropout=0.15)
        self.attns = [GlobalAttention(512, attn_type="general") for _ in range(0, encoder_count)]
        for i in range(0, encoder_count):
            setattr(self, "attn_{}".format(i + 1), self.attns[i])
        self.distinct_tokens = distinct_tokens
        self.max_len = max_len
        self.sos = sos
        self.eos = eos

    @staticmethod
    def to_one_hot(input_x, vocab_size):
        if type(input_x) is Variable:
            input_x = input_x.data
        input_type = type(input_x)
        batch_size = input_x.size(0)
        time_steps = input_x.size(1)
        input_x = input_x.unsqueeze(2).type(torch.LongTensor)
        onehot_x = Variable(
            torch.LongTensor(batch_size, time_steps, vocab_size).zero_().scatter_(-1, input_x, 1)).type(input_type)
        if CUDA:
            return onehot_x.type(torch.cuda.FloatTensor)
        return onehot_x.type(torch.FloatTensor)

    def forward(self, initials_and_encoder_outputs, targets=None, teacher_forced=True):
        outputs = []
        all_attn_dists = []
        initial_decoder_hiddens, all_encoder_outputs = list(zip(*initials_and_encoder_outputs))
        decoder_hidden = self.initial_hiddens(
            torch.cat([initial_decoder_hidden.transpose(0, 1) for initial_decoder_hidden in initial_decoder_hiddens],
                      dim=2)).transpose(0, 1)
        batch_size = decoder_hidden.shape[1]
        input = self.vocab_to_embedding(Variable(torch.LongTensor(batch_size, 1).fill_(self.sos).cuda()))
        if targets is not None:
            batch_size, timesteps = targets.size()
        else:
            timesteps = self.max_len
        for timestep in range(timesteps - 1):
            gru_output, decoder_hidden = self.gru(input, decoder_hidden)
            attn_outputs, attn_dists = zip(*[attn(input=decoder_hidden.permute(1, 0, 2)[:, -1:, :],
                                                  memory_bank=encoder_outputs) for attn, encoder_outputs in
                                             zip(self.attns, all_encoder_outputs)])
            attn_outputs, attn_dists = [attn_output.permute(1, 0, 2) for attn_output in attn_outputs], [
                attn_dist.permute(1, 0, 2) for attn_dist in attn_dists]
            all_attn_dists.append(attn_dists)
            output_tokens = self.to_tokens(torch.cat(attn_outputs, dim=2))
            outputs.append(output_tokens)
            if teacher_forced:
                input = self.vocab_to_embedding(targets[:, (timestep + 1):(timestep + 2)])
            else:
                input = self.vocab_to_embedding(output_tokens.topk(1, dim=2)[1].squeeze(1))
        return torch.cat(outputs, dim=1), (torch.cat(all_attn_dists, dim=1).data for attn_dist in zip(*all_attn_dists))
        # return outputs


class Combined(nn.Module):
    def __init__(self, dec, *encs):
        super(Combined, self).__init__()
        self.encs = encs
        for i in range(0, len(encs)):
            setattr(self, "enc_{}".format(i + 1), self.encs[i])
        self.dec = dec

    def forward(self, targets=None, *inputs):
        outs_and_hiddens = [enc(input) for enc, input in zip(self.encs, inputs)]
        initial_hiddens_and_outs = [(Combined.encoder_hidden_to_decoder_hidden(enc_hidden), enc_out) for
                                    enc_out, enc_hidden in outs_and_hiddens]
        teacher_force = False
        if targets is not None:
            teacher_force = True if np.random.random_sample() < 0.1 else False
        return self.dec(initial_hiddens_and_outs, targets=targets,
                        teacher_forced=teacher_force)

    @staticmethod
    def encoder_hidden_to_decoder_hidden(encoder_hidden):
        num_layers_times_num_directions, batch, hidden_size = encoder_hidden.size()
        return encoder_hidden.permute(1, 0, 2).contiguous().view(batch, 2, 2, -1)[:, -1:, :, :].contiguous().view(
            encoder_hidden.size()[1], 1, -1).permute(1, 0, 2)
