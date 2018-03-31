import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from onmt.modules import GlobalAttention, MultiHeadedAttention
import pendulum
import numpy as np

CUDA = True


class LipEncoder(nn.Module):
    def __init__(self):
        super(LipEncoder, self).__init__()
        self.gru = nn.GRU(input_size=512, hidden_size=256, num_layers=1, batch_first=True, bidirectional=True)  # 256
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=128, kernel_size=(2, 3, 3), stride=2, padding=(1, 0, 0))
        self.bn1 = nn.BatchNorm3d(128)
        self.conv2 = nn.Conv3d(in_channels=128, out_channels=256, kernel_size=(2, 3, 3), stride=2, padding=(1, 0, 0))
        self.bn2 = nn.BatchNorm3d(256)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), stride=2)
        self.bn3 = nn.BatchNorm2d(512)
        self.conv4 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=2)
        self.bn4 = nn.BatchNorm2d(512)
        self.conv5 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=2)
        self.bn5 = nn.BatchNorm2d(512)
        self.conv6 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=2)
        self.bn6 = nn.BatchNorm2d(512)
        self.fc1 = nn.Linear(1024, 512)

    def forward(self, x, lengths=None):
        x = x.squeeze(-1)
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        lst = []
        for i in x:
            d = i.permute(1, 0, 2, 3)
            d = F.leaky_relu(self.bn3(self.conv3(d)))
            d = F.leaky_relu(self.bn4(self.conv4(d)))
            d = F.leaky_relu(self.bn5(self.conv5(d)))
            d = F.leaky_relu(self.bn6(self.conv6(d)))
            d = d.view(len(d), -1)
            d = self.fc1(d)
            lst.append(d)
        output, hidden = self.gru(torch.stack(lst))
        return output, hidden


class Speller(nn.Module):
    def __init__(self, distinct_tokens, max_len=150, sos=1, eos=2):
        super(Speller, self).__init__()
        # embedding not needed for char-level model:
        self.vocab_to_embedding = nn.Embedding(distinct_tokens + 1, 8)  # 128 #+1 for padding
        self.to_tokens = nn.Linear(512, distinct_tokens + 1)  # 512
        self.gru = nn.GRU(input_size=8, hidden_size=512, num_layers=1, batch_first=True,
                          bidirectional=False, dropout=0.15)
        self.attn = GlobalAttention(512, attn_type="general")
        # self.attn = MultiHeadedAttention(1, 512)
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

    def forward(self, initial_decoder_hidden, encoder_outputs, targets=None, teacher_forced=True):
        # print("{}: Beginning decoder".format(pendulum.now()))
        outputs = []
        attn_dists = []
        decoder_hidden = initial_decoder_hidden
        if targets is not None:
            # input = self.to_one_hot(targets[:, [0]], self.distinct_tokens)  # should always be <sos>
            # target_embeddings = self.vocab_to_embedding(targets)
            input = self.vocab_to_embedding(targets[:, 0:1])
            # outputs = []
            batch_size, timesteps = targets.size()
            # outputs = torch.FloatTensor(batch_size, timesteps, self.distinct_tokens)
            # if CUDA:
            #     outputs = outputs.cuda()
        else:
            timesteps = self.max_len
        for timestep in range(timesteps - 1):
            gru_output, decoder_hidden = self.gru(input, decoder_hidden)
            attn_output, attn_dist = self.attn(input=decoder_hidden.permute(1, 0, 2)[:, -1:, :],
                                               memory_bank=encoder_outputs)
            attn_output, attn_dist = attn_output.permute(1, 0, 2), attn_dist.permute(1, 0, 2)
            # attn_output, attn_dist = self.attn(query=decoder_hidden.permute(1, 0, 2)[:, -1:, :],
            #                                    key=encoder_outputs, value=encoder_outputs)
            attn_dists.append(attn_dist)
            output_tokens = self.to_tokens(attn_output)
            outputs.append(output_tokens)
            # outputs[:, [timestep], :] = output_tokens.data
            # _, topi = output_tokens.data.topk(1, dim=2)
            # input = self.to_one_hot(topi.squeeze(0)).type(torch.FloatTensor)
            if teacher_forced:
                # input = self.to_one_hot(targets[:, [timestep + 1]], self.distinct_tokens)
                # input = target_embeddings[:, (timestep + 1):(timestep + 2), :]
                input = self.vocab_to_embedding(targets[:, (timestep + 1):(timestep + 2)])
            else:
                # input = output_tokens
                # TODO: the following probably doesn't work
                input = self.vocab_to_embedding(output_tokens.topk(1, dim=2)[1].squeeze(1))
        return torch.cat(outputs, dim=1), torch.cat(attn_dists, dim=1).data
        # return outputs

    def decode(self, decoder_hidden, beam_width=6):
        # freely decode with beam search, returning top result
        return


class Combined(nn.Module):
    def __init__(self, enc, dec):
        super(Combined, self).__init__()
        # embedding not needed for char-level model:
        # self.vocab_to_hidden = nn.Embedding(, 1024)
        self.enc = enc
        self.dec = dec

    def forward(self, lips, targets=None):
        enc_out, enc_hidden = self.enc(lips)
        teacher_force = False
        if targets is not None:
            teacher_force = True if np.random.random_sample() < 0.25 else False
        return self.dec(Combined.encoder_hidden_to_decoder_hidden(enc_hidden), enc_out, targets=targets,
                        teacher_forced=teacher_force)

    @staticmethod
    def encoder_hidden_to_decoder_hidden(encoder_hidden):
        return encoder_hidden.permute(1, 0, 2).contiguous().view(encoder_hidden.size()[1], 1, -1).permute(1, 0, 2)
