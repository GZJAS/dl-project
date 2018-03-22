import scipy.io.wavfile as wav
from nnresample import resample
# from scipy.signal import resample
from speechpy.feature import lmfe
from pathlib import Path
import numpy as np
from facedetection.face_detection import FaceDetector
from mediaio.video_io import VideoFileReader
from skimage.io import imread, imsave
from skvideo.io import vread
from skvideo.utils import rgb2gray
import multiprocessing
import tqdm
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import string
import re
import onmt




root_path = Path("../TCDTIMITprocessing/downloadTCDTIMIT")
# for srcwavpath in root_path.glob('**/*.wav'):
#     if "16khz_16khz" in srcwavpath.stem:
#         srcwavpath.unlink()
#         print(srcwavpath)

for srcwavpath in root_path.glob('**/*.wav'):
    tgtwavpath = (srcwavpath.parent / (srcwavpath.stem + "_16khz" + srcwavpath.suffix))
    tgtmfccpath = (srcwavpath.parent / (srcwavpath.stem + ".pkl"))
    # if not tgtmfccpath.exists():
    if not tgtwavpath.exists() and '16khz' not in srcwavpath.stem:
        _, srcsig = wav.read(srcwavpath)
        max_nb_bit = float(2 ** (16 - 1))
        srcsig = srcsig / (max_nb_bit + 1.0)
        # resampled = resample(srcsig, int(len(srcsig)/3))
        resampled = resample(srcsig, 16000, 48000)
        rs = (resampled * (max_nb_bit + 1.0)).astype(np.int16)
        wav.write(tgtwavpath, 16000, rs)
        print(tgtwavpath)
    # _, tgtsig = wav.read(tgtwavpath)
    # mfcc_feat = lmfe(tgtsig, 16000, frame_length=0.025, num_filters=80)


def create_pkl(mp4path):
    tgtpklpath = (mp4path.parent / (mp4path.stem + ".pkl"))
    # if not tgtmfccpath.exists():
    if not tgtpklpath.exists():
        try:
            images = rgb2gray(vread(mp4path)).astype(np.uint8).squeeze()
            face_detector = FaceDetector()
            faces = np.stack([face_detector.crop_mouth(image, bounding_box_shape=(220, 150)) for image in images], 0)
            faces.dump(tgtpklpath.as_posix())
        except Exception as e:
            print("{}: {}".format(tgtpklpath, e))


pool = multiprocessing.Pool(8)
for _ in tqdm.tqdm(pool.imap_unordered(create_pkl, list(root_path.glob('**/*.mp4'))),
                   total=len(list(root_path.glob('**/*.mp4')))):
    pass


def text_to_token_list(t, level):
    t = [s for s in open(t).read().split() if not s.isnumeric()]
    s = " ".join(t)
    if level == 'char':
        return ["<sos>"] + list(s) + ["<eos>"]
    return ["<sos>"] + re.split(r'(\s+)', s) + ["<eos>"]


w_text_dict = {t.parent.parts[-4:] + (t.stem.lower(),): text_to_token_list(t, 'word') for t
               in root_path.glob('**/*.txt') if 'straightcam' in t.parts}
text_dict = {t.parent.parts[-4:] + (t.stem.lower(),): text_to_token_list(t, 'char') for t
             in root_path.glob('**/*.txt') if 'straightcam' in t.parts}
distinct_tokens = [w for w, o in itertools.groupby(sorted(sum(text_dict.values(), [])))]
tokens_indexes = list(zip(distinct_tokens, range(0, len(distinct_tokens) + 0)))
# pad: 0
tokens2index = {w: i for w, i in tokens_indexes}
index2tokens = {i: w for w, i in tokens_indexes}
text_indexes = {k: np.asarray([tokens2index[w] for w in v]) for k, v in text_dict.items()}

mouth_frames = {t.parent.parts[-4:] + (t.stem.lower(),): np.load(t.as_posix()) for t in root_path.glob('**/*.pkl')}
# pickle.dump(mouth_frames, open('mouth_frames', 'wb'))

combined = sorted([(k, (mouth_frames[k], text_indexes[k])) for k in text_indexes.keys() if k in mouth_frames],
                  key=lambda x: x[0])
mouth_tensor = [Variable(torch.from_numpy(x[1][0])) for x in combined]


class LipEncoder(nn.Module):
    def __init__(self):
        super(LipEncoder, self).__init__()
        self.gru = nn.GRU(input_size=512, hidden_size=256, num_layers=2, batch_first=True, bidirectional=True)
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=128, kernel_size=(2, 3, 3), stride=2)
        self.bn1 = nn.BatchNorm3d(128)
        self.conv2 = nn.Conv3d(in_channels=128, out_channels=256, kernel_size=(2, 3, 3), stride=2)
        self.bn2 = nn.BatchNorm3d(256)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), stride=2)
        self.bn3 = nn.BatchNorm2d(512)
        self.conv4 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=2)
        self.bn4 = nn.BatchNorm2d(512)
        self.conv5 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=2)
        self.bn5 = nn.BatchNorm2d(512)
        self.fc1 = nn.Linear(7680, 512)

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
            d = d.view(len(d), -1)
            d = self.fc1(d)
            lst.append(d)
        output, hidden = self.gru(torch.stack(lst))
        return hidden, output.permute(1, 0, 2)


class Speller(nn.Module):
    def __init__(self):
        super(Speller, self).__init__()
        # embedding not needed for char-level model:
        # self.vocab_to_hidden = nn.Embedding(, 1024)
        self.to_tokens = nn.Linear(512, len(distinct_tokens))
        self.gru = nn.GRU(input_size=len(distinct_tokens), hidden_size=512, num_layers=2, batch_first=True,
                          bidirectional=False)

    @staticmethod
    def to_one_hot(input_x, vocab_size=len(distinct_tokens)):
        if type(input_x) is Variable:
            input_x = input_x.data
        input_type = type(input_x)
        batch_size = input_x.size(0)
        time_steps = input_x.size(1)
        input_x = input_x.unsqueeze(2).type(torch.LongTensor)
        onehot_x = Variable(
            torch.LongTensor(batch_size, time_steps, vocab_size).zero_().scatter_(-1, input_x, 1)).type(input_type)
        return onehot_x

    def forward_step(self, input, decoder_hidden, encoder_):
        gru_output, decoder_hidden = self.gru(input, decoder_hidden)
        return self.to_tokens(gru_output), decoder_hidden

    def forward(self, initial_decoder_hidden, targets, teacher_forced=True):
        if teacher_forced:
            one_hot_target = self.to_one_hot(targets).type(torch.FloatTensor)
            output, last_hidden_state = self.gru(one_hot_target, initial_decoder_hidden)
            return self.to_tokens(output)
        if not teacher_forced:
            decoder_hidden = initial_decoder_hidden
            input = targets[:, [0], :]
            outputs = []
            for timestep in range(targets.size()[1]):
                # output, decoder_hidden = self.gru(targets[:, [timestep], :], decoder_hidden)
                output, decoder_hidden = self.gru(input, decoder_hidden)
                outputs.append(output)
                output_tokens = self.to_tokens(output)
                _, topi = output_tokens.data.topk(1, dim=2)
                input = self.to_one_hot(topi.squeeze(0)).type(torch.FloatTensor)
            output = torch.stack(outputs, dim=1)

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

    def forward(self, lips, targets):
        enc_out, enc_hidden = self.enc(lips)
        return self.dec(Combined.encoder_hidden_to_decoder_hidden(enc_hidden), targets)

    @staticmethod
    def encoder_hidden_to_decoder_hidden(encoder_hidden):
        return encoder_hidden.permute(1, 0, 2).view(encoder_hidden.size()[1], 2, -1).permute(1, 0, 2)


com = Combined(LipEncoder(), Speller())
criterion = nn.CrossEntropyLoss()
optim = torch.optim.Adam(com.parameters(), lr=1e-4)
for lips, targets in range(0, 5):
    optim.zero_grad()
    pred = com(lips, targets)
    loss = criterion(pred.permute(0, 2, 1), targets)
    loss.backward()
    optim.step()
