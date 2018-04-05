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
import pickle
from torch.utils.data import Dataset
import pendulum
from custom_model import CUDA
import torch.nn.utils.rnn as rnn_utils
from scipy.misc import imresize

root_path = Path("../TCDTIMITprocessing/downloadTCDTIMIT")
mlf_file = root_path.parent / "MLFfiles" / "volunteer_labelfiles.mlf"

lines = open(mlf_file).read().split('\n')
dct = {}
for line in lines:
    if line.startswith('"'):
        k = line.split('.')[0].split('/')[-5:]
        k[-1] = k[-1].lower()
        k = tuple(k)
        dct[k] = []
    else:
        t = line.split(' ')
        if len(t) == 3:
            dct[k].append(tuple(t))
for k in dct:
    dct[k] = [(float(beg) / float(dct[k][-1][1]), float(end) / float(dct[k][-1][1]), phn) for beg, end, phn in dct[k]]

# def text_to_token_list(t, level):
#     t = [s for s in open(t).read().split() if not s.isnumeric()]
#     s = " ".join(t)
#     if level == 'char':
#         return ["<sos>"] + list(s) + ["<eos>"]
#     return ["<sos>"] + re.split(r'(\s+)', s) + ["<eos>"]
#
#
# w_text_dict = {t.parent.parts[-4:] + (t.stem.lower(),): text_to_token_list(t, 'word') for t
#                in root_path.glob('**/*.txt') if 'straightcam' in t.parts}
# text_dict = {t.parent.parts[-4:] + (t.stem.lower(),): text_to_token_list(t, 'char') for t
#              in root_path.glob('**/*.txt') if 'straightcam' in t.parts}
# distinct_tokens = [w for w, o in itertools.groupby(sorted(sum(text_dict.values(), [])))]
distinct_phns = ['aa', 'ae', 'ah', 'aw', 'ay', 'b', 'ch', 'd', 'dh', 'eh', 'er', 'ey', 'f', 'g', 'hh', 'ih', 'iy', 'jh',
                 'k', 'l', 'm', 'n', 'ng', 'ow', 'oy', 'p', 'r', 's', 'sh', 'sil', 't', 'th', 'uh', 'uw', 'v', 'w', 'y',
                 'z', '<sos>', '<eos>']
tokens_indexes = list(zip(distinct_phns, range(1, len(distinct_phns) + 1)))
tokens2index = {w: i for w, i in tokens_indexes}
index2tokens = {i: w for w, i in tokens_indexes}


# text_indexes = {k: np.asarray([tokens2index[w] for w in v]) for k, v in text_dict.items()}
# mouth_frames = pickle.load(open('mouth_frames', 'rb'))
# with open('mouth_frames2', 'rb') as f:
#     mouth_frames = pickle.load(f) + pickle.load(f)
# mouth_frames = {key: frames for key, frames in mouth_frames}
# mouth_frames = {t.parent.parts[-4:] + (t.stem.lower(),): np.load(t.as_posix()) for t in root_path.glob('**/*.pkl') if
#                 'straightcam' in t.parts}
# combined = sorted([(k, (mouth_frames[k],
#                         text_indexes[k])) for k in
#                    text_indexes.keys() if k in mouth_frames],
#                   key=lambda x: x[0])


def pad_tensor(vec, pad, dim, val=0):
    """
    args:
        vec - tensor to pad
        pad - the size to pad to
        dim - dimension to pad

    return:
        a new tensor padded to 'pad' in dimension 'dim'
    """
    pad_size = list(vec.shape)
    pad_size[dim] = pad - vec.size(dim)
    if pad_size[dim] == 0:
        return vec
    return torch.cat([vec, torch.zeros(*pad_size).fill_(val).type(type(vec))], dim=dim)


def collate_fn(lst):
    clip_lst, mfcc_feat_lst, lips_1_lst, lips_2_lst, text_indexes_lst = zip(*lst)
    if mfcc_feat_lst[0] is None:
        mfcc_feat = None
    else:
        max_mfcc_time = max([t.size()[0] for t in mfcc_feat_lst])
        mfcc_feat = torch.stack([pad_tensor(x, pad=max_mfcc_time, dim=0) for x in mfcc_feat_lst])

    if lips_1_lst[0] is None:
        lips_1 = None
    else:
        max_mouth_time = max([t.size()[1] for t in lips_1_lst])
        lips_1 = torch.stack([pad_tensor(x, pad=max_mouth_time, dim=1) for x in lips_1_lst])
    if lips_2_lst[0] is None:
        lips_2 = None
    else:
        max_mouth_time = max([t.size()[1] for t in lips_1_lst])
        lips_2 = torch.stack([pad_tensor(x, pad=max_mouth_time, dim=1) for x in lips_2_lst])
    max_target_time = max([t.size()[0] for t in text_indexes_lst])
    text_indexes = torch.stack([pad_tensor(x, pad=max_target_time, dim=0) for x in text_indexes_lst])
    return clip_lst, mfcc_feat, lips_1, lips_2, text_indexes


def mix_audio(wav1, wav2):
    wav2 = np.pad(wav2, mode='constant', pad_width=(0, max(len(wav1) - len(wav2), 0)))[0:len(wav1)]
    wav1, wav2 = 16000 / max(wav1) * wav1, 16000 / max(wav2) * wav2
    return ((wav1 + wav2) / 2).astype(np.int16)


def fractional_index(arr, beg, end, indices=False):
    length = arr.shape[0]
    beg_v, end_v = int(round(beg * length)), int(round(end * length))
    if beg_v == end_v:
        end_v = end_v + 1
    if indices:
        return arr[beg_v:end_v], beg_v, end_v
    return arr[beg_v:end_v]


with open('mouth_frames2', 'rb') as f:
    mouth_frames = pickle.load(f) + pickle.load(f)
mouth_frames = {key: frames for key, frames in mouth_frames}

audio_frames = pickle.load(open('audio_frames', 'rb'))
audio_frames = {(k[:-1] + (k[-1].replace('_16khz', ''),)): v for k, v in audio_frames.items()}


class PhonemeDataset(Dataset):
    # TODO: add capability for variable length of phonemes, different channels
    # TODO: things to try: better data augmentation
    # one channel, audio yes, lip yes
    # one channel, audio yes, lip no
    # one channel, audio no, lip yes
    # two channel, audio(mixed with same person), lip1, lip2
    # two channel, audio(mixed with random person), lip1, lip2
    def __init__(self, channels=1, audio='same', lips=True, seq_len=1):
        self.common = sorted(
            set(mouth_frames.keys()).intersection(set(dct.keys())).intersection(set(audio_frames.keys())))
        self.phnm_seqs_raw = [(key, list(zip(*(dct[key][1:-1][i:] for i in range(min(seq_len, len(dct[key][1:-1])))))))
                              for
                              key in
                              self.common]
        self.phnm_seqs = [(phn_seq, key) for (key, key_seqs) in self.phnm_seqs_raw for phn_seq in key_seqs]
        self.speaker_to_indices = {}
        for i in range(0, len(self.phnm_seqs)):
            phn_seq, key = self.phnm_seqs[i]
            speaker = key[:2]
            if speaker not in self.speaker_to_indices:
                self.speaker_to_indices[speaker] = []
            self.speaker_to_indices[speaker].append(i)
        # self.phnms = [(phn, key) for key in self.common for phn in dct[key][1:-1]]
        self.audio = audio
        self.lips = lips
        self.channels = channels

    def __len__(self):
        return len(self.phnm_seqs)

    def __getitem__(self, index):
        phn_seq, clip = self.phnm_seqs[index]
        if self.channels > 1:
            if self.audio == 'same':
                speaker = clip[:2]
                speaker_indices = self.speaker_to_indices[speaker]
                other_index = speaker_indices[np.random.randint(0, len(speaker_indices))]
            else:
                other_index = np.random.randint(0, len(self.phnm_seqs))
            other_phn_seq, other_clip = self.phnm_seqs[other_index]

        if self.audio:
            audio_1 = audio_frames[clip]
            if self.channels == 1:
                mfcc_feat = lmfe(audio_1, 16000, frame_length=0.025, num_filters=80)
            else:
                audio_2 = audio_frames[other_clip]
                mfcc_feat = lmfe(mix_audio(audio_1, audio_2), 16000, frame_length=0.025, num_filters=80)
            mfcc_feat = torch.from_numpy((fractional_index(mfcc_feat, phn_seq[0][0], phn_seq[-1][1]) - 10) / 2).type(
                torch.FloatTensor)
        else:
            mfcc_feat = None

        if self.lips:
            lips_1, beg_v, end_v = fractional_index(mouth_frames[clip], phn_seq[0][0], phn_seq[-1][1], indices=True)
            if np.random.random_sample() < 0.5:
                lips_1 = np.flip(lips_1, axis=2).copy()
            lips_1 = (torch.from_numpy(lips_1).unsqueeze(0).type(torch.FloatTensor) - 128) / 64
            if self.channels > 1:
                if beg_v >= len(mouth_frames[other_clip]):
                    lips_2 = np.zeros([1, lips_1.shape[2], lips_1.shape[3]])
                else:
                    lips_2 = mouth_frames[other_clip][beg_v:end_v]
                if np.random.random_sample() < 0.5:
                    lips_2 = np.flip(lips_2, axis=2).copy()
                lips_2 = (torch.from_numpy(lips_2).unsqueeze(0).type(torch.FloatTensor) - 128) / 64
            else:
                lips_2 = None
        else:
            lips_1, lips_2 = None, None
        text_indexes = torch.LongTensor(
            [tokens2index['<sos>']] + [tokens2index[p[2]] for p in phn_seq] + [tokens2index['<eos>']])
        # imresize((lips[0, 20, :, :] * 64 + 128).type(torch.LongTensor), (112, 112)))
        if self.channels > 1:
            clip = clip + other_clip
        clip = clip + (phn_seq)
        return clip, mfcc_feat, lips_1, lips_2, text_indexes
