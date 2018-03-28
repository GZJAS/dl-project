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

root_path = Path("../TCDTIMITprocessing/downloadTCDTIMIT")
mlf_file = root_path.parent / "MLFfiles" / "volunteer_labelfiles.mlf"

lines = open(mlf_file).read().split('\n')
dct = {}
for line in lines:
    if line.startswith('"'):
        k = tuple(line.split('.')[0].split('/')[-5:])
        dct[k] = []
    else:
        t = line.split(' ')
        if len(t) == 3:
            dct[k].append(tuple(t))


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
distinct_phns = ['aa', 'ae', 'ah', 'aw', 'ay', 'b', 'ch', 'd', 'dh', 'eh', 'er', 'ey', 'f', 'g', 'hh', 'ih', 'iy', 'jh',
                 'k', 'l', 'm', 'n', 'ng', 'ow', 'oy', 'p', 'r', 's', 'sh', 'sil', 't', 'th', 'uh', 'uw', 'v', 'w', 'y',
                 'z']
tokens_indexes = list(zip(distinct_tokens, range(1, len(distinct_tokens) + 1)))
tokens2index = {w: i for w, i in tokens_indexes}
index2tokens = {i: w for w, i in tokens_indexes}
text_indexes = {k: np.asarray([tokens2index[w] for w in v]) for k, v in text_dict.items()}
# mouth_frames = pickle.load(open('mouth_frames', 'rb'))
with open('mouth_frames2', 'rb') as f:
    mouth_frames = pickle.load(f) + pickle.load(f)
mouth_frames = {key: frames for key, frames in mouth_frames}
# mouth_frames = {t.parent.parts[-4:] + (t.stem.lower(),): np.load(t.as_posix()) for t in root_path.glob('**/*.pkl') if
#                 'straightcam' in t.parts}
combined = sorted([(k, (mouth_frames[k],
                        text_indexes[k])) for k in
                   text_indexes.keys() if k in mouth_frames],
                  key=lambda x: x[0])


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
    keys, tensor_lists = zip(*lst)
    mouth_frames, text_indexes = zip(*tensor_lists)
    max_mouth_time = max([t.size()[1] for t in mouth_frames])
    mouth_frames = [pad_tensor(x, pad=max_mouth_time, dim=1) for x in mouth_frames]
    max_target_time = max([t.size()[0] for t in text_indexes])
    text_indexes = [pad_tensor(x, pad=max_target_time, dim=0) for x in text_indexes]
    return keys, (torch.stack(mouth_frames), torch.stack(text_indexes))


class LipDataset(Dataset):
    def __init__(self):
        pass

    def __len__(self):
        return len(combined)

    def __getitem__(self, index):
        key, (mouth_frames, text_indexes) = combined[index]
        mouth_frames = (torch.from_numpy(mouth_frames).unsqueeze(0).type(torch.FloatTensor) - 128) / 64
        text_indexes = torch.from_numpy(text_indexes).type(torch.LongTensor)
        return key, (mouth_frames, text_indexes)
