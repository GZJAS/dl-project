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

root_path = Path("../TCDTIMITprocessing/downloadTCDTIMIT")


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
tokens2index = {w: i for w, i in tokens_indexes}
index2tokens = {i: w for w, i in tokens_indexes}
text_indexes = {k: np.asarray([tokens2index[w] for w in v]) for k, v in text_dict.items()}
# mouth_frames = pickle.load(open('mouth_frames', 'rb'))
mouth_frames = {t.parent.parts[-4:] + (t.stem.lower(),): np.load(t.as_posix()) for t in root_path.glob('**/*.pkl') if
                'straightcam' in t.parts}
combined = sorted([(k, (mouth_frames[k],
                        text_indexes[k])) for k in
                   text_indexes.keys() if k in mouth_frames],
                  key=lambda x: x[0])


class LipDataset(Dataset):
    def __init__(self):
        pass

    def __len__(self):
        return len(combined)

    def __getitem__(self, index):
        key, (mouth_frames, text_indexes) = combined[index]
        return key, ((torch.from_numpy(mouth_frames).unsqueeze(0).type(torch.FloatTensor) - 128) / 256,
                     torch.from_numpy(text_indexes).type(torch.LongTensor))
