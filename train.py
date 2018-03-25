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
from datasets import LipDataset, distinct_tokens, tokens2index, index2tokens
from custom_model import Combined, LipEncoder, Speller, CUDA

EPOCHS = 100

com = Combined(LipEncoder(), Speller(len(distinct_tokens)))
if CUDA:
    com = com.cuda()
criterion = nn.CrossEntropyLoss()
optim = torch.optim.Adam(com.parameters(), lr=1e-4)
data_set = LipDataset()
print('Loaded dataset')
data_loader = torch.utils.data.DataLoader(data_set, batch_size=1, shuffle=True)

for epoch in range(500):
    print("EPOCH: {}".format(epoch))
    print("---------------------------------------")
    for key, (lips, target) in data_loader:
        if CUDA:
            lips, target = lips.cuda(), target.cuda()
        optim.zero_grad()
        pred = com(Variable(lips), target)
        loss = criterion(pred.permute(0, 2, 1), Variable(target[:, 1:]))
        loss.backward()
        optim.step()
        print("".join([index2tokens[t] for t in target[0][1:]]))
        print("".join([index2tokens[t] for t in pred[0].topk(1, 1)[1].squeeze().data]))
