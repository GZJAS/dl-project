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
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from datasets import LipDataset, distinct_tokens, tokens2index, index2tokens, collate_fn
from custom_model import Combined, LipEncoder, Speller, CUDA
import pendulum

EPOCHS = 100
MANUAL_BATCH = 1

data_set = LipDataset()
print('Loaded dataset')
training_loader = DataLoader(data_set, batch_size=1, num_workers=0, sampler=SubsetRandomSampler(list(range(0, 5000))),
                             pin_memory=True, collate_fn=collate_fn)
test_loader = DataLoader(data_set, batch_size=2, num_workers=0,
                         sampler=SubsetRandomSampler(list(range(5000, len(data_set)))),
                         pin_memory=True, collate_fn=collate_fn)

curr_epoch = 0
com = Combined(LipEncoder(), Speller(len(distinct_tokens)))
if Path("model").exists():
    com.load_state_dict(torch.load("model"))
    curr_epoch = pickle.load(open("epoch", 'rb')) + 1
if CUDA:
    com = com.cuda()
optim = torch.optim.Adam(com.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss(ignore_index=0)

for epoch in range(curr_epoch, 500):
    print("EPOCH: {}".format(epoch))
    print("---------------------------------------")
    i = 0
    manual_batch_loss = 0
    for key, (lips, target) in training_loader:
        # print("Mean: {}, Std: {}".format(lips.mean(), lips.std()))
        if CUDA:
            lips, target = lips.cuda(), target.cuda()
        target = Variable(target)
        pred, attn_dist = com(Variable(lips), target)
        # pred = com(Variable(lips), target)
        loss = criterion(pred.permute(0, 2, 1), target[:, 1:])
        # manual_batch_loss = manual_batch_loss + loss
        # if i % MANUAL_BATCH == 0:
        #     ave_loss = manual_batch_loss / MANUAL_BATCH
        #     print(ave_loss.data[0])
        #     optim.zero_grad()
        #     ave_loss.backward()
        #     optim.step()
        #     manual_batch_loss = 0
        #     ave_loss = 0
        optim.zero_grad()
        print(loss.data[0])
        print(attn_dist[:, :5, 0].mean()/attn_dist[:, 5:, 0].mean())
        # print(attn_dist)
        loss.backward()
        optim.step()
        print("".join([index2tokens[t] for t in target.data[0][1:] if t != 0]))
        print("".join([index2tokens[t] for t in pred[0].topk(1, 1)[1].squeeze().data if t != 0]))
        i = i + 1
    torch.save(com.state_dict(), "model")
    pickle.dump(epoch, open("epoch", 'wb'))
    # for keys, (lips, target) in test_loader:
