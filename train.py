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
from datasets import PhonemeDataset, distinct_phns, tokens2index, index2tokens, collate_fn
from custom_model import Combined, LipEncoder, Speller, CUDA
import pendulum

EPOCHS = 100
MANUAL_BATCH = 1

data_set = PhonemeDataset(seq_len=1)
print('Loaded dataset')
training_loader = DataLoader(data_set, batch_size=3, num_workers=4,
                             sampler=SubsetRandomSampler(list(range(0, int(0.95 * len(data_set))))),
                             pin_memory=True, collate_fn=collate_fn)
test_loader = DataLoader(data_set, batch_size=2, num_workers=0,
                         sampler=SubsetRandomSampler(list(range(int(0.95 * len(data_set)), len(data_set)))),
                         pin_memory=True, collate_fn=collate_fn)

curr_epoch = 0
com = Combined(LipEncoder(), Speller(len(distinct_phns)))
if Path("model").exists():
    com.load_state_dict(torch.load("model"))
    curr_epoch = pickle.load(open("epoch", 'rb')) + 1
if CUDA:
    com = com.cuda()
optim = torch.optim.Adam(com.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss(ignore_index=0)

i = 0
tracking_loss = 0
for epoch in range(curr_epoch, 500):
    print("EPOCH: {}".format(epoch))
    print("---------------------------------------")
    for key, (lips, target) in training_loader:
        # print("Mean: {}, Std: {}".format(lips.mean(), lips.std()))
        if CUDA:
            lips, target = lips.cuda(), target.cuda()
        target = Variable(target)
        try:
            pred, attn_dist = com(Variable(lips), target)
        except Exception as e:
            print(e)
            print(lips.shape)
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
        tracking_loss = tracking_loss + loss.data[0]
        # print(attn_dist)
        loss.backward()
        optim.step()
        i = i + 1
        if i % 100 == 0:
            print(tracking_loss)
            print("".join([index2tokens[t] for t in target.data[0][1:] if t != 0]))
            print("".join([index2tokens[t] for t in pred[0].topk(1, 1)[1].squeeze().data if t != 0]))
            # print(attn_dist[0, :1, 0].mean() / min(attn_dist[0, 1:, 0].mean(), 0.01))
            print("Attention targets by inputs")
            print(attn_dist[0, :, :])
            tracking_loss = 0
            torch.save(com.state_dict(), "model")
            pickle.dump(epoch, open("epoch", 'wb'))
    torch.save(com.state_dict(), "model_back")
    pickle.dump(epoch, open("epoch_back", 'wb'))
    # for keys, (lips, target) in test_loader:
