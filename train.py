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
from datasets import PhonemeDataset, distinct_phns, tokens2index, index2tokens, collate_fn, pad_tensor
from custom_model import Combined, LipEncoder, AudioEncoder, Speller, CUDA
# import pendulum
import editdistance

EPOCHS = 100
MANUAL_BATCH = 1

data_set = PhonemeDataset(seq_len=70, channels=2, audio='same', lips=True)
test_data_set = PhonemeDataset(seq_len=70, channels=2, audio='different', lips=True)
print('Loaded dataset')
training_loader = DataLoader(data_set, batch_size=1, num_workers=2,
                             sampler=SubsetRandomSampler(list(range(0, int(0.85 * len(data_set))))),
                             pin_memory=True, collate_fn=collate_fn)
test_loader = DataLoader(test_data_set, batch_size=1, num_workers=2,
                         sampler=SubsetRandomSampler(list(range(int(0.85 * len(data_set)), len(data_set)))),
                         pin_memory=True, collate_fn=collate_fn)
sos = tokens2index["<sos>"]
eos = tokens2index["<eos>"]

lip_encoder = LipEncoder()
if Path("lip_encoder").exists():
    lip_encoder.load_state_dict(torch.load("lip_encoder"))
audio_encoder = AudioEncoder()
if Path("audio_encoder").exists():
    audio_encoder.load_state_dict(torch.load("audio_encoder"))
speller = Speller(len(distinct_phns), 3, sos=sos, eos=eos, max_len=70)
if Path("speller").exists():
    speller.load_state_dict(torch.load("speller"))
curr_epoch = 0
com = Combined(speller, audio_encoder, lip_encoder, lip_encoder)
# if Path("model").exists():
#     com.load_state_dict(torch.load("model"))
#     curr_epoch = pickle.load(open("epoch", 'rb')) + 1
if CUDA:
    com = com.cuda()
optim = torch.optim.Adam(com.parameters(), lr=2e-4)
print(com)
criterion = nn.CrossEntropyLoss(ignore_index=0)


def phn_error_rate(pred_y, true_y, eos=eos):
    return np.array([
        editdistance.eval(
            list(itertools.takewhile(lambda x: x != eos, [t for t in pred.topk(1, 1)[1].squeeze().data if t != 0])),
            [t for t in target.data[1:-1] if t != 0]) / len(target)
        for pred, target in zip(pred_y, true_y)]).mean()


i = 0
tracking_loss = 0
for epoch in range(curr_epoch, 500):
    print("EPOCH: {}".format(epoch))
    print("---------------------------------------")

    test_loss = 0
    test_per = 0
    for quintuplet in test_loader:
        clip_lst = quintuplet[0]
        target = quintuplet[-1]
        if CUDA:
            target = target.cuda()
        non_null_inputs = (input for input in quintuplet[1:-1] if input is not None)
        # print("Mean: {}, Std: {}".format(lips.mean(), lips.std()))
        if CUDA:
            non_null_inputs = (input.cuda() for input in non_null_inputs)
        target = Variable(target)
        pred, attn_dist = com(None, *(Variable(input) for input in non_null_inputs))
        loss = criterion(pred.permute(0, 2, 1)[:, :, :target.shape[1] - 1], target[:, 1:])
        test_per = test_per + phn_error_rate(pred, target)
        test_loss = test_loss + loss.data[0]
    print("Phoneme error rate: {}".format(test_per / len(test_loader)))
    print("Average test loss: {}".format(test_loss / len(test_loader)))

    for quintuplet in training_loader:
        clip_lst = quintuplet[0]
        target = quintuplet[-1]
        if CUDA:
            target = target.cuda()
        non_null_inputs = (input for input in quintuplet[1:-1] if input is not None)
        # print("Mean: {}, Std: {}".format(lips.mean(), lips.std()))
        if CUDA:
            non_null_inputs = (input.cuda() for input in non_null_inputs)
        target = Variable(target)
        pred, attn_dist = com(target, *(Variable(input) for input in non_null_inputs))
        loss = criterion(pred.permute(0, 2, 1), target[:, 1:])
        optim.zero_grad()
        tracking_loss = tracking_loss + loss.data[0]
        # print(attn_dist)
        loss.backward()
        optim.step()
        i = i + 1
        if i % 100 == 0:
            print(tracking_loss)
            print(phn_error_rate(pred, target))
            print("".join([index2tokens[t] for t in target.data[0][1:] if t != 0]))
            print("".join([index2tokens[t] for t in pred[0].topk(1, 1)[1].squeeze().data if t != 0]))
            # print("Attention targets by inputs")
            # print(attn_dist[0, :, :])
            tracking_loss = 0
            torch.save(com.state_dict(), "model")
            torch.save(lip_encoder.state_dict(), "lip_encoder")
            torch.save(audio_encoder.state_dict(), "audio_encoder")
            torch.save(speller.state_dict(), "speller")
            pickle.dump(epoch, open("epoch", 'wb'))
    torch.save(lip_encoder.state_dict(), "lip_encoder_backup")
    torch.save(audio_encoder.state_dict(), "audio_encoder_backup")
    torch.save(speller.state_dict(), "speller_backup")
    torch.save(com.state_dict(), "model_back")
    pickle.dump(epoch, open("epoch_back", 'wb'))
