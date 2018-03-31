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



# (wav1 + wav2[:len(wav1)])/2
# np.pad(wav2, mode='constant', pad_width=(0, 50)).shape

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

# pool = multiprocessing.Pool(8)
# for _ in tqdm.tqdm(pool.imap_unordered(create_pkl, list(root_path.glob('**/*.mp4'))),
#                    total=len(list(root_path.glob('**/*.mp4')))):
#     pass

# mouth_frames = {}
# print(len([t for t in root_path.glob('**/*.pkl') if 'straightcam' in t.parts]))
# i = 0
# with open('mouth_frames', 'wb') as f:
#     for t in [t for t in root_path.glob('**/*.pkl') if 'straightcam' in t.parts]:
#         print(t.parent.parts[-4:] + (t.stem.lower(),))
#         print(i)
#         pickle.dump((t.parent.parts[-4:] + (t.stem.lower(),), np.load(t.as_posix())), f, protocol=4)
#         i = i+1

# mouth_frames = {t.parent.parts[-4:] + (t.stem.lower(),): np.load(t.as_posix()) for t in root_path.glob('**/*.pkl') if
#                 'straightcam' in t.parts}
# pickle.dump(mouth_frames, open('mouth_frames', 'wb'), protocol=4)

# class LipEncoder(nn.Module):
#     def __init__(self):
#         super(LipEncoder, self).__init__()
#         self.gru = nn.GRU(input_size=512, hidden_size=256, num_layers=2, batch_first=True, bidirectional=True)
#         self.conv1 = nn.Conv3d(in_channels=1, out_channels=128, kernel_size=(2, 3, 3), stride=2)
#         self.bn1 = nn.BatchNorm3d(128)
#         self.conv2 = nn.Conv3d(in_channels=128, out_channels=256, kernel_size=(2, 3, 3), stride=2)
#         self.bn2 = nn.BatchNorm3d(256)
#         self.conv3 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), stride=2)
#         self.bn3 = nn.BatchNorm2d(512)
#         self.conv4 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=2)
#         self.bn4 = nn.BatchNorm2d(512)
#         self.conv5 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=2)
#         self.bn5 = nn.BatchNorm2d(512)
#         self.fc1 = nn.Linear(7680, 512)
#
#     def forward(self, x, lengths=None):
#         x = x.squeeze(-1)
#         x = F.leaky_relu(self.bn1(self.conv1(x)))
#         x = F.leaky_relu(self.bn2(self.conv2(x)))
#         lst = []
#         for i in x:
#             d = i.permute(1, 0, 2, 3)
#             d = F.leaky_relu(self.bn3(self.conv3(d)))
#             d = F.leaky_relu(self.bn4(self.conv4(d)))
#             d = F.leaky_relu(self.bn5(self.conv5(d)))
#             d = d.view(len(d), -1)
#             d = self.fc1(d)
#             lst.append(d)
#         output, hidden = self.gru(torch.stack(lst))
#         return hidden, output.permute(1, 0, 2)
