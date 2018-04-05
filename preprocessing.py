import multiprocessing
from pathlib import Path

import numpy as np
import scipy.io.wavfile as wav
import tqdm
from facedetection.face_detection import FaceDetector
from nnresample import resample
from skvideo.io import vread
from skvideo.utils import rgb2gray

root_path = Path("../TCDTIMITprocessing/downloadTCDTIMIT")

for srcwavpath in root_path.glob('**/*.wav'):
    tgtwavpath = (srcwavpath.parent / (srcwavpath.stem + "_16khz" + srcwavpath.suffix))
    tgtmfccpath = (srcwavpath.parent / (srcwavpath.stem + ".pkl"))
    # if not tgtmfccpath.exists():
    if not tgtwavpath.exists() and '16khz' not in srcwavpath.stem:
        _, srcsig = wav.read(srcwavpath)
        max_nb_bit = float(2 ** (16 - 1))
        srcsig = srcsig / (max_nb_bit + 1.0)
        resampled = resample(srcsig, 16000, 48000)
        rs = (resampled * (max_nb_bit + 1.0)).astype(np.int16)
        wav.write(tgtwavpath, 16000, rs)
        print(tgtwavpath)


def create_pkl(mp4path):
    tgtpklpath = (mp4path.parent / (mp4path.stem + ".pkl"))
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
