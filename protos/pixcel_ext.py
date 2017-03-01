
from utils import load_scan, get_pixels_hu, resample, normalize, zero_center

import os
import sys
import glob
import numpy as np
import pandas as pd
import dicom

from logging import getLogger
from cntk import load_model
from cntk.ops import combine
from cntk.io import MinibatchSource, ImageDeserializer, StreamDef, StreamDefs
from skimage.exposure import equalize_hist
from skimage.transform import resize
import cv2

MODEL_PATH = 'model/resnet-152'
logger = getLogger(__name__)

EXPERIMENT_NUMBER = '0001'

DATA_PATH = '../../'
STAGE1_FOLDER = DATA_PATH + 'stage1/stage1/'
# DATA_PATH + 'features/features' + EXPERIMENT_NUMBER + '/'
FEATURE_FOLDER = DATA_PATH + 'features_20170224_pixcel_cnt/'


def get_3d_data(path):
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key=lambda x: int(x.InstanceNumber))
    img = np.stack([s.pixel_array for s in slices])
    return img


def calc_features():
    """Execute the forward propagation on the images to obtain the features
    and save them as numpy arrays.

    """

    logger.info("Compute features")

    idx = np.arange(-1500, 2001)

    for folder in glob.glob(STAGE1_FOLDER + '*'):
        patient_id = os.path.basename(folder)

        batch = get_3d_data(folder)
        unique, counts = np.unique(batch, return_counts=True)
        logger.info("Batch size: {}".format(batch.shape))
        feats = pd.Series(counts, index=unique).ix[idx].fillna(0).values
        logger.info("Feats size: {}".format(feats.shape))
        logger.info("Saving features in %s" % (FEATURE_FOLDER + patient_id))
        np.save(FEATURE_FOLDER + patient_id, feats)


if __name__ == '__main__':
    from logging import StreamHandler, DEBUG, Formatter, FileHandler

    log_fmt = Formatter('%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s ')
    handler = FileHandler('feature.log', 'a')
    handler.setLevel(DEBUG)
    handler.setFormatter(log_fmt)
    logger.setLevel(DEBUG)
    logger.addHandler(handler)

    handler = StreamHandler()
    handler.setLevel(DEBUG)
    handler.setFormatter(log_fmt)
    logger.setLevel(DEBUG)
    logger.addHandler(handler)

    calc_features()
