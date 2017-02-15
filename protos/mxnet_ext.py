
from utils import load_scan, get_pixels_hu, resample, normalize, zero_center

import os
import sys
import glob
import numpy as np
import dicom

from logging import getLogger
from cntk import load_model
from cntk.ops import combine
from cntk.io import MinibatchSource, ImageDeserializer, StreamDef, StreamDefs
from skimage.exposure import equalize_hist
from skimage.transform import resize
import cv2
import mxnet as mx

MODEL_PATH = 'model/resnet-50'
logger = getLogger(__name__)

EXPERIMENT_NUMBER = '0001'

DATA_PATH = '../../'
STAGE1_FOLDER = DATA_PATH + 'stage1/stage1/'
FEATURE_FOLDER = DATA_PATH + 'features/'  # DATA_PATH + 'features/features' + EXPERIMENT_NUMBER + '/'


def get_extractor():
    model = mx.model.FeedForward.load(MODEL_PATH, 0, ctx=mx.cpu(), numpy_batch_size=1)
    fea_symbol = model.symbol.get_internals()["flatten0_output"]
    feature_extractor = mx.model.FeedForward(ctx=mx.cpu(), symbol=fea_symbol, numpy_batch_size=64,
                                             arg_params=model.arg_params, aux_params=model.aux_params,
                                             allow_extra_params=True)

    return feature_extractor


def get_3d_data(path):
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key=lambda x: int(x.InstanceNumber))
    return np.stack([s.pixel_array for s in slices])


def get_data_id(path):
    """Convert the images in the format accepted by the network trained on ImageNet, packing the
    images in groups of 3 gray images with size of 224x224 and performing some operations.

    """
    sample_image = get_3d_data(path)
    sample_image[sample_image == -2000] = 0
    batch = []
    batch = []
    cnt = 0
    dx = 40
    ds = 512

    for i in range(0, sample_image.shape[0] - 3, 3):
        tmp = []
        for j in range(3):
            img = sample_image[i + j]
            img = 255.0 / np.amax(img) * img
            img = cv2.equalizeHist(img.astype(np.uint8))
            img = img[dx: ds - dx, dx: ds - dx]
            img = cv2.resize(img, (224, 224))
            tmp.append(img)

        tmp = np.array(tmp)
        batch.append(np.array(tmp))

    batch = np.array(batch)
    return batch


def calc_features():
    """Execute the forward propagation on the images to obtain the features
    and save them as numpy arrays.

    """

    logger.info("Compute features")
    net = get_extractor()
    for folder in glob.glob(STAGE1_FOLDER + '*'):
        patient_id = os.path.basename(folder)

        batch = get_data_id(folder)
        logger.info("Batch size: {}".format(batch.shape))
        feats = net.predict(batch)
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
