
from utils import load_scan,get_pixels_hu, resample

import os
import sys
import glob
import numpy as np
from logging import getLogger
from cntk import load_model
from cntk.ops import combine
from cntk.io import MinibatchSource, ImageDeserializer, StreamDef, StreamDefs
from skimage.exposure import equalize_hist
from skimage.transform import resize

MODEL_PATH='data/ResNet_152.model'
logger = getLogger(__name__)

EXPERIMENT_NUMBER = '0001'

DATA_PATH = 'data/'
STAGE1_FOLDER = DATA_PATH + 'stage1/'
FEATURE_FOLDER=DATA_PATH + 'features/features' + EXPERIMENT_NUMBER + '/'


def get_extractor():
    """Load the CNN."""
    node_name = "z.x"
    loaded_model = load_model(MODEL_PATH)
    node_in_graph = loaded_model.find_by_name(node_name)
    output_nodes = combine([node_in_graph.owner])

    return output_nodes


def get_data_id(data):
    """Convert the images in the format accepted by the network trained on ImageNet, packing the
    images in groups of 3 gray images with size of 224x224 and performing some operations.

    """
    sample_image = data
    sample_image[sample_image == -2000] = 0
    batch = []

    for i in range(0, sample_image.shape[0] - 3, 3):
        tmp = []
        for j in range(3):
            img = sample_image[i + j]
            img = 255.0 / np.amax(img) * img
            img = equalize_hist(img.astype(np.uint8))
            img = resize(img, (224, 224))
            tmp.append(img)

        tmp = np.array(tmp)
        batch.append(np.array(tmp))

    batch = np.array(batch, dtype='int')
    return batch


def batch_evaluation(model, data, batch_size=50):
    """Evaluation of the data in batches too avoid consuming too much memory"""
    num_items = data.shape[0]
    chunks = np.ceil(num_items / batch_size)
    data_chunks = np.array_split(data, chunks, axis=0)
    feat_list = []
    for d in data_chunks:
        feat = model.eval(d)
        feat_list.append(feat)
    feats = np.concatenate(feat_list, axis=0)
    return feats


def calc_features():
    """Execute the forward propagation on the images to obtain the features
    and save them as numpy arrays.

    """
    logger.info("Compute features")
    net = get_extractor()
    for folder in glob.glob(STAGE1_FOLDER + '*'):
        patient_id = os.path.basename(folder)
        logger.info(patient_id)
        patient_data = load_scan(folder)
        patient_pixels = get_pixels_hu(patient_data)
        pix_resampled, spacing = resample(patient_pixels, patient_id, [1, 1, 1])

        batch = get_data_id(pix_resampled)
        logger.info("Batch size: {}".format(batch.shape))

        feats = batch_evaluation(net, batch, 50)
        logger.info("Feats size: {}".format(feats.shape))

        logger.info("Saving features in %s" % (FEATURE_FOLDER + patient_id))
        np.save(FEATURE_FOLDER + patient_id, feats)


if __name__ == '__main__':
    import logging
    log_fmt = '%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s '
    logging.basicConfig(format=log_fmt,
                        datefmt='%Y-%m-%d/%H:%M:%S',
                        level='INFO')
    calc_features()



