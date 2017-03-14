import os
import tensorflow as tf
import numpy as np
import pandas as pd
import gzip
import pickle
from multiprocessing import Pool
from scipy import ndimage as nd
from skimage.transform import rotate
from sklearn.model_selection import StratifiedKFold

DATA_PATH = '../../data/data/'
STAGE1_FOLDER = DATA_PATH + 'stage1/'
STAGE1_LABELS = DATA_PATH + 'stage1_labels.csv'

DATA_PATH = '../features/'
FEATURE_FOLDER = DATA_PATH + 'features_20170303_lung_binary_resize/'
FEATURE_FOLDER_OUT = DATA_PATH + 'features_20170303_lung_binary_resize_rotate_yoko_minus10/'
#FEATURE_FOLDER_FILL = DATA_PATH + 'features_20170303_lung_binary_fill/'


from logging import getLogger

logger = getLogger(__name__)

IMG_SIZE = (512, 512, 200)

N_CLASSES = 2


def load_data():
    #df = pd.read_csv(STAGE1_LABELS)

    p = Pool()
    p.map(_pros_data, os.listdir(FEATURE_FOLDER))
    p.close()
    p.join()


def _pros_data(_patient_id):
    patient_id = _patient_id.split('.')[0]
    if os.path.exists(FEATURE_FOLDER_OUT + patient_id + '.pkl.gz'):
        return

    with gzip.open(FEATURE_FOLDER + patient_id + '.pkl.gz', 'rb') as f:
        img = pickle.load(f)
    img = np.array([rotate(im, -10) for im in img])
    #img = rotate(img, -10).astype(np.int8)

    with gzip.open(FEATURE_FOLDER_OUT + patient_id + '.pkl.gz', 'wb') as f:
        pickle.dump(img, f, -1)

    logger.debug('{} img size] {}'.format(patient_id, img.shape))


if __name__ == '__main__':
    from logging import StreamHandler, DEBUG, Formatter, FileHandler

    log_fmt = Formatter('%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s ')
    handler = FileHandler('rotate.log', 'w')
    handler.setLevel(DEBUG)
    handler.setFormatter(log_fmt)
    logger.setLevel(DEBUG)
    logger.addHandler(handler)

    handler = StreamHandler()
    handler.setLevel(DEBUG)
    handler.setFormatter(log_fmt)
    logger.setLevel(DEBUG)
    logger.addHandler(handler)

    load_data()
