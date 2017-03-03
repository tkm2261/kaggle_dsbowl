
import os
import glob
import numpy as np
import pandas as pd
import dicom
from logging import getLogger

MODEL_PATH = 'model/resnet-152'
logger = getLogger(__name__)

EXPERIMENT_NUMBER = '0001'

DATA_PATH = '../input/'
STAGE1_FOLDER = DATA_PATH + 'stage1/stage1/'
PDF_FOLDER = 'features_20170224_pixcel_cnt/'


def get_3d_data(path):
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key=lambda x: int(x.InstanceNumber))
    img = np.stack([s.pixel_array for s in slices])
    return img


def calc_features():
    print("Compute features")

    idx = np.arange(-1500, 2001)

    for folder in glob.glob(STAGE1_FOLDER + '*'):
        patient_id = os.path.basename(folder)

        batch = get_3d_data(folder)
        unique, counts = np.unique(batch, return_counts=True)
        print("Batch size: {}".format(batch.shape))
        feats = pd.Series(counts, index=unique).ix[idx].fillna(0).values
        print("Feats size: {}".format(feats.shape))
        print("Saving features in %s" % (FEATURE_FOLDER + patient_id))
        np.save(PDF_FOLDER + patient_id, feats)


if __name__ == '__main__':
    calc_features()
