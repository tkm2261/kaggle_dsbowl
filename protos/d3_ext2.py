
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import dicom
import os
import glob
import pickle
import gzip
import scipy.sparse as sp
import scipy.ndimage
import matplotlib.pyplot as plt
from multiprocessing import Pool
from skimage import measure, morphology
from utils import load_scan, get_pixels_hu, resample

DATA_PATH = '../../data/data/'
STAGE1_FOLDER = DATA_PATH + 'stage1/'


DATA_PATH = '../features/'
FEATURE_FOLDER = DATA_PATH + 'features_20170303_lung_binary_spacing/'
FEATURE_FOLDER_FILL = DATA_PATH + 'features_20170303_lung_binary_fill_spacing/'


from logging import getLogger

logger = getLogger(__name__)


def largest_label_volume(im, bg=-1):
    vals, counts = np.unique(im, return_counts=True)

    counts = counts[vals != bg]
    vals = vals[vals != bg]

    if len(counts) > 0:
        return vals[np.argmax(counts)]
    else:
        return None


def segment_lung_mask(image, fill_lung_structures=True):

    # not actually binary, but 1 and 2.
    # 0 is treated as background, which we do not want
    binary_image = np.array(image > -320, dtype=np.int8) + 1
    labels = measure.label(binary_image)

    # Pick the pixel in the very corner to determine which label is air.
    #   Improvement: Pick multiple background labels from around the patient
    #   More resistant to "trays" on which the patient lays cutting the air
    #   around the person in half
    background_label = labels[0, 0, 0]

    # Fill the air around the person
    binary_image[background_label == labels] = 2

    # Method of filling the lung structures (that is superior to something like
    # morphological closing)
    if fill_lung_structures:
        # For every slice we determine the largest solid structure
        for i, axial_slice in enumerate(binary_image):
            axial_slice = axial_slice - 1
            labeling = measure.label(axial_slice)
            l_max = largest_label_volume(labeling, bg=0)

            if l_max is not None:  # This slice contains some lung
                binary_image[i][labeling != l_max] = 1

    binary_image -= 1  # Make the image actual binary
    binary_image = 1 - binary_image  # Invert it, lungs are now 1

    # Remove other air pockets insided body
    labels = measure.label(binary_image, background=0)
    l_max = largest_label_volume(labels, bg=0)
    if l_max is not None:  # There are air pockets
        binary_image[labels != l_max] = 0

    return binary_image


def calc_features():
    logger.info("Compute features")
    p = Pool()
    p.map(_calc_features, glob.glob(STAGE1_FOLDER + '*'))
    p.close()
    p.join()


def _calc_features(folder):
    patient_id = os.path.basename(folder)
    patient = load_scan(folder)
    patient_pixels = get_pixels_hu(patient)

    if 1:
        pix_resampled, _ = resample(patient_pixels, patient, [1, 1, 1])
    else:
        pix_resampled = patient_pixels

    segmented_lungs = segment_lung_mask(pix_resampled, False)
    segmented_lungs_fill = segment_lung_mask(pix_resampled, True)

    logger.info("Feats. size: {} {}".format(segmented_lungs.shape, segmented_lungs_fill.shape))

    with gzip.open(FEATURE_FOLDER + patient_id + '.pkl.gz', 'wb') as f:
        pickle.dump(segmented_lungs, f, -1)
    with gzip.open(FEATURE_FOLDER_FILL + patient_id + '.pkl.gz', 'wb') as f:
        pickle.dump(segmented_lungs_fill, f, -1)
        # np.save(FEATURE_FOLDER + patient_id, segmented_lungs)
    # np.save(FEATURE_FOLDER_FILL + patient_id, segmented_lungs_fill)


if __name__ == '__main__':
    from logging import StreamHandler, DEBUG, Formatter, FileHandler

    log_fmt = Formatter('%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s ')
    handler = StreamHandler()
    handler.setLevel(DEBUG)
    handler.setFormatter(log_fmt)
    logger.setLevel(DEBUG)
    logger.addHandler(handler)

    calc_features()
