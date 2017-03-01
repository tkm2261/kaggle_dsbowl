
from utils import load_scan, get_pixels_hu, resample, normalize, zero_center

import os
import sys
import glob
import numpy as np
import dicom
from logging import getLogger
import matplotlib as mpl

logger = getLogger(__name__)


DATA_PATH = '../../'
STAGE1_FOLDER = DATA_PATH + 'stage1/stage1/'


def get_3d_data(path):
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key=lambda x: int(x.InstanceNumber))
    img = np.stack([s.pixel_array for s in slices])

    return img


def calc_features():
    """Execute the forward propagation on the images to obtain the features
    and save them as numpy arrays.

    """

    import matplotlib.pyplot as plt
    plt.switch_backend('agg')
    from matplotlib.backends.backend_pdf import PdfPages
    logger.info("Compute features")
    for folder in glob.glob(STAGE1_FOLDER + '*'):
        patient_id = os.path.basename(folder)
        logger.info("Saving pdf in %s" % (patient_id))
        img = get_3d_data(folder)
        with PdfPages('../../pdf/%s.pdf' % patient_id) as pdf:
            for i in range(img.shape[0]):
                plt.figure()
                plt.imshow(img[i], cmap=plt.cm.gray)
                pdf.savefig()
                plt.close()


if __name__ == '__main__':
    from logging import StreamHandler, DEBUG, Formatter, FileHandler

    log_fmt = Formatter('%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s ')
    handler = StreamHandler()
    handler.setLevel(DEBUG)
    handler.setFormatter(log_fmt)
    logger.setLevel(DEBUG)
    logger.addHandler(handler)

    calc_features()
