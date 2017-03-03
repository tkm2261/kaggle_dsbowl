
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import dicom
import os
import glob
import scipy.ndimage
import matplotlib.pyplot as plt

from skimage import measure, morphology, segmentation
from utils import load_scan, get_pixels_hu, resample
import scipy.ndimage as ndimage

DATA_PATH = '../../'
STAGE1_FOLDER = DATA_PATH + 'stage1/stage1/'
FEATURE_FOLDER = DATA_PATH + 'features_20170303_lung_seg/'


from logging import getLogger

logger = getLogger(__name__)


def generate_markers(image):
    # Creation of the internal Marker
    marker_internal = image < -400
    marker_internal = segmentation.clear_border(marker_internal)
    marker_internal_labels = measure.label(marker_internal)
    areas = [r.area for r in measure.regionprops(marker_internal_labels)]
    areas.sort()
    if len(areas) > 2:
        for region in measure.regionprops(marker_internal_labels):
            if region.area < areas[-2]:
                for coordinates in region.coords:
                    marker_internal_labels[coordinates[0], coordinates[1]] = 0
    marker_internal = marker_internal_labels > 0
    # Creation of the external Marker
    external_a = ndimage.binary_dilation(marker_internal, iterations=10)
    external_b = ndimage.binary_dilation(marker_internal, iterations=55)
    marker_external = external_b ^ external_a
    # Creation of the Watershed Marker matrix
    marker_watershed = np.zeros((512, 512), dtype=np.int)
    marker_watershed += marker_internal * 255
    marker_watershed += marker_external * 128

    return marker_internal, marker_external, marker_watershed


def seperate_lungs(image):
    # Creation of the markers as shown above:
    marker_internal, marker_external, marker_watershed = generate_markers(image)

    # Creation of the Sobel-Gradient
    sobel_filtered_dx = ndimage.sobel(image, 1)
    sobel_filtered_dy = ndimage.sobel(image, 0)
    sobel_gradient = np.hypot(sobel_filtered_dx, sobel_filtered_dy)
    sobel_gradient *= 255.0 / np.max(sobel_gradient)

    # Watershed algorithm
    watershed = morphology.watershed(sobel_gradient, marker_watershed)

    # Reducing the image created by the Watershed algorithm to its outline
    outline = ndimage.morphological_gradient(watershed, size=(3, 3))
    outline = outline.astype(bool)

    # Performing Black-Tophat Morphology for reinclusion
    # Creation of the disk-kernel and increasing its size a bit
    blackhat_struct = [[0, 0, 1, 1, 1, 0, 0],
                       [0, 1, 1, 1, 1, 1, 0],
                       [1, 1, 1, 1, 1, 1, 1],
                       [1, 1, 1, 1, 1, 1, 1],
                       [1, 1, 1, 1, 1, 1, 1],
                       [0, 1, 1, 1, 1, 1, 0],
                       [0, 0, 1, 1, 1, 0, 0]]
    blackhat_struct = ndimage.iterate_structure(blackhat_struct, 8)
    # Perform the Black-Hat
    outline += ndimage.black_tophat(outline, structure=blackhat_struct)

    # Use the internal marker and the Outline that was just created to generate the lungfilter
    lungfilter = np.bitwise_or(marker_internal, outline)
    # Close holes in the lungfilter
    # fill_holes is not used here, since in some slices the heart would be reincluded by accident
    lungfilter = ndimage.morphology.binary_closing(lungfilter, structure=np.ones((5, 5)), iterations=3)

    # Apply the lungfilter (note the filtered areas being assigned -2000 HU)
    segmented = np.where(lungfilter == 1, image, -2000 * np.ones((512, 512)))

    return segmented, lungfilter, outline, watershed, sobel_gradient, marker_internal, marker_external, marker_watershed


def calc_features():
    logger.info("Compute features")

    for folder in glob.glob(STAGE1_FOLDER + '*'):
        patient_id = os.path.basename(folder)
        patient = load_scan(folder)
        patient_pixels = get_pixels_hu(patient)

        feats = []
        for img in patient_pixels:
            segmented, lungfilter, outline, watershed, sobel_gradient, marker_internal, marker_external, marker_watershed = seperate_lungs(
                img)
            feats.append(segmented)
        feats = np.array(feats)
        logger.info("Feats. size: {} {}".format(feats.shape))

        np.save(FEATURE_FOLDER + patient_id, feats)


if __name__ == '__main__':
    from logging import StreamHandler, DEBUG, Formatter, FileHandler

    log_fmt = Formatter('%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s ')
    handler = StreamHandler()
    handler.setLevel(DEBUG)
    handler.setFormatter(log_fmt)
    logger.setLevel(DEBUG)
    logger.addHandler(handler)

    calc_features()
