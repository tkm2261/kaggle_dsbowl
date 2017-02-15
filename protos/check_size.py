m
import os
import pickle
from logging import getLogger
import pandas as pd
import numpy as np
from sklearn import cross_validation
from sklearn.metrics import log_loss
from sklearn.model_selection import GridSearchCV, ParameterGrid

from lightgbm.sklearn import LGBMClassifier

DATA_PATH = '../../'
STAGE1_LABELS = DATA_PATH + 'stage1_labels.csv'
FEATURE_FOLDER = DATA_PATH + 'features/'  # DATA_PATH + 'features/features' + EXPERIMENT_NUMBER + '/'

logger = getLogger(__name__)


def train_lightgbm(verbose=True):
    """Train a boosted tree with LightGBM."""
    logger.info("Training with LightGBM")
    df = pd.read_csv(STAGE1_LABELS)
    n_slices = []
    for id in df['id'].tolist():
        data = np.load(os.path.join(FEATURE_FOLDER, '%s.npy' % str(id)))
        aaa = data.shape[0]
        n_slices.append(aaa)
    logger.info("{}".format(sorted(n_slices)[:10]))


if __name__ == "__main__":
    from logging import StreamHandler, DEBUG, Formatter, FileHandler

    log_fmt = Formatter('%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s ')

    handler = StreamHandler()
    handler.setLevel(DEBUG)
    handler.setFormatter(log_fmt)
    logger.setLevel(DEBUG)
    logger.addHandler(handler)

    train_lightgbm(verbose=True)
