
import os
from logging import getLogger
import pandas as pd
import numpy as np
import pickle
from sklearn import cross_validation
from sklearn.metrics import log_loss
from sklearn.model_selection import GridSearchCV, ParameterGrid

from lightgbm.sklearn import LGBMClassifier
from features import FEATURE

DATA_PATH = '../../'
STAGE1_LABELS = DATA_PATH + 'stage1_labels.csv'
STAGE1_SAMPLE_SUBMISSION = DATA_PATH + 'stage1_sample_submission.csv'
# FEATURE_FOLDER = DATA_PATH + 'features_20170215_mxnet/'  # DATA_PATH + 'features/features' + EXPERIMENT_NUMBER + '/'
#FEATURE_FOLDER_2 = DATA_PATH + 'features_20170216_resnet152/'

from mxnet_train import FEATURE_FOLDER, FEATURE_FOLDER_2

logger = getLogger(__name__)


def compute_prediction(clf, verbose=True):
    """Wrapper function to perform the prediction."""

    df = pd.read_csv(STAGE1_SAMPLE_SUBMISSION)
    #x = np.array([np.mean(np.load(FEATURE_FOLDER + '/%s.npy' % str(id)), axis=0) for id in df['id'].tolist()])
    # x = np.array([np.load(FEATURE_FOLDER + '/%s.npy' % str(id))[:30].flatten()
    #              for id in df['id'].tolist()])[:, FEATURE]
    x = np.array([np.r_[np.mean(np.load(FEATURE_FOLDER + '/%s.npy' % str(id)), axis=0)]
                  for id in df['id'].tolist()])
    x2 = np.array([np.load(FEATURE_FOLDER_2 + '/%s.npy' % str(id))
                   for id in df['id'].tolist()])[:, FEATURE]
    x = np.c_[x, x2]

    pred = clf.predict_proba(x)[:, 1]
    df['cancer'] = pred
    return df

if __name__ == "__main__":
    from logging import StreamHandler, DEBUG, Formatter, FileHandler

    log_fmt = Formatter('%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s ')
    handler = FileHandler('hoge.log', 'w')
    handler.setLevel(DEBUG)
    handler.setFormatter(log_fmt)
    logger.setLevel(DEBUG)
    logger.addHandler(handler)

    handler = StreamHandler()
    handler.setLevel(DEBUG)
    handler.setFormatter(log_fmt)
    logger.setLevel(DEBUG)
    logger.addHandler(handler)

    with open('model.pkl', 'rb') as f:
        clf = pickle.load(f)
    df = compute_prediction(clf)
    df.to_csv('submit.csv', index=False)
