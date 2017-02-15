
import os
from logging import getLogger
import pandas as pd
import numpy as np
import pickle
from sklearn import cross_validation
from sklearn.metrics import log_loss
from sklearn.model_selection import GridSearchCV, ParameterGrid

from lightgbm.sklearn import LGBMClassifier

DATA_PATH = '../../'
STAGE1_LABELS = DATA_PATH + 'stage1_labels.csv'
STAGE1_SAMPLE_SUBMISSION = DATA_PATH + 'stage1_sample_submission.csv'
FEATURE_FOLDER = DATA_PATH + 'features_20170205_first/'  # DATA_PATH + 'features/features' + EXPERIMENT_NUMBER + '/'
IDX = [0, 1, 4, 13, 14, 24, 27, 28, 33, 35, 37, 42, 46]
logger = getLogger(__name__)


def compute_prediction(clf, verbose=True):
    """Wrapper function to perform the prediction."""

    df = pd.read_csv(STAGE1_SAMPLE_SUBMISSION)
    # x = np.array([np.mean(np.load(os.path.join(FEATURE_FOLDER, '%s.npy' % str(id))), axis=0).flatten()
    #              for id in df['id'].tolist()])

    x = np.array([np.load(os.path.join(FEATURE_FOLDER, '%s.npy' % str(id)))[IDX].flatten()
                  for id in df['id'].tolist()])

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
