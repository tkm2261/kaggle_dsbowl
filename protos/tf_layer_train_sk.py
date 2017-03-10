
import os
import pickle
from logging import getLogger
import pandas as pd
import numpy as np
from sklearn import cross_validation
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, ParameterGrid, StratifiedKFold
import xgboost as xgb
from lightgbm.sklearn import LGBMClassifier

DATA_PATH = '../../'
STAGE1_LABELS = DATA_PATH + 'stage1_labels.csv'

DATA_PATH = '../features/'
# 'features_20170215_mxnet'  # DATA_PATH + 'features/features' + EXPERIMENT_NUMBER + '/'
FEATURE_FOLDER = DATA_PATH + 'features_20170309_simple'
#FEATURE_FOLDER_2 = DATA_PATH + 'features_20170307_3d_cnn_drop_p2'
#FEATURE_FOLDER_3 = DATA_PATH + 'features_20170224_pixcel_cnt'

# DATA_PATH + 'features/features' + EXPERIMENT_NUMBER + '/'
# FEATURE_FOLDER_2 = DATA_PATH + 'features_20170216_resnet152/'

logger = getLogger(__name__)

from features import FEATURE


def train_xgboost():
    df = pd.read_csv(STAGE1_LABELS)

    x = np.array([np.load(FEATURE_FOLDER + '/%s.npy' % str(id))
                  for id in df['id'].tolist()])
    """
    x2 = np.array([np.load(FEATURE_FOLDER_2 + '/%s.npy' % str(id))
                   for id in df['id'].tolist()])
    x3 = np.array([np.load(FEATURE_FOLDER_3 + '/%s.npy' % str(id))
                   for id in df['id'].tolist()])
    x = np.c_[x, x2, x3]
    """
    y = df['cancer'].as_matrix()
    all_params = {'C': [10**i for i in range(-4, 2)],
                  'penalty': ['l1', 'l2'],
                  'class_weight': ['balanced', None],
                  'fit_intercept': [True, False]
                  }
    min_score = 100
    min_params = None

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=871)
    for params in ParameterGrid(all_params):
        list_score = []
        for train, test in cv.split(x, y):
            trn_x = x[train]
            val_x = x[test]
            trn_y = y[train]
            val_y = y[test]

            clf = LogisticRegression(**params)
            clf.fit(trn_x, trn_y)
            _score = log_loss(val_y, clf.predict_proba(val_x)[:, 1])
            # logger.debug('   _score: %s' % _score)
            list_score.append(_score)
        score = np.mean(list_score)

        logger.info('param: %s' % (params))
        logger.info('score: %s (avg %s min %s max %s)' %
                    (score, np.mean(list_score), np.min(list_score), np.max(list_score)))
        if min_score > score:
            min_score = score
            min_params = params
        logger.info('best score: %s' % min_score)
        logger.info('best_param: %s' % (min_params))

    clf = LogisticRegression(**min_params)
    clf.fit(x, y)

    return clf


if __name__ == "__main__":
    from logging import StreamHandler, DEBUG, Formatter, FileHandler

    log_fmt = Formatter('%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s ')
    handler = FileHandler('mxnet_train.log', 'w')
    handler.setLevel(DEBUG)
    handler.setFormatter(log_fmt)
    logger.setLevel(DEBUG)
    logger.addHandler(handler)

    handler = StreamHandler()
    handler.setLevel(DEBUG)
    handler.setFormatter(log_fmt)
    logger.setLevel(DEBUG)
    logger.addHandler(handler)

    # clf = train_lightgbm(verbose=False)
    clf = train_xgboost()
    with open('model.pkl', 'wb') as f:
        pickle.dump(clf, f, -1)
