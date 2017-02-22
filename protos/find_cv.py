
import os
import pickle
from logging import getLogger
import pandas as pd
import numpy as np
from sklearn import cross_validation
from sklearn.metrics import log_loss
from sklearn.model_selection import GridSearchCV, ParameterGrid, StratifiedKFold
import xgboost as xgb
from lightgbm.sklearn import LGBMClassifier

DATA_PATH = '../../'
STAGE1_LABELS = DATA_PATH + 'stage1_labels.csv'
FEATURE_FOLDER = DATA_PATH + 'features_20170215_mxnet/'  # DATA_PATH + 'features/features' + EXPERIMENT_NUMBER + '/'
# DATA_PATH + 'features/features' + EXPERIMENT_NUMBER + '/'
FEATURE_FOLDER_2 = DATA_PATH + 'features_20170216_resnet152/'

logger = getLogger(__name__)

from features import FEATURE


def train_xgboost():
    df = pd.read_csv(STAGE1_LABELS)
    x = np.array([np.mean(np.load(FEATURE_FOLDER + '/%s.npy' % str(id)), axis=0) for id in df['id'].tolist()])
    y = df['cancer'].as_matrix()

    trn_x, val_x, trn_y, val_y = cross_validation.train_test_split(x, y, random_state=42, stratify=y,
                                                                   test_size=0.20)

    clf = xgb.XGBRegressor(max_depth=10,
                           n_estimators=1500,
                           min_child_weight=9,
                           learning_rate=0.05,
                           nthread=8,
                           subsample=0.80,
                           colsample_bytree=0.80,
                           seed=4242)

    clf.fit(trn_x, trn_y, eval_set=[(val_x, val_y)], verbose=True, eval_metric='logloss', early_stopping_rounds=50)
    return clf

df = pd.read_csv(STAGE1_LABELS)
data = []
for id in df['id'].tolist():
    dt = np.load(FEATURE_FOLDER + '/%s.npy' % str(id))
    dt = np.r_[np.mean(dt, axis=0), np.max(dt, axis=0), np.min(dt, axis=0), np.var(dt, axis=0)]
    data.append(dt)
x = np.array(data)[:2048]
x2 = np.array(data)[:, FEATURE]
y = df['cancer'].as_matrix()


def train_lightgbm(verbose=True, seed=0):
    """Train a boosted tree with LightGBM."""

    """
    trn_x, val_x, trn_y, val_y = cross_validation.train_test_split(x, y, random_state=42, stratify=y,
                                                                   test_size=0.20)
    """
    all_params = {'max_depth': [3],
                  'learning_rate': [0.06],
                  'n_estimators': [1500],
                  'min_child_weight': [0],
                  'subsample': [1],
                  'colsample_bytree': [0.5],
                  'boosting_type': ['gbdt'],
                  #'num_leaves': [2, 3],
                  #'reg_alpha': [0.1, 0, 1],
                  #'reg_lambda': [0.1, 0, 1],
                  #'is_unbalance': [True, False],
                  #'subsample_freq': [1, 3],
                  'seed': [2261]
                  }
    min_score = 100
    min_params = None
    for params in ParameterGrid(all_params):
        pass
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    list_score = []
    for train, test in cv.split(x, y):
        trn_x = x[train]
        val_x = x[test]
        trn_y = y[train]
        val_y = y[test]

        clf = LGBMClassifier(**params)
        clf.fit(trn_x, trn_y,
                eval_set=[(val_x, val_y)],
                verbose=verbose,
                # eval_metric=log_loss,
                early_stopping_rounds=300
                )
        _score = log_loss(val_y, clf.predict_proba(val_x)[:, 1])
        list_score.append(_score)
    score = np.mean(list_score)
    max_score = np.max(list_score)

    list_score = []
    for train, test in cv.split(x, y):
        trn_x = x2[train]
        val_x = x2[test]
        trn_y = y[train]
        val_y = y[test]

        clf = LGBMClassifier(**params)
        clf.fit(trn_x, trn_y,
                eval_set=[(val_x, val_y)],
                verbose=verbose,
                # eval_metric=log_loss,
                early_stopping_rounds=300
                )
        _score = log_loss(val_y, clf.predict_proba(val_x)[:, 1])
        list_score.append(_score)
    score2 = np.mean(list_score)
    max_score2 = np.max(list_score)
    logger.info('seed: %s, score: %s %s %s %s' % (seed, score, score2, score - score2, max_score - max_score2))

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
    for _ in range(100):
        clf = train_lightgbm(verbose=False, seed=np.random.randint(0, 1000))
    # clf = train_xgboost()
    with open('model.pkl', 'wb') as f:
        pickle.dump(clf, f, -1)
