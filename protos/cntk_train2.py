
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
FEATURE_FOLDER = DATA_PATH + 'features_20170205_first/'  # DATA_PATH + 'features/features' + EXPERIMENT_NUMBER + '/'

logger = getLogger(__name__)


def train_lightgbm(verbose=True, idx=0):
    """Train a boosted tree with LightGBM."""
    df = pd.read_csv(STAGE1_LABELS)
    data = []
    use_idx = []
    for id in df['id'].tolist():
        nd = np.load(os.path.join(FEATURE_FOLDER, '%s.npy' % str(id)))
        if nd.shape[0] <= idx:
            continue
        data.append(nd[idx].flatten())
        use_idx.append(id)
    x = np.array(data)
    y = df[df['id'].isin(use_idx)]['cancer'].as_matrix()

    trn_x, val_x, trn_y, val_y = cross_validation.train_test_split(x, y, random_state=42, stratify=y,
                                                                   test_size=0.20)

    all_params = {'max_depth': [5, 10, 20, 50],
                  'learning_rate': [0.06, 0.1],
                  'n_estimators': [1500],
                  'min_child_weight': [0, 0.1],
                  'subsample': [0.99, 0.8],
                  'colsample_bytree': [0.8, 0.5, 1],
                  'boosting_type': ['gbdt'],
                  'num_leaves': [10, 21, 50],
                  'seed': [2261]
                  }
    min_score = 100
    min_params = None
    for params in ParameterGrid(all_params):
        params = {'seed': 2261, 'min_child_weight': 0, 'boosting_type': 'gbdt', 'subsample': 0.99, 'num_leaves': 21,
                  'n_estimators': 18, 'max_depth': 20, 'colsample_bytree': 0.8, 'learning_rate': 0.1}

        clf = LGBMClassifier(**params)
        clf.fit(trn_x, trn_y,
                eval_set=[(val_x, val_y)],
                verbose=verbose,
                # eval_metric=log_loss,
                early_stopping_rounds=300
                )
        score = log_loss(val_y, clf.predict_proba(val_x)[:, 1])
        logger.info('idx: %s, score: %s' % (idx, score))
        break

    return clf

if __name__ == "__main__":
    from logging import StreamHandler, DEBUG, Formatter, FileHandler

    log_fmt = Formatter('%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s ')
    handler = FileHandler('slice_score.log', 'w')
    handler.setLevel(DEBUG)
    handler.setFormatter(log_fmt)
    logger.setLevel(DEBUG)
    logger.addHandler(handler)

    handler = StreamHandler()
    handler.setLevel(DEBUG)
    handler.setFormatter(log_fmt)
    logger.setLevel(DEBUG)
    logger.addHandler(handler)

    for idx in range(100):
        train_lightgbm(verbose=False, idx=idx)
