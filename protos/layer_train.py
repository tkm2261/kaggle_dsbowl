
import os
import pickle
from logging import getLogger
import pandas as pd
import numpy as np
from sklearn import cross_validation
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, ParameterGrid, StratifiedKFold
import xgboost as xgb
from lightgbm.sklearn import LGBMClassifier

DATA_PATH = '../../'
STAGE1_LABELS = DATA_PATH + 'stage1_labels.csv'

logger = getLogger(__name__)

from features import FEATURE


def score_eval(y_true, y_pred):
    return ('auc', roc_auc_score(y_true, y_pred), True)


def train_lightgbm(verbose=True):
    """Train a boosted tree with LightGBM."""
    logger.info("Training with LightGBM")

    num = 5
    df = pd.read_csv('model0311_range900/prev_train_%s.csv' % num)
    df_pred = pd.read_csv('model0311_range900/prev_pred_%s.csv' % num)

    x = df.values[:, :-2]
    y = df['cancer'].values
    logger.info('data size: {}'.format(x.shape))
    """
    all_params = {'max_depth': [3, 5, 10],
                  'learning_rate': [0.06, 0.1, 0.2],
                  'n_estimators': [1500],
                  'min_child_weight': [0],
                  'subsample': [1],
                  'colsample_bytree': [0.5, 0.6],
                  'boosting_type': ['gbdt'],
                  #'num_leaves': [2, 3],
                  #'reg_alpha': [0.1, 0, 1],
                  #'reg_lambda': [0.1, 0, 1],
                  'is_unbalance': [True, False],
                  #'subsample_freq': [1, 3],
                  'seed': [2261]
                  }
    """
    all_params = {'C': [10**i for i in range(-4, 0)],
                  'penalty': ['l1', 'l2'],
                  'class_weight': ['balanced', None],
                  'fit_intercept': [True, False],
                  'random_state': [2261]
                  }

    use_score = 0
    min_score = (100, 100, 100)
    min_score2 = (100, 100, 100)
    min_params = None
    use_gbm = False
    if use_gbm:
        Model = LGBMClassifier
    else:
        Model = LogisticRegression
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=871)
    for params in ParameterGrid(all_params):
        list_score = []
        list_score2 = []
        list_best_iter = []
        for train, test in cv.split(x, y):
            trn_x = x[train]
            val_x = x[test]
            trn_y = y[train]
            val_y = y[test]

            clf = Model(**params)
            if use_gbm:
                clf.fit(trn_x, trn_y,
                        eval_set=[(val_x, val_y)],
                        verbose=verbose,
                        eval_metric=score_eval,  # log_loss,  # log_loss,
                        early_stopping_rounds=300
                        )
                list_best_iter.append(clf.best_iteration)
            else:
                clf.fit(trn_x, trn_y)

            _score = log_loss(val_y, clf.predict_proba(val_x)[:, 1])
            _score2 = - roc_auc_score(val_y, clf.predict_proba(val_x)[:, 1])
            # logger.debug('   _score: %s' % _score)
            list_score.append(_score)
            list_score2.append(_score2)

        if use_gbm:
            params['n_estimators'] = np.mean(list_best_iter, dtype=int)
        score = (np.mean(list_score), np.min(list_score), np.max(list_score))
        score2 = (np.mean(list_score2), np.min(list_score2), np.max(list_score2))

        logger.info('param: %s' % (params))
        logger.info('loss: {} (avg min max {})'.format(score[use_score], score))
        logger.info('score: {} (avg min max {})'.format(score2[use_score], score2))
        if min_score[use_score] > score[use_score]:
            min_score = score
            min_score2 = score2
            min_params = params
        logger.info('best score: {} {}'.format(min_score[use_score], min_score))
        logger.info('best score2: {} {}'.format(min_score2[use_score], min_score2))
        logger.info('best_param: {}'.format(min_params))

    """
    imp = pd.DataFrame(clf.feature_importances_, columns=['imp'])
    with open('features.py', 'a') as f:
        f.write('FEATURE = [' + ','.join(map(str, imp[imp['imp'] > 0].index.values)) + ']\n')
    """
    clf = Model(**min_params)
    clf.fit(x, y)

    pred = clf.predict_proba(df_pred.values[:, :-2])[:, 1]

    df_pred['cancer'] = pred
    df_pred[['id', 'cancer']].to_csv('submit_3dcnn.csv', index=False)

    return clf

if __name__ == "__main__":
    from logging import StreamHandler, DEBUG, Formatter, FileHandler

    log_fmt = Formatter('%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s ')
    handler = FileHandler('layer_train.py.log', 'w')
    handler.setLevel(DEBUG)
    handler.setFormatter(log_fmt)
    logger.setLevel(DEBUG)
    logger.addHandler(handler)

    handler = StreamHandler()
    handler.setLevel(DEBUG)
    handler.setFormatter(log_fmt)
    logger.setLevel(DEBUG)
    logger.addHandler(handler)

    clf = train_lightgbm(verbose=False)
    # clf = train_xgboost()
    with open('model.pkl', 'wb') as f:
        pickle.dump(clf, f, -1)
