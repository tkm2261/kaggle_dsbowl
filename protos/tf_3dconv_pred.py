import tensorflow as tf
import numpy as np
import pandas as pd
import gzip
import pickle
import random

random.seed(0)

DATA_PATH = '../../'
STAGE1_FOLDER = DATA_PATH + 'stage1/'
STAGE1_LABELS = DATA_PATH + 'stage1_labels.csv'
STAGE1_SAMPLE_SUBMISSION = DATA_PATH + 'stage1_sample_submission.csv'

from tf_3dconv2 import FEATURE_FOLDER, MODEL_FOLDER
from tf_3dconv2 import convolutional_neural_network

MODEL_EPOC = 1

from logging import getLogger

logger = getLogger(__name__)


def _load_data(patient_id):
    path = FEATURE_FOLDER
    with gzip.open(path + patient_id + '.pkl.gz', 'rb') as f:
        img = pickle.load(f)
    return img


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def train_neural_network(epoch):
    x = tf.placeholder('float')
    #y = tf.placeholder('float')
    keep_prob = tf.placeholder(tf.float32)

    prediction, prev_layer = convolutional_neural_network(x, keep_prob, is_train=False)

    with tf.Session() as sess:
        # 変数の読み込み
        saver = tf.train.Saver()

        saver.restore(sess, MODEL_FOLDER + 'model.ckpt-%s' % epoch)
        save_path = saver.save(sess, MODEL_FOLDER + "model_pred.ckpt")
        logger.info("model saved %s" % save_path)

        df = pd.read_csv(STAGE1_SAMPLE_SUBMISSION)
        list_pred = []
        list_prev = []
        for i, patient_id in enumerate(df['id'].tolist()):
            if i % 10 == 0:
                logger.info("predict %s" % i)

            try:
                X = np.array([_load_data(patient_id)])
            except EOFError:
                continue
            pred, prev = sess.run([prediction, prev_layer], feed_dict={x: X, keep_prob: 1.})
            pred = pred[0][1]
            list_pred.append(pred)
            list_prev.append(prev[0])
        pred = np.array(pred)
        df['cancer'] = sigmoid(pred)
        df.to_csv('submit.csv', index=False)

        df_prev = pd.DataFrame(list_prev)
        df_prev['id'] = df['id'].tolist()
        df_prev['cancer'] = None
        df_prev.to_csv(MODEL_FOLDER + 'prev_pred_%s.csv' % epoch, index=False)


if __name__ == '__main__':
    from logging import StreamHandler, DEBUG, Formatter, FileHandler

    log_fmt = Formatter('%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s ')
    handler = FileHandler('tf_3dconv2.log', 'w')
    handler.setLevel(DEBUG)
    handler.setFormatter(log_fmt)
    logger.setLevel(DEBUG)
    logger.addHandler(handler)

    handler = StreamHandler()
    handler.setLevel('INFO')
    handler.setFormatter(log_fmt)
    logger.setLevel('INFO')
    logger.addHandler(handler)
    for i in range(30, 1000):
        train_neural_network(i)
