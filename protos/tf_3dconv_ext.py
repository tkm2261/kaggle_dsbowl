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

from tf_3dconv2 import FEATURE_FOLDER, MODEL_FOLDER, DTYPE, BATCH_SIZE
from tf_3dconv2 import convolutional_neural_network, split_batch

BATCH_SIZE = 2
from logging import getLogger

logger = getLogger(__name__)

x = tf.placeholder(DTYPE)
y = tf.placeholder(DTYPE)
keep_prob = tf.placeholder(DTYPE)

df = pd.read_csv(STAGE1_LABELS)
list_patient_id = df['id'].tolist()
labels = df['cancer'].tolist()

list_batch = split_batch(list_patient_id, BATCH_SIZE)
list_labels = split_batch(labels, BATCH_SIZE)

prediction, prev_layer = convolutional_neural_network(x, keep_prob, is_train=False)


def load_data2(batch):
    return [_load_data(p) for p in batch]


def _load_data(patient_id):
    path = FEATURE_FOLDER
    with gzip.open(path + patient_id + '.pkl.gz', 'rb') as f:
        img = pickle.load(f)
    return img


def train_neural_network(epoch):

    with tf.Session() as sess:
        # 変数の読み込み
        saver = tf.train.Saver()
        saver.restore(sess, MODEL_FOLDER + 'model.ckpt-%s' % epoch)
        cnt = 0
        list_prev = []
        list_ord_batch = []
        list_ord_label = []

        for cnt, batch in enumerate(list_batch):
            X = load_data2(batch)
            prev = sess.run(prev_layer, feed_dict={x: X, keep_prob: 1.})
            list_prev += [prev[j] for j in range(len(batch))]
            list_ord_batch += batch
            list_ord_label += list_labels[i]

            logger.info('{} batch: {}'.format(epoch, cnt))
        df_prev = pd.DataFrame(list_prev)
        df_prev['id'] = list_ord_batch
        df_prev['cancer'] = list_ord_label
        df_prev.to_csv(MODEL_FOLDER + 'prev_train_%s.csv' % epoch, index=False)
        logger.info('prev data: {}'.format(df_prev.shape))

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

    # data, labels = load_data()
    # If you are working with the basic sample data, use maybe 2 instead of
    # 100 here... you don't have enough data to really do this
    # Run this locally:
    for i in range(100):
        train_neural_network(i)
