import tensorflow as tf
import numpy as np
import pandas as pd
import gzip
import pickle
import random
from multiprocessing import Pool
from scipy import ndimage as nd

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression

from logging import getLogger

logger = getLogger(__name__)

random.seed(0)

DATA_PATH = '../../'
STAGE1_FOLDER = DATA_PATH + 'stage1/'
STAGE1_LABELS = DATA_PATH + 'stage1_labels.csv'
STAGE1_SAMPLE_SUBMISSION = DATA_PATH + 'stage1_sample_submission.csv'

from tf_3dconv2 import FEATURE_FOLDER, MODEL_FOLDER, DTYPE, BATCH_SIZE, DROP_RATE
from tf_3dconv2 import convolutional_neural_network, split_batch


df = pd.read_csv(STAGE1_LABELS)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=871)
for train_idx, test_idx in cv.split(df['cancer'].tolist(), df['cancer'].tolist()):
    df_train = df.ix[train_idx]
    df_test = df.ix[test_idx]

list_patient_id = df_test['id'].tolist()
labels = df_test['cancer'].tolist()

list_test_patient_id = df_train['id'].tolist()
test_labels = df_train['cancer'].tolist()

x = tf.placeholder(DTYPE)
y = tf.placeholder(DTYPE)
keep_prob = tf.placeholder(DTYPE)
is_train = tf.placeholder(bool)


def load_data2(batch):
    return [_load_data(p) for p in batch]


def _load_data(patient_id):
    path = FEATURE_FOLDER
    with gzip.open(path + patient_id + '.pkl.gz', 'rb') as f:
        img = pickle.load(f)
    return img


def train_neural_network(use_epoch):
    list_batch = split_batch(list_patient_id, BATCH_SIZE)
    list_labels = split_batch(labels, BATCH_SIZE)

    prediction, prev_layer = convolutional_neural_network(x, keep_prob, is_train)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(cost)

    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, MODEL_FOLDER + "model.ckpt-%s" % use_epoch)

        total_runs = 0
        for epoch in range(use_epoch):
            logger.info('epoch: %s' % epoch)
            epoch_loss = 0
            successful_runs = 0

            tmp = list(range(len(list_batch)))
            random.shuffle(tmp)

            for cnt, i in enumerate(tmp):
                batch = list_batch[i]
                total_runs += 1
                successful_runs += len(batch)
                logger.debug('batch: {}, batch_size: {}'.format(i, len(batch)))
                try:
                    X = load_data2(batch)
                    Y = [[0, 1] if lb == 1 else [1, 0] for lb in list_labels[i]]
                    _, c, prev = sess.run([optimizer, cost, prev_layer], feed_dict={
                                          x: X, y: Y, keep_prob: DROP_RATE, is_train: True})
                    epoch_loss += c
                    successful_runs += len(batch)
                except Exception as e:
                    logger.info(str(e))
                if cnt % 10 == 0:
                    logger.info('batch loss: %s %s' % (cnt, epoch_loss / successful_runs))

            save_path = saver.save(sess, MODEL_FOLDER + "model-%s.ckpt" % epoch, global_step=epoch)
            logger.info("model saved %s" % save_path)


if __name__ == '__main__':
    from logging import StreamHandler, DEBUG, Formatter, FileHandler

    log_fmt = Formatter('%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s ')
    handler = FileHandler('tf_3dconv_cv.py.log', 'w')
    handler.setLevel(DEBUG)
    handler.setFormatter(log_fmt)
    logger.setLevel(DEBUG)
    logger.addHandler(handler)

    handler = StreamHandler()
    handler.setLevel('INFO')
    handler.setFormatter(log_fmt)
    logger.setLevel('INFO')
    logger.addHandler(handler)

    train_neural_network()
