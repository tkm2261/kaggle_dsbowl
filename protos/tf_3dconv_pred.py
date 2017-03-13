import tensorflow as tf
import numpy as np
import pandas as pd
import gzip
import pickle
import random
from multiprocessing import Pool
from scipy import ndimage as nd

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression

random.seed(0)

DATA_PATH = '../../'
STAGE1_FOLDER = DATA_PATH + 'stage1/'
STAGE1_LABELS = DATA_PATH + 'stage1_labels.csv'
STAGE1_SAMPLE_SUBMISSION = DATA_PATH + 'stage1_sample_submission.csv'


from tf_3dconv2 import BATCH_SIZE, FEATURE_FOLDER, LIST_FEATURE_FOLDER, IMG_SIZE, N_CLASSES, FC_SIZE, DTYPE, MODEL_FOLDER

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


from tf_3dconv2 import convolutional_neural_network


def train_neural_network():
    x = tf.placeholder('float')
    y = tf.placeholder('float')
    keep_prob = tf.placeholder(tf.float32)

    prediction, prev_layer = convolutional_neural_network(x, keep_prob, is_train=False)

    with tf.Session() as sess:
        # 変数の読み込み
        saver = tf.train.Saver()

        saver.restore(sess, MODEL_FOLDER + 'model.ckpt-%s' % MODEL_EPOC)
        save_path = saver.save(sess, MODEL_FOLDER + "model_pred.ckpt")
        logger.info("model saved %s" % save_path)

        df = pd.read_csv(STAGE1_SAMPLE_SUBMISSION)
        pred = []
        for i, patient_id in enumerate(df['id'].tolist()):
            if i % 10 == 0:
                logger.info("predict %s" % i)

            try:
                X = np.array([_load_data(patient_id)])
            except EOFError:
                continue
            p = sess.run(prediction, feed_dict={x: X, keep_prob: 1.})[0][1]
            pred.append(p)
        pred = np.array(pred)
        df['cancer'] = sigmoid(pred)
        df.to_csv('submit.csv', index=False)

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

    train_neural_network()
