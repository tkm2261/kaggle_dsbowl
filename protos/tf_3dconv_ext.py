import tensorflow as tf
import numpy as np
import pandas as pd
import gzip
import pickle
import random
from multiprocessing import Pool
from scipy import ndimage as nd

from sklearn.model_selection import StratifiedKFold

random.seed(0)

DATA_PATH = '../../'
STAGE1_FOLDER = DATA_PATH + 'stage1/'
STAGE1_LABELS = DATA_PATH + 'stage1_labels.csv'
STAGE1_SAMPLE_SUBMISSION = DATA_PATH + 'stage1_sample_submission.csv'

DATA_PATH = '../features/'
FEATURE_FOLDER = DATA_PATH + 'features_20170303_lung_binary_resize/'
FEATURE_FOLDER_OUT = DATA_PATH + 'features_20170306_3d_cnn_new/'
# FEATURE_FOLDER_FILL = DATA_PATH + 'features_20170303_lung_binary_fill/'


from logging import getLogger

logger = getLogger(__name__)

IMG_SIZE = (512, 512, 200)

N_CLASSES = 2
BATCH_SIZE = 4

#df = pd.read_csv(STAGE1_LABELS)
import glob
import os
list_patient_id = [os.path.basename(folder).split('.')[0] for folder in glob.glob(FEATURE_FOLDER + '*')]

# df['id'].tolist()


def split_batch(list_data, batch_size):
    ret = []
    for i in range(int(len(list_data) / batch_size) + 1):
        from_idx = i * batch_size
        next_idx = from_idx + batch_size if from_idx + batch_size <= len(list_data) else len(list_data)

        if from_idx >= next_idx:
            break

        ret.append(list_data[from_idx:next_idx])
    return ret

_list_batch = split_batch(list_patient_id, BATCH_SIZE)


def load_data():
    df = pd.read_csv(STAGE1_LABELS)
    labels = df['cancer'].tolist()
    p = Pool()
    images = p.map(_load_data, df['id'].tolist())
    p.close()
    p.join()
    logger.info("image num: %s".format(len(images)))
    return np.array(images), labels


def load_data2(batch):
    return [_load_data(p) for p in batch]


def _load_data(patient_id):

    with gzip.open(FEATURE_FOLDER + patient_id + '.pkl.gz', 'rb') as f:
        img = pickle.load(f)
        # img = nd.interpolation.zoom(img, [float(IMG_SIZE[i]) / img.shape[i] for i in range(3)])
        # logger.debug('{} img size] {}'.format(patient_id, img.shape))
    return img


FC_SIZE = 1024
DTYPE = tf.float32


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def _weight_variable(name, shape):
    return tf.get_variable(name, shape, DTYPE, tf.truncated_normal_initializer(stddev=0.1))


def _bias_variable(name, shape):
    return tf.get_variable(name, shape, DTYPE, tf.constant_initializer(0.1, dtype=DTYPE))

from tf_3dconv_pred import convolutional_neural_network


def train_neural_network():
    x = tf.placeholder('float')
    y = tf.placeholder('float')

    list_batch = _list_batch
    prediction, prev_layer = convolutional_neural_network(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(cost)

    hm_epochs = 1

    with tf.Session() as sess:
        # 変数の読み込み
        saver = tf.train.Saver()
        saver.restore(sess, "model0307_new/model.ckpt-3")
        cnt = 0
        for batch in list_batch:
            X = load_data2(batch)
            p = sess.run(prev_layer, feed_dict={x: X})

            for i, patient_id in enumerate(batch):
                logger.info('{} {} {}'.format(cnt, patient_id, p[i].shape))
                np.save(FEATURE_FOLDER_OUT + patient_id, p[i])
                cnt += 1

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

    train_neural_network()
