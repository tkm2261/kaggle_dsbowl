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
FEATURE_FOLDER = DATA_PATH + 'features_20170308_simple_3dimage_resize/'

"""
FEATURE_FOLDER_1 = DATA_PATH + 'features_20170303_lung_binary_resize_rotate_yoko_plus10/'
FEATURE_FOLDER_2 = DATA_PATH + 'features_20170303_lung_binary_resize_rotate_yoko_minus10/'
FEATURE_FOLDER_3 = DATA_PATH + 'features_20170303_lung_binary_resize_rotate_tate_plus10/'
FEATURE_FOLDER_4 = DATA_PATH + 'features_20170303_lung_binary_resize_rotate_tate_minus10/'
"""
LIST_FEATURES = [FEATURE_FOLDER]  # , FEATURE_FOLDER_1, FEATURE_FOLDER_2, FEATURE_FOLDER_3, FEATURE_FOLDER_4]

from logging import getLogger

logger = getLogger(__name__)

IMG_SIZE = (512, 512, 200)

N_CLASSES = 2
BATCH_SIZE = 4

df = pd.read_csv(STAGE1_LABELS)
list_patient_id = df['id'].tolist()
labels = df['cancer'].tolist()
keep_prob = tf.placeholder(tf.float32)


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
_list_labels = split_batch(labels, BATCH_SIZE)


def load_data():
    df = pd.read_csv(STAGE1_LABELS)
    labels = df['cancer'].tolist()
    p = Pool()
    images = p.map(_load_data, df['id'].tolist())
    p.close()
    p.join()
    logger.info("image num: %s".format(len(images)))
    return np.array(images), labels


from tf_3dconv2_rot import load_data2, _load_data
"""
def load_data2(batch):
    return [_load_data(p) for p in batch]



def _load_data(patient_id):

    with gzip.open(FEATURE_FOLDER + patient_id + '.pkl.gz', 'rb') as f:
        img = pickle.load(f)
        # img = nd.interpolation.zoom(img, [float(IMG_SIZE[i]) / img.shape[i] for i in range(3)])
        # logger.debug('{} img size] {}'.format(patient_id, img.shape))
    return img
"""

FC_SIZE = 1024
DTYPE = tf.float32


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def _weight_variable(name, shape):
    return tf.get_variable(name, shape, DTYPE, tf.truncated_normal_initializer(stddev=0.1))


def _bias_variable(name, shape):
    return tf.get_variable(name, shape, DTYPE, tf.constant_initializer(0.1, dtype=DTYPE))


def convolutional_neural_network(x):

    x = tf.reshape(x, shape=[-1, IMG_SIZE[0], IMG_SIZE[1], IMG_SIZE[2], 1])

    prev_layer = x

    in_filters = 1
    with tf.variable_scope('conv1') as scope:
        out_filters = 16
        kernel = _weight_variable('weights', [10, 10, 5, in_filters, out_filters])
        conv = tf.nn.conv3d(prev_layer, kernel, [1, 5, 5, 5, 1], padding='SAME')
        biases = _bias_variable('biases', [out_filters])
        bias = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(bias, name=scope.name)

        prev_layer = conv1
        in_filters = out_filters

    pool1 = tf.nn.max_pool3d(prev_layer, ksize=[1, 3, 3, 3, 1], strides=[1, 1, 1, 1, 1], padding='SAME')
    norm1 = pool1  # tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta = 0.75, name='norm1')

    prev_layer = norm1

    with tf.variable_scope('conv2') as scope:
        out_filters = 32
        kernel = _weight_variable('weights', [5, 5, 5, in_filters, out_filters])
        conv = tf.nn.conv3d(prev_layer, kernel, [1, 3, 3, 3, 1], padding='SAME')
        biases = _bias_variable('biases', [out_filters])
        bias = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(bias, name=scope.name)

        prev_layer = conv2
        in_filters = out_filters

    # normalize prev_layer here
    prev_layer = tf.nn.max_pool3d(prev_layer, ksize=[1, 3, 3, 3, 1], strides=[1, 2, 2, 2, 1], padding='SAME')

    with tf.variable_scope('conv3_1') as scope:
        out_filters = 64
        kernel = _weight_variable('weights', [5, 5, 5, in_filters, out_filters])
        conv = tf.nn.conv3d(prev_layer, kernel, [1, 2, 2, 2, 1], padding='SAME')
        biases = _bias_variable('biases', [out_filters])
        bias = tf.nn.bias_add(conv, biases)
        prev_layer = tf.nn.relu(bias, name=scope.name)
        in_filters = out_filters

    with tf.variable_scope('conv3_2') as scope:
        out_filters = 64
        kernel = _weight_variable('weights', [5, 5, 5, in_filters, out_filters])
        conv = tf.nn.conv3d(prev_layer, kernel, [1, 1, 1, 1, 1], padding='SAME')
        biases = _bias_variable('biases', [out_filters])
        bias = tf.nn.bias_add(conv, biases)
        prev_layer = tf.nn.relu(bias, name=scope.name)
        in_filters = out_filters

    with tf.variable_scope('conv3_3') as scope:
        out_filters = 32
        kernel = _weight_variable('weights', [5, 5, 5, in_filters, out_filters])
        conv = tf.nn.conv3d(prev_layer, kernel, [1, 1, 1, 1, 1], padding='SAME')
        biases = _bias_variable('biases', [out_filters])
        bias = tf.nn.bias_add(conv, biases)
        prev_layer = tf.nn.relu(bias, name=scope.name)
        in_filters = out_filters

    # normalize prev_layer here
    prev_layer = tf.nn.max_pool3d(prev_layer, ksize=[1, 3, 3, 3, 1], strides=[1, 2, 2, 2, 1], padding='SAME')
    """
    with tf.variable_scope('local3') as scope:
        dim = np.prod(prev_layer.get_shape().as_list()[1:])
        prev_layer_flat = tf.reshape(prev_layer, [-1, dim])
        weights = _weight_variable('weights', [dim, FC_SIZE])
        biases = _bias_variable('biases', [FC_SIZE])
        local3 = tf.nn.relu(tf.matmul(prev_layer_flat, weights) + biases, name=scope.name)

        local3 = tf.nn.dropout(local3, keep_prob)

    prev_layer = local3
    """
    with tf.variable_scope('local4') as scope:
        dim = np.prod(prev_layer.get_shape().as_list()[1:])
        prev_layer_flat = tf.reshape(prev_layer, [-1, dim])
        weights = _weight_variable('weights', [dim, FC_SIZE])
        biases = _bias_variable('biases', [FC_SIZE])
        local4 = tf.nn.relu(tf.matmul(prev_layer_flat, weights) + biases, name=scope.name)
        local4 = tf.nn.dropout(local4, keep_prob)
    prev_layer = local4

    with tf.variable_scope('softmax_linear') as scope:
        dim = np.prod(prev_layer.get_shape().as_list()[1:])
        weights = _weight_variable('weights', [dim, N_CLASSES])
        biases = _bias_variable('biases', [N_CLASSES])
        softmax_linear = tf.add(tf.matmul(prev_layer, weights), biases, name=scope.name)

    return softmax_linear, prev_layer


def train_neural_network():
    x = tf.placeholder('float')
    y = tf.placeholder('float')

    list_batch = _list_batch
    list_labels = _list_labels

    prediction, prev_layer = convolutional_neural_network(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(cost)

    hm_epochs = 3

    with tf.Session() as sess:
        # 変数の読み込み
        saver = tf.train.Saver()
        saver.restore(sess, "model0309_simble/model.ckpt-16")

        for epoch in range(hm_epochs):
            logger.info('epoch: %s' % epoch)

            test_loss = 0
            test_num = 0
            try:
                for folder in LIST_FEATURES:
                    batch = list_batch[-1]
                    X = load_data2(batch, folder)
                    Y = [[0, 1] if lb == 1 else [1, 0] for lb in list_labels[-1]]
                    # logger.info('batch: %s Accuracy:' % i, accuracy.eval({x: X, y: Y}))

                    _, c = sess.run([optimizer, cost], feed_dict={x: X, y: Y, keep_prob: 0.8})
                    test_loss += c
                    test_num += len(batch)
            except Exception as e:
                logger.info(str(e))
            logger.info('test loss: %s' % (test_loss / test_num))

        save_path = saver.save(sess, "model0309_simble/model_pred.ckpt")
        logger.info("model saved %s" % save_path)

        df = pd.read_csv(STAGE1_SAMPLE_SUBMISSION)
        pred = []
        for i, patient_id in enumerate(df['id'].tolist()):
            if i % 10 == 0:
                logger.info("predict %s" % i)
            tmp = []
            for folder in LIST_FEATURES[:1]:
                try:
                    X = np.array([_load_data(patient_id, folder)])
                except EOFError:
                    continue
                p = sess.run(prediction, feed_dict={x: X, keep_prob: 1.})[0][1]
                # tmp.append(p)
            #p = np.array(tmp).mean(axis=0)
            pred.append(p)
        pred = np.array(pred)
        df['cancer'] = sigmoid(pred)  # tf.logits(pred)
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

    # data, labels = load_data()
    # If you are working with the basic sample data, use maybe 2 instead of
    # 100 here... you don't have enough data to really do this
    # Run this locally:

    train_neural_network()
