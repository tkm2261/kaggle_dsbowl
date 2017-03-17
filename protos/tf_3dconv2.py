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

DATA_PATH = '../features/'


#DATA_PATH = '../../data/features/'
FEATURE_FOLDER = DATA_PATH + 'features_20170314_range900_1154_fil_resize/'
FEATURE_FOLDER_TP = DATA_PATH + 'features_20170314_range900_1154_fil_rotate_y_p10/'
FEATURE_FOLDER_TN = DATA_PATH + 'features_20170314_range900_1154_fil_rotate_y_m10/'
FEATURE_FOLDER_YP = DATA_PATH + 'features_20170314_range900_1154_fil_rotate_t_p10/'
FEATURE_FOLDER_YN = DATA_PATH + 'features_20170314_range900_1154_fil_rotate_t_m10/'

# FEATURE_FOLDER_FILL = DATA_PATH + 'features_20170303_lung_binary_fill/'

LIST_FEATURE_FOLDER = [FEATURE_FOLDER] #, FEATURE_FOLDER_TP, FEATURE_FOLDER_TN, FEATURE_FOLDER_YP, FEATURE_FOLDER_YN]


IMG_SIZE = (200, 512, 512)

N_CLASSES = 2
BATCH_SIZE = 5
DROP_RATE = 0.5
HM_EPOCHS = 10000

MODEL_FOLDER = "model0316"

FC_SIZE = 32
DTYPE = tf.float32


df = pd.read_csv(STAGE1_LABELS)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=871)
for train_idx, test_idx in cv.split(df['cancer'].tolist(), df['cancer'].tolist()):
    df_train = df.ix[train_idx]
    df_test = df.ix[test_idx]

list_patient_id = df_train['id'].tolist()
labels = df_train['cancer'].tolist()

list_test_patient_id = df_test['id'].tolist()
test_labels = df_test['cancer'].tolist()


def train_neural_network():
    x = tf.placeholder(DTYPE)
    y = tf.placeholder(DTYPE)
    keep_prob = tf.placeholder(DTYPE)
    is_train = tf.placeholder(bool)
    list_batch = split_batch(list_patient_id, BATCH_SIZE)
    list_labels = split_batch(labels, BATCH_SIZE)

    list_test_batch = split_batch(list_test_patient_id, BATCH_SIZE)
    list_test_labels = split_batch(test_labels, BATCH_SIZE)

    prediction, prev_layer = convolutional_neural_network(x, keep_prob, is_train)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(cost)

    with tf.Session() as sess:
        if 1:
            sess.run(tf.initialize_all_variables())
            saver = tf.train.Saver()
        else:
            saver = tf.train.Saver()
            saver.restore(sess, MODEL_FOLDER + "model.ckpt-13")

        total_runs = 0

        for epoch in range(0, HM_EPOCHS):
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
                    # logger.info('batch: %s Accuracy:' % i, accuracy.eval({x: X, y: Y}))
                    _, c, prev = sess.run([optimizer, cost, prev_layer], feed_dict={
                                          x: X, y: Y, keep_prob: DROP_RATE, is_train: True})
                    epoch_loss += c
                    successful_runs += len(batch)

                except Exception as e:
                    logger.info(str(e))
                if cnt % 10 == 0:
                    logger.info('batch loss: %s %s' % (cnt, epoch_loss / successful_runs))

            list_prev = []
            for i in range(len(list_test_batch)):
                X = load_data2(list_test_batch[i])
                Y = [[0, 1] if lb == 1 else [1, 0] for lb in list_test_labels[i]]
                prev = sess.run(prev_layer, feed_dict={x: X, y: Y, keep_prob: 1., is_train: False})
                list_prev += [prev[j] for j in range(len(list_test_batch[i]))]
                if cnt % 10 == 0:
                    logger.debug('test batch: %s' % (i))

            clf = LogisticRegression(C=0.1, random_state=0)

            scores = cross_val_score(clf, list_prev, test_labels, cv=5, scoring='roc_auc', n_jobs=-1)
            logger.info('auc score: %s' % (scores.mean()))
            scores = cross_val_score(clf, list_prev, test_labels, cv=5, scoring='neg_log_loss', n_jobs=-1)
            logger.info('logloss score: %s' % (- scores.mean()))
            """
            df_prev = pd.DataFrame(list_prev)
            df_prev['id'] = list_test_batch
            df_prev['cancer'] = list_test_label
            df_prev.to_csv(MODEL_FOLDER + 'prev_%s.csv' % epoch, index=False)
            logger.info('prev data: {}'.format(df_prev.shape))
            """
            save_path = saver.save(sess, MODEL_FOLDER + "model.ckpt", global_step=epoch)
            logger.info("model saved %s" % save_path)


def convolutional_neural_network(x, keep_prob, is_train):
    x = tf.reshape(x, shape=[-1, IMG_SIZE[0], IMG_SIZE[1], IMG_SIZE[2], 1])

    prev_layer = x

    in_filters = 1
    with tf.variable_scope('conv1') as scope:
        out_filters = 2
        kernel = _weight_variable('weights', [5, 5, 5, in_filters, out_filters])
        conv = tf.nn.conv3d(prev_layer, kernel, [1, 2, 3, 3, 1], padding='SAME')
        biases = _bias_variable('biases', [out_filters])
        bias = tf.nn.bias_add(conv, biases)
        h2 = tf.contrib.layers.batch_norm(bias,
                                          center=True, scale=True,
                                          is_training=is_train,
                                          scope=scope.name)

        conv1 = tf.nn.relu(h2, name=scope.name)
        prev_layer = conv1
        in_filters = out_filters

    pool1 = tf.nn.max_pool3d(prev_layer, ksize=[1, 2, 3, 3, 1], strides=[1, 1, 1, 1, 1], padding='SAME')
    norm1 = pool1  # tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta = 0.75, name='norm1')

    prev_layer = norm1

    with tf.variable_scope('conv2') as scope:
        out_filters = 4
        kernel = _weight_variable('weights', [2, 5, 5, in_filters, out_filters])
        conv = tf.nn.conv3d(prev_layer, kernel, [1, 2, 3, 3, 1], padding='SAME')
        biases = _bias_variable('biases', [out_filters])
        bias = tf.nn.bias_add(conv, biases)
        h2 = tf.contrib.layers.batch_norm(bias,
                                          center=True, scale=True,
                                          is_training=is_train,
                                          scope=scope.name)
        conv2 = tf.nn.relu(h2, name=scope.name)
        prev_layer = conv2
        in_filters = out_filters

    # normalize prev_layer here
    prev_layer = tf.nn.max_pool3d(prev_layer, ksize=[1, 1, 3, 3, 1], strides=[1, 1, 2, 2, 1], padding='SAME')

    with tf.variable_scope('conv3_1') as scope:
        out_filters = 8
        kernel = _weight_variable('weights', [5, 5, 5, in_filters, out_filters])
        conv = tf.nn.conv3d(prev_layer, kernel, [1, 2, 2, 2, 1], padding='SAME')
        biases = _bias_variable('biases', [out_filters])
        bias = tf.nn.bias_add(conv, biases)
        h2 = tf.contrib.layers.batch_norm(bias,
                                          center=True, scale=True,
                                          is_training=is_train,
                                          scope=scope.name)

        prev_layer = tf.nn.relu(h2, name=scope.name)
        in_filters = out_filters

    with tf.variable_scope('conv3_2') as scope:
        out_filters = 8
        kernel = _weight_variable('weights', [5, 5, 5, in_filters, out_filters])
        conv = tf.nn.conv3d(prev_layer, kernel, [1, 1, 1, 1, 1], padding='SAME')
        biases = _bias_variable('biases', [out_filters])
        bias = tf.nn.bias_add(conv, biases)
        h2 = tf.contrib.layers.batch_norm(bias,
                                          center=True, scale=True,
                                          is_training=is_train,
                                          scope=scope.name)

        prev_layer = tf.nn.relu(h2, name=scope.name)
        in_filters = out_filters

    with tf.variable_scope('conv3_3') as scope:
        out_filters = 16
        kernel = _weight_variable('weights', [5, 5, 5, in_filters, out_filters])
        conv = tf.nn.conv3d(prev_layer, kernel, [1, 1, 1, 1, 1], padding='SAME')
        biases = _bias_variable('biases', [out_filters])
        bias = tf.nn.bias_add(conv, biases)
        h2 = tf.contrib.layers.batch_norm(bias,
                                          center=True, scale=True,
                                          is_training=is_train,
                                          scope=scope.name)

        prev_layer = tf.nn.relu(h2, name=scope.name)
        in_filters = out_filters

    # normalize prev_layer here
    prev_layer = tf.nn.max_pool3d(prev_layer, ksize=[1, 3, 3, 3, 1], strides=[1, 2, 2, 2, 1], padding='SAME')

    with tf.variable_scope('local3') as scope:
        dim = np.prod(prev_layer.get_shape().as_list()[1:])
        prev_layer_flat = tf.reshape(prev_layer, [-1, dim])
        weights = _weight_variable('weights', [dim, FC_SIZE])
        biases = _bias_variable('biases', [FC_SIZE])

        h1 = tf.matmul(prev_layer_flat, weights) + biases
        h2 = tf.contrib.layers.batch_norm(h1,
                                          center=True, scale=True,
                                          is_training=is_train,
                                          scope=scope.name)
        local3 = tf.nn.relu(h2, name=scope.name)
        local3 = tf.nn.dropout(local3, keep_prob)

    prev_layer = local3

    with tf.variable_scope('local4') as scope:
        dim = np.prod(prev_layer.get_shape().as_list()[1:])
        prev_layer_flat = tf.reshape(prev_layer, [-1, dim])
        weights = _weight_variable('weights', [dim, FC_SIZE])
        biases = _bias_variable('biases', [FC_SIZE])
        h1 = tf.matmul(prev_layer_flat, weights) + biases
        h2 = tf.contrib.layers.batch_norm(h1,
                                          center=True, scale=True,
                                          is_training=is_train,
                                          scope=scope.name)
        local4 = tf.nn.relu(h2, name=scope.name)
        local4 = tf.nn.dropout(local4, keep_prob)
    prev_layer = local4

    with tf.variable_scope('softmax_linear') as scope:
        dim = np.prod(prev_layer.get_shape().as_list()[1:])
        # prev_layer_flat = tf.reshape(prev_layer, [-1, dim])
        weights = _weight_variable('weights', [dim, N_CLASSES])
        biases = _bias_variable('biases', [N_CLASSES])
        h1 = tf.matmul(prev_layer, weights)

        softmax_linear = tf.add(h1, biases, name=scope.name)

    return softmax_linear, prev_layer


def split_batch(list_data, batch_size):
    ret = []
    for i in range(int(len(list_data) / batch_size) + 1):
        from_idx = i * batch_size
        next_idx = from_idx + batch_size if from_idx + batch_size <= len(list_data) else len(list_data)

        if from_idx >= next_idx:
            break

        ret.append(list_data[from_idx:next_idx])
    return ret


def load_data2(batch):
    """
    p = Pool(len(batch))
    ret = p.map(_load_data, batch)
    p.close()
    p.join()
    """
    return [_load_data(p) for p in batch]


def _load_data(patient_id):
    path = random.choice(LIST_FEATURE_FOLDER)
    with gzip.open(path + patient_id + '.pkl.gz', 'rb') as f:
        img = pickle.load(f)
    return img


def _weight_variable(name, shape):
    return tf.get_variable(name, shape, DTYPE, tf.truncated_normal_initializer(stddev=0.1))


def _bias_variable(name, shape):
    return tf.get_variable(name, shape, DTYPE, tf.constant_initializer(0.1, dtype=DTYPE))


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
