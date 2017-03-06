import tensorflow as tf
import numpy as np
import pandas as pd
import gzip
import pickle
from multiprocessing import Pool
from scipy import ndimage as nd

from sklearn.model_selection import StratifiedKFold

DATA_PATH = '../../'
STAGE1_FOLDER = DATA_PATH + 'stage1/'
STAGE1_LABELS = DATA_PATH + 'stage1_labels.csv'

DATA_PATH = '../features/'
FEATURE_FOLDER = DATA_PATH + 'features_20170303_lung_binary_resize/'
# FEATURE_FOLDER_FILL = DATA_PATH + 'features_20170303_lung_binary_fill/'


from logging import getLogger

logger = getLogger(__name__)

IMG_SIZE = (512, 512, 200)

N_CLASSES = 2
BATCH_SIZE = 10

df = pd.read_csv(STAGE1_LABELS)
list_patient_id = df['id'].tolist()
labels = df['cancer'].tolist()


def split_batch(list_data, batch_size):
    ret = []
    for i in range(int(len(list_data) / batch_size) + 1):
        from_idx = i
        next_idx = i + batch_size if i + batch_size <= len(list_data) else len(list_data)

        if from_idx >= next_idx:
            break
        ret.append(list_data[from_idx:next_idx])
    return ret

list_batch = split_batch(list_patient_id, BATCH_SIZE)
list_labels = split_batch(labels, BATCH_SIZE)


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


def conv3d(x, W):
    return tf.nn.conv3d(x, W, strides=[1, 5, 5, 5, 1], padding='SAME')


def maxpool3d(x):
    #                        size of window         movement of window as you slide about
    return tf.nn.max_pool3d(x, ksize=[1, 3, 3, 3, 1], strides=[1, 2, 2, 2, 1], padding='SAME')


def convolutional_neural_network(x, keep_rate=0.8):
    #                # 5 x 5 x 5 patches, 1 channel, 32 features to compute.
    weights = {'W_conv1': tf.Variable(tf.random_normal([15, 15, 15, 1, 32])),
               #       5 x 5 x 5 patches, 32 channels, 64 features to compute.
               'W_conv2': tf.Variable(tf.random_normal([3, 3, 3, 32, 64])),
               #                                  64 features
               'W_fc': tf.Variable(tf.random_normal([4608, 1024])),
               'out': tf.Variable(tf.random_normal([1024, N_CLASSES]))}

    biases = {'b_conv1': tf.Variable(tf.random_normal([32])),
              'b_conv2': tf.Variable(tf.random_normal([64])),
              'b_fc': tf.Variable(tf.random_normal([1024])),
              'out': tf.Variable(tf.random_normal([N_CLASSES]))}

    #                            image X      image Y        image Z
    x = tf.reshape(x, shape=[-1, IMG_SIZE[0], IMG_SIZE[1], IMG_SIZE[2], 1])

    conv1 = tf.nn.relu(conv3d(x, weights['W_conv1']) + biases['b_conv1'])
    conv1 = maxpool3d(conv1)

    conv2 = tf.nn.relu(conv3d(conv1, weights['W_conv2']) + biases['b_conv2'])
    conv2 = maxpool3d(conv2)

    fc = tf.reshape(conv2, [-1, 4608])
    fc = tf.nn.relu(tf.matmul(fc, weights['W_fc']) + biases['b_fc'])
    fc = tf.nn.dropout(fc, keep_rate)

    output = tf.matmul(fc, weights['out']) + biases['out']

    return output


def train_neural_network():
    x = tf.placeholder('float')
    y = tf.placeholder('float')

    prediction = convolutional_neural_network(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(cost)

    hm_epochs = 10000

    test_data = load_data2(list_batch[-1])
    test_label = list_labels[-1]

    correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        saver = tf.train.Saver()
        total_runs = 0

        for epoch in range(hm_epochs):
            logger.info('epoch: %s' % epoch)
            epoch_loss = 0
            successful_runs = 0
            for i, batch in enumerate(list_batch[:-1]):
                total_runs += 1
                logger.info('batch: {}, batch_size: {}'.format(i, len(batch)))
                try:
                    X = load_data2(batch)
                    Y = [[0, 1] if lb == 1 else [1, 0] for lb in list_labels[i]]
                    #logger.info('batch: %s Accuracy:' % i, accuracy.eval({x: X, y: Y}))

                    for j in range(len(Y)):
                        _, c = sess.run([optimizer, cost], feed_dict={x: X[j], y: Y[j]})
                        epoch_loss += c
                        successful_runs += 1
                        logger.debug('   loss: %s' % (epoch_loss / successful_runs))
                except Exception as e:
                    logger.info(str(e))
                logger.info('batch loss: %s' % (epoch_loss / successful_runs))

            test_loss = 0
            test_num = 0
            try:
                X = load_data2(list_batch[-1])
                Y = [[0, 1] if lb == 1 else [1, 0] for lb in list_labels[-1]]
                #logger.info('batch: %s Accuracy:' % i, accuracy.eval({x: X, y: Y}))

                for j in range(len(Y)):
                    c = sess.run(cost, feed_dict={x: X[j], y: Y[j]})
                    test_loss += c
                    test_num += 1
            except Exception as e:
                logger.info(str(e))
            logger.info('test loss: %s' % (test_loss / test_num))

            for j in range(len(Y)):
                try:
                    _, c = sess.run([optimizer, cost], feed_dict={x: X[j], y: Y[j]})
                except Exception as e:
                    logger.info(str(e))

            save_path = saver.save(sess, "model.ckpt")
            logger.info("model saved %s" % save_path)
        logger.info('Done. Finishing accuracy:')


if __name__ == '__main__':
    from logging import StreamHandler, DEBUG, Formatter, FileHandler

    log_fmt = Formatter('%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s ')
    handler = FileHandler('tf_3dconv.log', 'w')
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
