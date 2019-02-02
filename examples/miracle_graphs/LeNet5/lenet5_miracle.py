import logging

import tensorflow as tf
import numpy as np

import miracle
from examples.miracle_graphs.LeNet5.mnist_data import MnistData, LABEL_SIZE

logging.getLogger().setLevel(logging.INFO)

SUMMARIES_DIR = 'out/graphs/mnist_miracle/no_opt'
COMPRESSED_FILES_DIR = 'out/compressed_files/miracle'

BATCH_SIZE = 256
BLOCK_SIZE_VARS = 30
BITS_PER_BLOCK = 10

PRETRAIN_ITERATIONS = 10000
TRAIN_ITERATIONS = 70000
RETRAIN_ITERATIONS = 100

COMPRESSED_FILE_NAME = 'lenet5_{0}_{1}.mrcl'.format(BLOCK_SIZE_VARS, BITS_PER_BLOCK)
COMPRESSED_FILE_PATH = '{}/{}'.format(COMPRESSED_FILES_DIR, COMPRESSED_FILE_NAME)

dataset = MnistData()

x, y = dataset.get_data()


# Define helper functions
def conv2d_help(x, W, b, padding='SAME', strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding=padding)
    x = tf.nn.bias_add(x, b)

    return tf.nn.relu(x)


def maxpool2d_help(x, k=2, padding='SAME'):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding=padding)


with tf.name_scope('Lenet5'):
    with tf.name_scope('convolution_1'):
        wc1 = miracle.create_variable(shape=[5, 5, 1, 20], hash_group_size=1, name='wc1')
        bc1 = miracle.create_variable(shape=[20], hash_group_size=1, name='bc1')

        conv1 = conv2d_help(x, wc1, bc1, padding='VALID')
        logging.info("Result after convolution 1 shape: {}".format(conv1.shape))

    with tf.name_scope('maxpool_1'):
        maxp1 = maxpool2d_help(conv1, k=2, padding='SAME')
        logging.info("Result after maxpool 1 shape: {}".format(maxp1.shape))

    with tf.name_scope('convolution_2'):
        wc2 = miracle.create_variable(shape=[5, 5, 20, 50], hash_group_size=2, name='wc2')
        bc2 = miracle.create_variable(shape=[50], hash_group_size=1, name='bc2')

        conv2 = conv2d_help(maxp1, wc2, bc2, padding='VALID')
        logging.info("Result after convolution 2 shape: {}".format(conv2.shape))

    with tf.name_scope('maxpool_2'):
        maxp2 = maxpool2d_help(conv2, k=2, padding='SAME')
        logging.info("Result after maxpool 2 shape: {}".format(maxp2.shape))

    with tf.name_scope('fully_connected'):
        wd1 = miracle.create_variable(shape=[4 * 4 * 50, 500], hash_group_size=50, name='wd1')
        bd1 = miracle.create_variable(shape=[500], hash_group_size=1, name='bd1')

        # Reshape maxp2 to fit the fully connected layer input

        maxp2_reshaped = tf.reshape(maxp2, [-1, 4 * 4 * 50])

        fc = tf.add(tf.matmul(maxp2_reshaped, wd1), bd1)
        fc = tf.nn.relu(fc)
        logging.info("Result after fc layer shape: {}".format(fc.shape))

    with tf.name_scope('out'):
        out_W = miracle.create_variable(shape=[500, LABEL_SIZE], hash_group_size=1, name='out')
        out_bias = miracle.create_variable(shape=[LABEL_SIZE], hash_group_size=1, name='bout')

        logits = tf.add(tf.matmul(fc, out_W), out_bias)
        logging.info("Result after out layer shape: {}".format(logits.shape))

    prediction = tf.nn.softmax(logits, name='prediction')

    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')

    # Define loss and optimizer
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))

    global_step = tf.train.get_or_create_global_step()

    learning_rate = tf.train.exponential_decay(
        learning_rate=0.001,  # Base learning rate.
        global_step=global_step,  # Current index into the dataset.
        decay_steps=30 * dataset.train_data[0].shape[0] / BATCH_SIZE,  # Decay step, once every 30 epochs
        decay_rate=1.,  # Decay rate.
        staircase=True)

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

miracle.create_compression_graph(loss=loss, optimizer=optimizer,
                                 bits_per_block=BITS_PER_BLOCK,
                                 block_size_vars=BLOCK_SIZE_VARS)

with tf.Session() as sess:
    # Def print operations to output during training
    def print_train(iteration):
        """Execute this after every iteration iterations"""
        if iteration % 500 == 0:
            acc, current_loss, kl_loss, mean_kl, kl_target = sess.run([accuracy, loss, miracle.graph.kl_loss,
                                                                       miracle.graph.mean_kl, miracle.graph.kl_target])
            print("\nIteration {0}, Train Accuracy {1}, Loss {2}, KL loss {3}, Mean KL_2 {4}".format(
                iteration, acc, current_loss, kl_loss, mean_kl / np.log(2.), kl_target))
            # y_ev, pred_ev = sess.run([y, prediction])
            # print("Labels {}\nPredictions {}".format(y_ev[0], pred_ev[0]))

            # mu, sigma = miracle.graph.variables[0]
            # print("MU: {}".format(sess.run(mu)[:10]))
            # print("Sigma: {}".format(sess.run(sigma[:10])))

    def print_retrain():
        """Print the accuracy after every iteration"""
        acc = sess.run(accuracy)
        print("Train Accuracy: {}\n".format(acc))


    sess.run(tf.global_variables_initializer())
    dataset.initialize_train_data(sess, BATCH_SIZE)

    miracle.assign_session(sess)

    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(SUMMARIES_DIR, sess.graph)

    # miracle.pretrain(iterations=PRETRAIN_ITERATIONS, f=print_train)
    # miracle.train(iterations=TRAIN_ITERATIONS, f=print_train)
    # miracle.compress(retrain_iterations=RETRAIN_ITERATIONS, out_file=COMPRESSED_FILE_PATH, f=print_retrain)

    miracle.load(COMPRESSED_FILE_PATH)
    dataset.initialize_test_data(sess)

    print("Final accuracy on test: {}".format(sess.run(accuracy)))
