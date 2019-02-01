import logging

import tensorflow as tf
from examples.miracle_graphs.LeNet5.mnist_data import MnistData, LABEL_SIZE

logging.getLogger().setLevel(logging.INFO)

SUMMARIES_DIR = 'out/graphs/mnist_tf'

BATCH_SIZE = 256
TRAIN_ITERATIONS = 20000

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
        wc1 = tf.Variable(tf.random_normal(shape=[5, 5, 1, 20]), name='wc1')
        bc1 = tf.Variable(tf.random_normal(shape=[20]), name='bc1')

        conv1 = conv2d_help(x, wc1, bc1, padding='VALID')
        logging.info("Result after convolution 1 shape: {}".format(conv1.shape))

    with tf.name_scope('maxpool_1'):
        maxp1 = maxpool2d_help(conv1, k=2, padding='SAME')
        logging.info("Result after maxpool 1 shape: {}".format(maxp1.shape))

    with tf.name_scope('convolution_2'):
        wc2 = tf.Variable(tf.random_normal(shape=[5, 5, 20, 50]), name='wc2')
        bc2 = tf.Variable(tf.random_normal(shape=[50]), name='bc2')

        conv2 = conv2d_help(maxp1, wc2, bc2, padding='VALID')
        logging.info("Result after convolution 2 shape: {}".format(conv2.shape))

    with tf.name_scope('maxpool_2'):
        maxp2 = maxpool2d_help(conv2, k=2, padding='SAME')
        logging.info("Result after maxpool 2 shape: {}".format(maxp2.shape))

    with tf.name_scope('fully_connected'):
        wd1 = tf.Variable(tf.random_normal(shape=[4 * 4 * 50, 500]), name='wd1')
        bd1 = tf.Variable(tf.random_normal(shape=[500]), name='bd1')

        # Reshape maxp2 to fit the fully connected layer input

        maxp2_reshaped = tf.reshape(maxp2, [-1, 4 * 4 * 50])

        fc = tf.add(tf.matmul(maxp2_reshaped, wd1), bd1)
        fc = tf.nn.relu(fc)
        logging.info("Result after fc layer shape: {}".format(fc.shape))

    with tf.name_scope('out'):
        out_W = tf.Variable(tf.random_normal(shape=[500, LABEL_SIZE]), name='out')
        out_bias = tf.Variable(tf.random_normal(shape=[LABEL_SIZE]), name='bout')

        logits = tf.add(tf.matmul(fc, out_W), out_bias)
        logging.info("Result after out layer shape: {}".format(logits.shape))

    prediction = tf.nn.softmax(logits, name='prediction')

    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')

    # Define loss and optimizer
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))

    global_step = tf.Variable(initial_value=0, name='global_step', trainable=False)

    learning_rate = tf.train.exponential_decay(
        0.001,  # Base learning rate.
        global_step,  # Current index into the dataset.
        30 * dataset.train_data[0].shape[0] / BATCH_SIZE,  # Decay step, once every 30 epochs
        1.,  # Decay rate.
        staircase=True)

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

    train_op = optimizer.minimize(loss)

with tf.Session() as sess:
    # Def print operations to output during training

    sess.run(tf.global_variables_initializer())
    dataset.initialize_train_data(sess, BATCH_SIZE)

    for iteration in range(TRAIN_ITERATIONS):
        sess.run(train_op)

        if iteration % 500 == 0:
            acc = sess.run(accuracy)
            print("Iteration {0}, Train Accuracy {1}".format(iteration, acc))

    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(SUMMARIES_DIR, sess.graph)

    dataset.initialize_test_data(sess)

    print("Final accuracy on test: {}".format(sess.run(accuracy)))
