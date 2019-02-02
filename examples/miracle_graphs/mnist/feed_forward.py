import miracle
import tensorflow as tf
import logging

from examples.linear_regression_experiment.mnist_data.mnist_data import MnistData, IMAGE_SIZE, LABEL_SIZE


logging.getLogger().setLevel(logging.INFO)

SUMMARIES_DIR = 'out/mnist_feed_forward'
COMPRESSED_FILES_DIR = 'out/compressed_files'

BATCH_SIZE = 128
HASH_GROUP_SIZE = 1
BLOCK_SIZE_VARS = 30
BITS_PER_BLOCK = 10

COMPRESSED_FILE_NAME = 'mnist_{0}_{1}_{2}.mrcl'.format(BLOCK_SIZE_VARS, BITS_PER_BLOCK, HASH_GROUP_SIZE)
COMPRESSED_FILE_PATH = '{}/{}'.format(COMPRESSED_FILES_DIR, COMPRESSED_FILE_NAME)

dataset = MnistData()

x, y = dataset.get_data()
with tf.name_scope('graph'):
    with tf.name_scope('W'):
        W = miracle.create_variable(shape=[IMAGE_SIZE, LABEL_SIZE], hash_group_size=HASH_GROUP_SIZE)
    with tf.name_scope('bias'):
        bias = miracle.create_variable(shape=[LABEL_SIZE])
    logits = tf.matmul(x, W) + bias

with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

with tf.name_scope('Loss'):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y))

miracle.create_compression_graph(loss, block_size_vars=BLOCK_SIZE_VARS, bits_per_block=BITS_PER_BLOCK)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
dataset.initialize_train_data(sess, BATCH_SIZE)

miracle.assign_session(sess)


def print_train(iteration):
    """Print the accuracy once every 1000 iterations"""
    if iteration % 1000 == 0:
        acc, mean_kl = sess.run([accuracy, miracle.graph.mean_kl])
        print("Iteration {0}, Train Accuracy {1}, Mean KL {2}".format(iteration, acc, mean_kl))


def print_retrain():
    """Print the accuracy after every iteration"""
    acc = sess.run(accuracy)
    print("Train Accuracy: {}\n".format(acc))


miracle.pretrain(iterations=4000, f=print_train)
miracle.train(iterations=120000, f=print_train)
miracle.compress(retrain_iterations=10, out_file=COMPRESSED_FILE_PATH, f=print_retrain)

miracle.load(model_file=COMPRESSED_FILE_PATH)

merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter(SUMMARIES_DIR, sess.graph)

dataset.initialize_test_data(sess)

print("Final accuracy on test: {}".format(sess.run(accuracy)))


