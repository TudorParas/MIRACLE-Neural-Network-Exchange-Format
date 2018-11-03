import tensorflow as tf

from linear_regression_experiment.mnist_data.mnist_data import MnistData, IMAGE_SIZE, LABEL_SIZE

BATCH_SIZE = 128
LEARNING_RATE = 1e-3
ITERATIONS = 4000

SUMMARIES_DIR = 'out/mnist_linear'

dataset = MnistData()

x, y = dataset.get_data()

with tf.name_scope('Linear_regression'):
    W = tf.Variable(tf.random_normal(shape=[IMAGE_SIZE, LABEL_SIZE]), "linear_matrix")
    bias = tf.Variable(tf.random_normal(shape=[LABEL_SIZE]), "bias")

    logits = tf.matmul(x, W) + bias

with tf.name_scope('Loss'):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y))

with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

tf.summary.scalar('accuracy', accuracy)

with tf.name_scope('Training'):
    train_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

with tf.Session() as sess:
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(SUMMARIES_DIR + '/train',
                                         sess.graph)

    sess.run(tf.global_variables_initializer())
    dataset.initialize_train_data(sess, batch_size=BATCH_SIZE)

    for i in range(ITERATIONS):
        summary, _ = sess.run(fetches=[merged, train_op])
        train_writer.add_summary(summary, i)
        if i % 100 == 0:
            train_accuracy = sess.run(fetches=accuracy)
            print("Train accuracy at iteration {0} is {1}".format(i, train_accuracy))

    # Test data
    dataset.initialize_test_data(sess)
    test_accuracy = sess.run(fetches=accuracy)

    print('Model test accuracy: {0}'.format(test_accuracy))
