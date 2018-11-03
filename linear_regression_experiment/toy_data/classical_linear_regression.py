import tensorflow as tf

from linear_regression_experiment.toy_data.toy_data import ToyData

SAMPLES = 60000
ROWS = 10
OUTPUTS = 2

BATCH_SIZE = 128
LEARNING_RATE = 1e-3
ITERATIONS = 4000

SUMMARIES_DIR = 'out/classic_linear'

dataset = ToyData(samples=SAMPLES, rows=ROWS, num_outputs=OUTPUTS)

x, y = dataset.get_data()

with tf.name_scope('Linear_regression'):
    W = tf.Variable(tf.random_normal(shape=[ROWS, OUTPUTS]), "linear_matrix")
    bias = tf.Variable(tf.random_normal(shape=[OUTPUTS]), "bias")

    predictions = tf.matmul(x, W) + bias

with tf.name_scope('Loss'):
    loss = tf.reduce_mean(tf.square(predictions - y))

with tf.name_scope('Training'):
    train_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    dataset.initialize_train_data(sess, batch_size=BATCH_SIZE)

    tf.summary.FileWriter(SUMMARIES_DIR + '/train',
                          sess.graph)

    for i in range(ITERATIONS):
        loss_amount, _ = sess.run(fetches=[loss, train_op])

        if i % 500 == 0:
            print("Loss at iteration {0} is {1}".format(i, loss_amount))

    # Test data
        dataset.initialize_test_data(sess)

    predicted, expected = sess.run(fetches=[predictions, y])

    print("Expected values: {0}".format(expected))
    print("Predicted values: {0}".format(predicted))

