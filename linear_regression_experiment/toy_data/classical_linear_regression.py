import tensorflow as tf

from linear_regression_experiment.toy_data.toy_data import ToyData

SAMPLES = 10000
SLOPE = 0.5

INITIAL_W_VALUE = 0
BATCH_SIZE = 64
LEARNING_RATE = 1e-2
ITERATIONS = 300

SUMMARIES_DIR = 'out/classic_linear'

dataset = ToyData(samples=SAMPLES, slope=SLOPE)

x, y = dataset.get_data()

with tf.name_scope('Linear_regression'):
    W = tf.Variable(tf.random_normal([1]), dtype=tf.float32, name="W")

    predictions = x * W

with tf.name_scope('Loss'):
    loss = tf.reduce_mean(tf.square(predictions - y))

with tf.name_scope('Training'):
    train_op = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    dataset.initialize_train_data(sess, batch_size=BATCH_SIZE)

    tf.summary.FileWriter(SUMMARIES_DIR + '/train',
                          sess.graph)

    for i in range(ITERATIONS):
        loss_amount, _ = sess.run(fetches=[loss, train_op])

        if i % 100 == 0:
            print("Loss at iteration {0} is {1}".format(i, loss_amount))

    # Test data
        dataset.initialize_test_data(sess)

    fitted_w = sess.run(fetches=W)

    print("Expected W: {0}".format(SLOPE))
    print("Fitted W: {0}".format(fitted_w[0]))

