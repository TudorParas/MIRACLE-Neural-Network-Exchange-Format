import tensorflow as tf

from linear_regression_experiment.toy_data import create_data

SAMPLES = 64
ROWS = 2000
OUTPUTS = 1

LEARNING_RATE = 1e-2
ITERATIONS = 10000

x = tf.placeholder(dtype=tf.float32, shape=[None, ROWS])
y = tf.placeholder(dtype=tf.float32, shape=[None, OUTPUTS])

W = tf.Variable(tf.random_normal(shape=[ROWS, OUTPUTS]), "linear_matrix")
bias = tf.Variable(tf.random_normal(shape=[OUTPUTS]), "bias")

predictions = tf.matmul(x, W) + bias

loss = tf.reduce_mean(tf.square(predictions - y))

train_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(ITERATIONS):
        input_data, output_data = create_data(SAMPLES, ROWS, OUTPUTS)
        loss_amount, _ = sess.run(fetches=[loss, train_op], feed_dict={x: input_data, y: output_data})

        if i % 500 == 0:
            print("Loss at iteration {0} is {1}".format(i, loss_amount))

    # Test data
    input_data, output_data = create_data(4, ROWS, OUTPUTS)

    predicted = sess.run(fetches=predictions, feed_dict={x: input_data, y: output_data})

    print("Expected values: {0}".format(output_data))
    print("Predicted values: {0}".format(predicted))

