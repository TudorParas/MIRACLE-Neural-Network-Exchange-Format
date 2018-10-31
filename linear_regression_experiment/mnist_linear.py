import tensorflow as tf
from linear_regression_experiment.mnist_data import MnistData


IMAGE_SIZE = 28 * 28
LABEL_SIZE = 10  # 10 classes

BATCH_SIZE = 64
LEARNING_RATE = 1e-2
ITERATIONS = 5000


dataset = MnistData()

x, y = dataset.get_data()

W = tf.Variable(tf.random_normal(shape=[IMAGE_SIZE, LABEL_SIZE]), "linear_matrix")
bias = tf.Variable(tf.random_normal(shape=[LABEL_SIZE]), "bias")

logits = tf.matmul(x, W) + bias

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))

correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

train_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    dataset.initialize_train_data(sess, batch_size=BATCH_SIZE)

    for i in range(ITERATIONS):
        _ = sess.run(fetches=train_op)

        if i % 500 == 0:
            train_accuracy = sess.run(fetches=accuracy)
            print("Accuracy at iteration {0} is {1}".format(i, train_accuracy))

    # Test data
    dataset.initialize_test_data(sess)
    test_accuracy = sess.run(fetches=accuracy)

    print('Model test accuracy: {0}'.format(test_accuracy))
