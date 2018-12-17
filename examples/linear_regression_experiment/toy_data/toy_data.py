"""This file is meant to create the toy data for training"""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

RANDOM_SEED = 53
LOWER_BOUND = -2
UPPER_BOUND = 2
SLOPE = 0.4

np.random.seed(RANDOM_SEED)


class ToyData(object):
    def __init__(self, samples=60000, test_size=0.1, slope=SLOPE):
        self.slope = slope
        input_data, output_data = self.create_data(samples, self.slope)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(input_data, output_data,
                                                                                test_size=test_size)
        # Set placeholders for test and train data and for the batch size
        with tf.name_scope('data'):
            self.x = tf.placeholder(tf.float32, shape=[None])
            self.y = tf.placeholder(dtype=tf.float32, shape=[None])
        self.batch_size = tf.placeholder(tf.int64, name='batch_size')
        with tf.name_scope('dataset'):
            self.dataset = tf.data.Dataset.from_tensor_slices((self.x, self.y)). \
                shuffle(buffer_size=self.batch_size * 10).repeat().batch(self.batch_size)

            self.dataset_iterator = self.dataset.make_initializable_iterator()

    def get_data(self):
        return self.dataset_iterator.get_next()

    def initialize_train_data(self, sess, batch_size):
        """Initialize the dataset with the train data in a TensorFlow session"""
        sess.run(self.dataset_iterator.initializer, feed_dict={self.x: self.X_train, self.y: self.y_train,
                                                               self.batch_size: batch_size})

    def initialize_test_data(self, sess):
        """Initialize the dataset with the test data in a TensorFlow session. Use the full batch"""
        sess.run(self.dataset_iterator.initializer, feed_dict={self.x: self.X_test, self.y: self.y_test,
                                                               self.batch_size: self.X_test.shape[0]})

    def plot_data(self, datapoints=500):
        """Plot some of the training data"""
        x_axis = self.X_train[:datapoints]
        y_points = self.y_train[:datapoints]
        fitted_line = x_axis * self.slope

        fig, ax = plt.subplots()
        ax.scatter(x_axis, y_points, label='Output points', alpha=0.2)
        ax.plot(x_axis, fitted_line, label="True line", color='r')
        # Stylistic changes
        ax.grid(True, which='both')

        ax.axhline(y=0, color='k')
        ax.axvline(x=0, color='k')
        ax.legend()

    def create_data(self, samples, slope):
        """
        Create toy data. The input is the input with gaussian noise added to it

        Parameters:
            samples: int
                Number of input-output pairs
            slope: int
                The slope we'll try to learn during training
        Returns:
            tuple
                Tuple of array of input vectors and array of output vectors
        """

        input_data = np.random.uniform(low=LOWER_BOUND, high=UPPER_BOUND, size=samples)
        output_data = input_data * slope + np.random.normal(size=samples, scale=self.slope / 4)

        return input_data, output_data

# t= ToyData()
# t.plot_data()
# plt.show()