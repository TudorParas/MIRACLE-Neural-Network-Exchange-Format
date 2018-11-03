"""This file is meant to create the toy data for training"""

import numpy as np
import tensorflow as tf


class ToyData(object):
    def __init__(self, samples=60000, train_size=50000, rows=10, num_outputs=2, ):
        input_data, output_data = create_data(samples, rows, num_outputs)
        self.train_data = input_data[:train_size], output_data[:train_size]
        self.test_data = input_data[train_size:], output_data[train_size:]
        # Set placeholders for test and train data and for the batch size
        with tf.name_scope('data'):
            self.x = tf.placeholder(tf.float32, shape=[None, rows])
            self.y = tf.placeholder(dtype=tf.float32, shape=[None, num_outputs])
        self.batch_size = tf.placeholder(tf.int64, name='batch_size')
        with tf.name_scope('dataset'):
            self.dataset = tf.data.Dataset.from_tensor_slices((self.x, self.y)). \
                shuffle(buffer_size=self.batch_size * 10).repeat().batch(self.batch_size)

            self.dataset_iterator = self.dataset.make_initializable_iterator()

    def get_data(self):
        return self.dataset_iterator.get_next()

    def initialize_train_data(self, sess, batch_size):
        """Initialize the dataset with the train data in a TensorFlow session"""
        sess.run(self.dataset_iterator.initializer, feed_dict={self.x: self.train_data[0], self.y: self.train_data[1],
                                                               self.batch_size: batch_size})

    def initialize_test_data(self, sess):
        """Initialize the dataset with the test data in a TensorFlow session. Use the full batch"""
        sess.run(self.dataset_iterator.initializer, feed_dict={self.x: self.test_data[0], self.y: self.test_data[1],
                                                               self.batch_size: self.test_data[0].shape[0]})




def create_data(samples=1000, rows=10, num_outputs=None):
    """
    Create toy data consisting of

    Parameters:
        samples: int
            Number of input-output pairs
        rows: int
            Number of rows of the input and of the output
        num_outputs: int
            Number of outputs we want
    Returns:
        tuple
            Tuple of array of input vectors and array of output vectors
    """

    input_data = np.random.random(size=(samples, rows))
    output_data = create_output_data(input_data, num_outputs)

    return input_data, output_data


def _sum_max_function(vector):
    """Given a numpy array return a numpy array of its sum and its average"""
    return np.array([vector.sum(), vector.mean()])


def _avg_function(vector):
    """Just return the sum"""
    return np.array([vector.mean()])


def create_output_data(input_data, num_outputs):
    """
    Create the output data by appying the function to all the output data

    Parameters:
        input_data: np.array
        func: np.array -> np.array

    Returns:
        np.array
            Apply func to each input vector
    """
    if num_outputs == 1:
        func = _avg_function
    else:
        func = _sum_max_function
    return np.array(list(map(func, input_data)))
