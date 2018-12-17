import tensorflow as tf
from tensorflow.keras.datasets import mnist

IMAGE_SIZE = 28 * 28
LABEL_SIZE = 10  # 10 classes


class MnistData(object):
    def __init__(self):
        self.train_data, self.test_data = mnist.load_data()
        # Set placeholders for test and train data and for the batch size
        with tf.name_scope('data'):
            self.x = tf.placeholder(tf.float32, shape=[None, 28, 28])
            self.y = tf.placeholder(dtype=tf.int32, shape=[None])
        self.batch_size = tf.placeholder(tf.int64, name='batch_size')
        with tf.name_scope('dataset'):
            self.dataset = tf.data.Dataset.from_tensor_slices((self.x, self.y)).map(self.preprocess_data). \
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

    def preprocess_data(self, images, labels):
        return self._preprocess_images(images), self._preprocess_labels(labels)

    def _preprocess_images(self, images):
        # Flatten the images and scale them to be between 0 and 1
        images = tf.reshape(images, [IMAGE_SIZE])
        images = tf.cast(images, tf.float32) / 255.0

        return images

    def _preprocess_labels(self, labels):
        # One hot encode the labels
        return tf.one_hot(labels, LABEL_SIZE)
