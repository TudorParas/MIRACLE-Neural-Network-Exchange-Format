import numpy as np
import tensorflow as tf

from linear_regression_experiment.toy_data.toy_data import ToyData

# Data parameters
SAMPLES = 60000
ROWS = 10
OUTPUTS = 2

# Parameters for defining the graph
DTYPE = tf.float32
WEIGHT_DECAY = 5e-4
LOG_INITIAL_SIGMA = -10.
LOG_P_INITIAL_SIGMA = -2.

# Training parameters
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
ITERATIONS = 40000
INITIAL_KL_PENALTY = 1e-08
KL_PENALTY_STEP = 1.00005  # Should be > 1

# Tensorboard parameters
SUMMARIES_DIR = 'out/basic_miracle'


class BasicMiracle(object):
    def __init__(self, compressed_size):
        """
        Define the graph for doing linear regression using the basic algorithm

        Parameters
        ----------
        compressed_size: int
            Size of the final compressed model in bits
        """
        self.dataset = ToyData(samples=SAMPLES, rows=ROWS, num_outputs=OUTPUTS)
        self.kl_target = tf.constant(compressed_size * np.log(2), dtype=DTYPE)  # Transform from bits to nats.
        self.kl_penalty_step = KL_PENALTY_STEP  # How much we increase/decrease penalty if exceeding/not target KL
        self._create_graph()

    def _create_graph(self):
        """Create a graph of the linear regression matrix which we'll compress and the bias which we won't"""
        self.x, self.y = self.dataset.get_data()
        with tf.name_scope('Linear_regression'):
            weight_matrix = self._create_linear_reg_matrix()
            bias = tf.Variable(tf.random_normal(shape=[OUTPUTS]), "bias")

            predictions = tf.matmul(self.x, weight_matrix) + bias
        with tf.name_scope('Loss'):
            self._create_kl_loss()
            self.loss = tf.reduce_mean(tf.square(predictions - self.y)) + self.kl_loss

        with tf.name_scope('Training'):
            optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
            self.train_op = optimizer.minimize(self.loss)

            # Operation that does not train the standard deviation for the prior
            no_scales_list = [v for v in tf.trainable_variables() if v is not self.p_scale_var]
            self.train_op_no_scales = optimizer.minimize(self.loss, var_list=no_scales_list)

    def _create_prior(self):
        # Create the "prior" and the shared source of randomness. The variance of the prior is trainable
        self.p_scale_var = tf.Variable(LOG_INITIAL_SIGMA, dtype=DTYPE)
        self.p_scale = tf.exp(self.p_scale_var)
        self.p = tf.contrib.distributions.Normal(loc=0., scale=self.p_scale)

    def _create_linear_reg_matrix(self):
        """Create a gaussian for each variable. Weigh their variance by their size"""
        # Define a 10 x 2 matrix of Gaussians.
        shape = [ROWS, OUTPUTS]

        # Mean of each variable
        mu_init = np.random.normal(size=shape, loc=0., scale=np.sqrt(1. / shape[0]))
        self.mu = tf.Variable(mu_init, dtype=DTYPE, name='mu')
        # Variance for all weight blocks initilized to 1e-10. We want the exponenet to be the trained variable.
        self.sigma_var = tf.Variable(tf.fill(shape, tf.cast(LOG_P_INITIAL_SIGMA, dtype=DTYPE, name='sigma')))
        self.sigma = tf.exp(self.sigma_var)
        epsilon = tf.random_normal(shape)

        variational_weights = self.mu + epsilon * self.sigma

        return variational_weights

    def _create_kl_loss(self):
        """Create the KL loss which states how 'far away' our distribution is from the prior"""
        # Create the distribution which we use to compute the KL loss
        self.w_dist = tf.distributions.Normal(loc=self.mu, scale=self.sigma)
        # Only one KL penalty as we don't use blocks
        self.kl_penalty = tf.Variable(INITIAL_KL_PENALTY, trainable=False)
        # Compute the current kl
        current_kl = tf.distributions.kl_divergence(self.w_dist, self.p)

        # Set it as variable so that we can easily enable / disable kl loss
        self.enable_kl_loss = tf.Variable(1., dtype=DTYPE, trainable=False)  # 0 or 1
        # Define the KL loss as the actual loss multiplied by the penalty and the enable.
        self.kl_loss = current_kl * self.kl_penalty * self.enable_kl_loss

        # Operation to update the penalty in case the kl_loss exceeds the target kl
        self.kl_penalty_update = self.kl_penalty.assign(
            tf.cond(pred=tf.greater(current_kl, self.kl_target),
                    true_fn=self.kl_penalty * self.kl_penalty_step,  # Increase penalty
                    false_fn=self.kl_penalty / self.kl_penalty_step)  # Decrease penalty
        )

    def _compressor(self):
        """Compress the linear regression matrix"""
        pass
