"""Define a miracle graph in which we create the layers"""
import tensorflow as tf
import numpy as np


class MiracleGraph(object):
    def __init__(self, log_initial_sigma=1e-10, dtype=tf.float32):
        """
        Define the graph we'll use for constructing the miracle layers

        Parameters
        ----------
        log_initial_sigma: int
            Natural logarithm of the standard dev we want to initialize the variables with.
        dtype: tf.DType
            The default dtype we'll use for our floats.
        """
        self.log_initial_sigma = log_initial_sigma
        self.dtype = dtype
        # Define a variable used to accumulate all the parameters so that we can block them when doing the KL loss
        self.variables = list()

    def create_variable(self, shape, hash_group_size=1, scale=1):
        """
        Create a tensorflow variable where each parameter is a Gaussian.

        Parameters
        ----------
        shape: tuple
            Shape of the created variable
        hash_group_size: int
            The size of the hash_group. This reduces
        scale: int
            The standard deviation used when initializing the means
        Returns
        --------
        tf.Variable

        """
        nr_hasehd_vars = np.prod(shape) // hash_group_size  # this * hash_group_size = nr of vars hashed
        nr_leftover_vars = np.prod(shape) % hash_group_size  # leftover bc it doesn't divide perfectly
        nr_trained_vars = nr_hasehd_vars + nr_leftover_vars
        with tf.name_scope('mu'):
            # Create the mean for each var in the layer. The sampling stdev is scaled by first dim so that
            # Bigger layers are initialized closer to 0.
            mu_init = np.random.normal(size=nr_trained_vars, scale=scale)
            mu = tf.Variable(mu_init, dtype=self.dtype, name='mu')

        with tf.name_scope('sigma'):
            sigma_init = tf.fill([nr_trained_vars], tf.cast(self.log_initial_sigma, dtype=self.dtype), name='sigma_init')
            # We use log sigma as a Variable because we want sigma to always be positive
            sigma_var = tf.Variable(sigma_init, name='sigma_var')
            sigma = tf.exp(sigma_var)

        with tf.name_scope('weights'):
            # Applly the reparametrization trick to get the weights
            epsilon = tf.random_normal([nr_trained_vars], name='epsilon')
            weights = mu + epsilon * sigma

            with tf.name_scope('expand_weights'):
                expanded_weights = self._expand_variable(weights, shape, nr_hasehd_vars, hash_group_size)

        # Keep track of the actual weights, as they will be used to define the KL loss and during compression
        self.variables.append((mu, sigma))

        return expanded_weights


    def _expand_variable(self, var, shape, nr_hashed_vars, hash_group_size):
        """
        Given a flattened variable var, do:
            - repeat the first 'nr_hashed_vars' in 'var' a 'hash_group_size' nrber of times
            - append to this the last variables
            - reshape the created variable into 'shape'
        """
        hashed_vars = var[:nr_hashed_vars]
        expanded_hashed = tf.multiply(tf.expand_dims(hashed_vars, axis=1), np.ones(shape=hash_group_size),
                                      name='var_expansion')
        expanded_hashed = tf.reshape(expanded_hashed, shape=[-1], name='flatten')  # flatten them

        expanded_vars = tf.concat([expanded_hashed, var[nr_hashed_vars:]], axis=0, name='expanded_vars')

        return tf.reshape(expanded_vars, shape=shape)

    def create_kl_loss(self):
        """Create the kl loss that will be used during optimization"""
        pass