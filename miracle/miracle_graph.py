"""Define a miracle graph in which we create the layers"""
import numpy as np
import tensorflow as tf


class MiracleGraph(object):
    def __init__(self, block_size_vars=10, log_initial_sigma=1e-10, log_p_initial_sigma=1e-2, train_p_scale=True,
                 dtype=tf.float32, seed=53,
                 ):
        """
        Define the graph we'll use for constructing the miracle layers

        Parameters
        ----------
        block_size_vars: int
            How many variables we assign per block.
        log_initial_sigma: int
            Natural logarithm of the standard dev we want to initialize the variables with.
        log_p_initial_sigma: int
             Natural logarithm of the standard dev we want to initialize the prior variables with.
        train_p_scale: bool
            Whether we train the standard dev of the layers. Recommended to leave as default.
        dtype: tf.DType
            The default dtype we'll use for our floats.
        seed: int
            Seed used for seeding numpy. Needed for the compression.
            Decompression has to be made with the same numpy seed
        """
        self.block_size_vars = block_size_vars
        self.log_initial_sigma = log_initial_sigma
        self.log_p_initial_sigma = log_p_initial_sigma
        self.train_p_scale = train_p_scale
        self.dtype = dtype
        # Define a lsit used to accumulate all the parameters so that we can block them when doing the KL loss
        self.variables = list()
        # Define a list that accumulates the p_scale for each layer
        self.p_scale_vars = list()
        # Define list that accumulates shapes of each layer
        self.shapes = list()

        # Variable that keeps track of whether we've initialized the compressor
        self.initialized_compressor = False

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
        if self.initialized_compressor:
            # ToDo: make this thing a warning
            print("WARNING: compressor has already been initialized. Adding other variables might not lead to the "
                  "desired result")
        nr_hasehd_vars = np.prod(shape) // hash_group_size  # this * hash_group_size = nr of vars hashed
        nr_leftover_vars = np.prod(shape) % hash_group_size  # leftover bc it doesn't divide perfectly
        nr_trained_vars = nr_hasehd_vars + nr_leftover_vars
        with tf.name_scope('mu'):
            # Create the mean for each var in the layer. The sampling stdev is scaled by first dim so that
            # Bigger layers are initialized closer to 0.
            mu_init = np.random.normal(size=nr_trained_vars, scale=scale)
            mu = tf.Variable(mu_init, dtype=self.dtype, name='mu')

        with tf.name_scope('sigma'):
            sigma_init = tf.fill([nr_trained_vars], tf.cast(self.log_initial_sigma, dtype=self.dtype),
                                 name='sigma_init')
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
        # Create the prior scale for this layer
        p_scale_var = tf.Variable(self.log_p_initial_sigma, dtype=self.dtype, trainable=self.train_p_scale)
        self.p_scale_vars.append(p_scale_var)
        self.shapes.append(shape)

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
        """Create the kl loss that will be used during optimization. During this we'll also initialize the compressor.
        The create_variable function should no longer be used after calling this function."""
        self.initialized_compressor = True
        print("Creating KL loss for {} layers".format(len(self.variables)))
        # Compute some parameters we'll need during creating the loss
        self._create_loss_parameters()
        self._create_mu_sigma()
        self._create_p_scale()

        # Define the prior distribution and the variational distribution
        w_dist = tf.contrib.distribution.Normal(loc=self.mu, scale=self.sigma)
        p = tf.contrib.distributions.Normal(loc=0, scale=self.p_scale)

    def _create_loss_parameters(self):
        """Create variables that will help in creating the KL loss"""
        self.nr_layers = len(self.shapes)
        self.layer_sizes = [np.prod(shape) for shape in self.shapes]
        self.nr_actual_vars = sum(self.layer_sizes)
        self.nr_blocks = int(np.ceil(self.nr_actual_vars / self.block_size_vars))
        self.accb_shape = [self.nr_blocks, self.block_size_vars]  # Shape of the accumulated blocks. One block per row
        # The last block can be not fully filled, so we'll have variables not used in inference, but used for kl loss.
        self.nr_train_vars = np.prod(self.accb_shape)
        # Define the permutations used to put variables into blocks
        self.permutation = np.random.permutation(self.nr_train_vars)
        # Get the inverse of the permutation. Used to rearrange the created variables into the desired shape.
        self.permutation_inv = np.argsort(self.permutation)

    def _create_mu_sigma(self):
        """
        For all the parameters defined in the elements in self.variables, do:
            - accumulate them into one list
            - pad the list so that the number of parameters is divisible by the block size
            - apply the permutation to the list. This is done in order to ensure that elements in a block are random
            - reshape the list into the accumulated blocks shape
        """
        # Get the mu's and sigmas and concat them
        mu, sigma = list(zip(*self.variables))
        mu = tf.concat(mu, axis=0, name='mu')
        sigma = tf.concat(sigma, axis=0, name='sigma')

        # Pad mu and sigma so that we have variables for each block
        mu_padding = tf.Variable(tf.zeros(self.nr_train_vars - self.nr_actual_vars), dtype=self.dtype)
        sigma_padding = tf.Variable(tf.zeros(self.nr_train_vars - self.nr_actual_vars), dtype=self.dtype)
        padded_mu = tf.concat((mu, mu_padding), axis=0, name='padded_mu')
        padded_sigma = tf.concat((sigma, sigma_padding), axis=0, name='padded_sigma')

        # Apply the permutation
        permuted_mu = tf.gather(padded_mu, self.permutation)
        permuted_sigma = tf.gather(padded_sigma, self.permutation)

        # Reshape mu, sigma and p so that it has the accb shape
        self.mu = tf.reshape(permuted_mu, self.accb_shape)
        self.sigma = tf.reshape(permuted_sigma, self.accb_shape)

    def _create_p_scale(self):
        """
        Since in a block we have parameters from multiple layers, we need to find a way to relate back to those
        when doing the kl divergence. To do that we'll first create a mapping from elements in self.mu to which layer
        they were in. Then we'll reshape this array in accb shape and make its elements be the actual scale of the
        layers
        """
        # Create and extra layer which would contain the variables created for padding
        extra_layer = tf.Variable(self.log_p_initial_sigma, dtype=self.dtype, trainable=self.train_p_scale)
        p_scale_vars = tf.concat(self.p_scale_vars + [extra_layer], axis=0)
        p_scale = tf.exp(p_scale_vars)  # apply the exponent so that we get the actual scale
        # Define the mapping from variables to which layer they are
        # This function outputs 000111122 if 1st layer has 3 variables, second layer has 4, and out padding size is 2
        p_scale_permutation = np.repeat(range(self.nr_layers + 1),
                                        self.layer_sizes + [self.nr_train_vars - self.nr_actual_vars])
        p_scale_permutation = p_scale_permutation[self.permutation]  # apply the actual permutation to get the mapping
        permuted_p_scale = tf.gather(p_scale, p_scale_permutation)
        # Reshape to accb shape
        self.p_scale = tf.reshape(permuted_p_scale, self.accb_shape)
