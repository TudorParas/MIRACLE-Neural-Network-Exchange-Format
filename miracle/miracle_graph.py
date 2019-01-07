"""Define a miracle graph in which we create the layers"""
import logging

import numpy as np
import tensorflow as tf

import miracle.miracle_graph_utils as mgu


class MiracleGraph(object):
    def __init__(self, log_initial_sigma=1e-10, log_p_initial_sigma=1e-2, train_p_scale=True,
                 dtype=tf.float32, seed=53):
        """
        Define the graph we'll use for constructing the miracle layers

        Parameters
        ----------
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
        self.log_initial_sigma = log_initial_sigma
        self.log_p_initial_sigma = log_p_initial_sigma
        self.train_p_scale = train_p_scale
        self.dtype = dtype
        # Define a lsit used to accumulate all the parameters so that we can block them when doing the KL loss
        self.variables = list()
        # Accumulate the fixed weights and the uncompressed masks for each layer.
        # Each entry has the format (new_fixed_weight, new_mask)
        self.layers_fw_um = list()
        # Define a list that accumulates the p_scale for each layer
        self.p_scale_vars = list()
        # Define list that accumulates the number of actual standing vars in each layer after hashing
        self.hashed_layer_sizes = list()

    def create_variable(self, shape, hash_group_size=1, scale=None):
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
        if scale is None:
            # The bigger the layer, the smaller the standard deviation.
            scale = np.sqrt(1 / np.prod(shape[:-1]))

        nr_hashed_vars = np.prod(shape) // hash_group_size  # this * hash_group_size = nr of vars hashed
        nr_leftover_vars = np.prod(shape) % hash_group_size  # leftover bc it doesn't divide perfectly
        nr_trained_vars = nr_hashed_vars + nr_leftover_vars
        logging.info("Creating variable of shape {} with hash size {}".format(shape, hash_group_size))
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
            epsilon = tf.random_normal([nr_trained_vars], name='epsilon')  # expensive
            weights = mu + epsilon * sigma

        with tf.name_scope('fixed_weights'):
            # Define a set of fixed weights which we'll use one we compress a variable.
            # This allows for training uncompressed blocks whilst keeping compressed blocks steady
            fixed_weights = tf.Variable(tf.zeros_like(weights), trainable=False, name="fixed_weights")
            # Create the uncompressed mask which allows us to make the choice between the fixed and variational weights
            layer_uncompressed_mask = tf.Variable(tf.ones_like(weights), trainable=False,
                                                  name='uncompressed_mssk')

        with tf.name_scope("combined_weights"):
            combined_weights = layer_uncompressed_mask * weights + (
                    1 - layer_uncompressed_mask) * fixed_weights

        with tf.name_scope('expand_weights'):
            # Expand the hashed weights and put them in the corresponding shape
            expanded_weights = mgu.expand_variable(combined_weights, shape, nr_hashed_vars, hash_group_size)

        # Keep track of the actual weights, as they will be used to define the KL loss and during compression
        self.variables.append((mu, sigma))
        self.layers_fw_um.append((fixed_weights, layer_uncompressed_mask))
        # Create the prior scale for this layer
        p_scale_var = tf.Variable(self.log_p_initial_sigma, dtype=self.dtype, trainable=self.train_p_scale)
        self.p_scale_vars.append(p_scale_var)
        self.hashed_layer_sizes.append(nr_trained_vars)

        return expanded_weights

    def compress(self, loss, compressed_size_bytes=None, block_size_vars=None, bits_per_block=None, optimizer=None,
                 initial_kl_penalty=1e-08, kl_penalty_step=1.0002, pretrain_iterations=None):
        """Create the kl loss that will be used during optimization. During this we'll also initialize the compressor.
        The create_variable function should no longer be used after calling this function.

        Parameters
        ---------
        loss: tf.Tensor
            The loss defined in the graph. Must be a 0-D tensor of type float
        compressed_size_bytes: int
            Size of the compressed file in bytes.
            Mandatory if the user doesn't specify both block_size_vars and bits_per_block.
        block_size_vars: int
            Number of variables in a compression block. Advanced parameter that requires knowledge of the algorithm.
        bits_per_block: int
            Number of bits we assign to a block. Advanced parameter that requires knowledge of the algorithm.
        optimizer: tf.train.Optimizer
            The optimizer we'll use for training. Defaults to AdamOptimizer
        initial_kl_penalty: float
            How much we multiply the KL loss by initially. Bigger values means KL loss has a big impact from the start.
            A low initial value is recommended in order to slowly anneal the loss in.
        kl_penalty_step: float
            How much we multiply the KL penalty at each step if our block KL is still bigger than the target KL.
            Higher values means that we anneal in KL loss faster. Might increase training speed at the expense of
            accuracy.
        pretrain_iterations: int
            Maximum number of iterations we would like to pretrain for.
        """
        # Create the graph
        with tf.name_scope('train_compression'):
            self._initialize_train_compression_graph(loss, compressed_size_bytes, block_size_vars, bits_per_block,
                                                     optimizer, initial_kl_penalty, kl_penalty_step)

        # Execute the graph
        self._pretrain(pretrain_iterations)

    def _initialize_train_compression_graph(self, loss, compressed_size_bytes, block_size_vars,
                                            bits_per_block, optimizer, initial_kl_penalty, kl_penalty_step):
        """
        Function in place only to give structure to the library.
        Here we define the whole graph before doing the execution in a TensorFlow session.
        """
        logging.info("Initializing compression graph")
        self._create_graph_parameters(compressed_size_bytes, block_size_vars, bits_per_block)

        with tf.name_scope('mu_sigma'):
            self._create_mu_sigma()
        with tf.name_scope('p_scale'):
            self._create_p_scale()

        with tf.name_scope("KL_loss"):
            self._initialize_kl_loss(initial_kl_penalty, kl_penalty_step)

        with tf.name_scope("Training"):
            self._initialize_training_graph(loss, optimizer)

        with tf.name_scope('Compression'):
            self._initialize_compressor()

    def _create_graph_parameters(self, compressed_size_bytes, block_size_vars, bits_per_block):
        """Create variables that will help in defining the graph"""
        self.nr_layers = len(self.hashed_layer_sizes)
        self.nr_actual_vars = sum(self.hashed_layer_sizes)
        self.block_size_vars, self.bits_per_block = mgu.parse_compressed_size(compressed_size_bytes,
                                                                              self.nr_actual_vars,
                                                                              block_size_vars,
                                                                              bits_per_block)
        self.kl_target = tf.constant(self.bits_per_block * np.log(2), dtype=self.dtype)
        self.nr_blocks = int(np.ceil(self.nr_actual_vars / self.block_size_vars))
        # Shape of the accumulated blocks. One block per row. We use this because it makes it easier for us to access
        # variables in the same block
        self.accb_shape = [self.nr_blocks, self.block_size_vars]
        # The last block can be not fully filled, so we'll have variables not used in inference, but used for kl loss.
        self.nr_train_vars = int(np.prod(self.accb_shape))
        # Define the permutations used to put variables into blocks
        self.permutation = np.random.permutation(self.nr_train_vars)
        # Get the inverse of the permutation. Used to rearrange the created variables into the desired shape.
        self.permutation_inv = np.argsort(self.permutation)
        logging.info("\n\tUsing {nr_layers} layers\n"
                     "\tNumber of actual variables {nr_actual_vars}\n"
                     "\tNumber of trained variables {nr_trained_vars}\n"
                     "\tBlock size {block_size_vars}, Bits per block {bits_per_block}\n"
                     "\tKL target {kl_target}\n"
                     "\tAccumulated Blocks Shape {accb_shape}".format(
            nr_layers=self.nr_layers, nr_actual_vars=self.nr_actual_vars, nr_trained_vars=self.nr_train_vars,
            block_size_vars=self.block_size_vars, bits_per_block=self.bits_per_block, kl_target=self.kl_target,
            accb_shape=self.accb_shape))

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
        with tf.name_scope('mu'):
            mu = tf.concat(mu, axis=0, name='mu')
            # Pad mu so that we have variables for each block
            mu_padding = tf.Variable(tf.zeros(self.nr_train_vars - self.nr_actual_vars), dtype=self.dtype,
                                     name='mu_padding')
            padded_mu = tf.concat((mu, mu_padding), axis=0, name='padded_mu')
            # Apply the permutation
            permuted_mu = tf.gather(padded_mu, self.permutation, name='permuted_mu')
            # Reshape mu to accb shape
            self.mu = tf.reshape(permuted_mu, self.accb_shape)

        with tf.name_scope('sigma'):
            sigma = tf.concat(sigma, axis=0, name='sigma')
            # Pad sigma so that we have variables for each block
            sigma_padding = tf.Variable(tf.zeros(self.nr_train_vars - self.nr_actual_vars), dtype=self.dtype,
                                        name='sigma_padding')
            padded_sigma = tf.concat((sigma, sigma_padding), axis=0, name='padded_sigma')
            # Apply the permutation
            permuted_sigma = tf.gather(padded_sigma, self.permutation, name='permuted_sigma')
            # Reshape sigma to accb shape
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
        p_scale_vars = tf.stack(self.p_scale_vars + [extra_layer], axis=0)
        p_scale = tf.exp(p_scale_vars)  # apply the exponent so that we get the actual scale
        # Define the mapping from variables to which layer they are
        # This function outputs 000111122 if 1st layer has 3 variables, second layer has 4, and out padding size is 2
        p_scale_permutation = np.repeat(range(self.nr_layers + 1),
                                        self.hashed_layer_sizes + [self.nr_train_vars - self.nr_actual_vars])
        p_scale_permutation = p_scale_permutation[self.permutation]  # apply the actual permutation to get the mapping
        permuted_p_scale = tf.gather(p_scale, p_scale_permutation)  # Apply the mapping to our p_scale
        # Reshape to accb shape
        self.p_scale = tf.reshape(permuted_p_scale, self.accb_shape)

    def _initialize_kl_loss(self, initial_kl_penalty, kl_penalty_step):
        logging.info("Creating KL loss for {} layers".format(len(self.variables)))
        # Define the prior distribution and the variational distribution
        w_dist = tf.contrib.distributions.Normal(loc=self.mu, scale=self.sigma)
        p = tf.contrib.distributions.Normal(loc=0., scale=self.p_scale)
        # Compute the block-wise kl divergence. We use that KL divergence is additive along dimensions.
        # Because each parameter is a new dimensions, the sum along columns gives us the block KL
        total_kl = tf.distributions.kl_divergence(w_dist, p)
        block_kl = tf.reduce_sum(total_kl, axis=1)
        self.mean_kl = tf.reduce_mean(block_kl)  # used for logging purposes

        # Define a variable that keeps track of which layers are compressed.
        # This is redundant; can be inferred from layer_uncompressed_masks. Created because the code is simpler
        self.block_uncompressed_mask = tf.Variable(tf.ones(shape=self.nr_blocks), trainable=False)
        # Define variable that allows us to control whether we want to enable the loss or not
        self.enable_kl_loss = tf.Variable(1., dtype=self.dtype, trainable=False)

        # Define the KL penalties. We use this to increase or decrease the KL loss.
        self.kl_penalties = tf.Variable(tf.fill([self.nr_blocks], initial_kl_penalty), dtype=self.dtype)
        # Operation to update the penalty in case the kl_loss exceeds the target kl
        self.kl_penalty_update = self.kl_penalties.assign(
            tf.where(
                condition=tf.logical_and(  # Block is uncompressed and the block's KL is greater than the target
                    tf.cast(self.block_uncompressed_mask, tf.bool),
                    tf.greater(block_kl, self.kl_target)),
                x=self.kl_penalties * kl_penalty_step,  # Increase penalty
                y=self.kl_penalties / kl_penalty_step),  # Decrease penalty
            name='KL_penalty_update')

        # Finally, define the KL loss
        self.kl_loss = tf.reduce_sum(
            block_kl * self.block_uncompressed_mask * self.kl_penalties) * self.enable_kl_loss

    def _initialize_training_graph(self, loss, optimizer):
        loss = tf.cast(loss, dtype=self.dtype)
        self.total_loss = loss + self.kl_loss

        if optimizer is None:
            # Use the Adam Optimizer with default parameters:
            optimizer = tf.train.AdamOptimizer()

        self.train_op = optimizer.minimize(self.total_loss)

        # Define train operation that does not train the layer p scales.
        # This is used after during retraining after beginning compression.
        no_pscales_list = [v for v in tf.trainable_variables() if v not in self.p_scale_vars]
        self.train_op_no_pscales = optimizer.minimize(self.total_loss, var_list=no_pscales_list)

    def _initialize_compressor(self):
        """Create the graph for the compression operations"""
        self.block_to_compress_index = tf.placeholder(tf.int32)
        sample_block = self._get_normal_sample_block()
        with tf.name_scope('block_info'):
            # Get the mu and sigma for this block
            block_mu = self.mu[self.block_to_compress_index, :]
            block_sigma = self.sigma[self.block_to_compress_index, :]
            block_p = self.p_scale[self.block_to_compress_index, :]

        with tf.name_scope('p_sample'):
            """Generate the samples for this prior distribution"""
            p_sample = sample_block * block_p

        with tf.name_scope('probabilities'):
            # Compute the negative probabilities of the sample for the p and q distributions.
            self.log_q_probs = tf.reduce_sum(
                tf.log(1 / tf.sqrt(2 * np.pi * tf.pow(block_sigma, 2))) - (
                        tf.square(p_sample - block_mu) / (2 * tf.square(block_sigma))
                ),
                axis=1)
            self.log_p_probs = tf.reduce_sum(tf.log(1 / tf.sqrt(2 * np.pi * tf.square(block_p))) - (
                    tf.square(sample_block) / 2), axis=1)

        with tf.name_scope('sampling'):
            # Create a Categorical distribution which we'll use for sampling.
            # By using softmax we turn the log probabilities into actual probabilities
            alphas = tf.nn.softmax(self.log_q_probs - self.log_p_probs)
            cat_distr = tf.distributions.Categorical(probs=alphas)

            self.chosen_seed = cat_distr.sample()
            # self.chosen_seed = tf.argmax(self.log_q_probs)  # Choose the value most like to have come from q
            self.chosen_sample = p_sample[self.chosen_seed, :]

        self._create_intermediate_fw_um()

        with tf.name_scope('compression_operations'):
            self.fixed_weights_update = tf.scatter_update(ref=self.intermediate_concat_fixed_weights,
                                                          indices=self.block_to_compress_index,
                                                          updates=self.chosen_sample,
                                                          name='fixed_weights_update')
            # ToDo remove this once concat_uncompressed_mask is changed
            self.mask_update = tf.scatter_update(ref=self.intermediate_concat_uncompressed_mask,
                                                 indices=self.block_to_compress_index,
                                                 updates=tf.fill([self.accb_shape[1]], 1.),
                                                 name='mask_update')
            self.block_mask_update = tf.scatter_update(self.block_uncompressed_mask,
                                                       indices=self.block_to_compress_index,
                                                       updates=1.,
                                                       name='block_mask_update')

    def _get_normal_sample_block(self):
        """Generate the block that we will use to sample. Generating it now for all functions saves time"""
        samples = np.power(2, self.bits_per_block)  # total nr of samples we consider
        sample_block = np.random.normal(size=[samples, self.block_size_vars])
        sample_block = tf.constant(sample_block, dtype=self.dtype)

        return sample_block

    def _create_intermediate_fw_um(self):
        logging.info("Creating intermediate fixed weights and uncompressed mask")
        # Update the fixed weights and the layer masks. We will use an intermediate variable which we'll update,
        # and then we update the original (given to the user) fixed weights and uncompressed masks.

        # Create a fixed weight variable that will replesent all the fixed weights concatenated.
        # These will be as if the permutation has already been done because it leads to easier updating.
        self.intermediate_concat_fixed_weights = tf.Variable(tf.fill(self.accb_shape, 0.), dtype=self.dtype,
                                                             name='concatenated_fixed_weights')
        # ToDo create this from block_uncompressed_mask
        self.intermediate_concat_uncompressed_mask = tf.Variable(tf.fill(self.accb_shape, 0.), dtype=self.dtype,
                                                                 name='concatenated_uncompressed_wasks')

        with tf.name_scope('FW_UM_copying'):
            logging.info("Restoring the intermediate weights to fit the shape of the original ones")
            # Flatten and apply the inverse permutation to the weights and the masks. This is to allow us to map the
            # values of these concatenated FW and UM back to the original ones
            restored_concat_fixed_weights = tf.gather(tf.reshape(self.intermediate_concat_fixed_weights, [-1]),
                                                      self.permutation_inv)
            restored_concat_uncompressed_masks = tf.gather(tf.reshape(self.intermediate_concat_uncompressed_mask, [-1]),
                                                           self.permutation_inv)

            # Create the graph that maps these variables back the the original ones. This is done by splitting them
            # And then using the assign operations given by the original ones.
            split_fw = tf.split(restored_concat_fixed_weights,
                                self.hashed_layer_sizes + [
                                    self.nr_train_vars - self.nr_actual_vars])  # all layers + the padded layer
            split_um = tf.split(restored_concat_uncompressed_masks,
                                self.hashed_layer_sizes + [
                                    self.nr_train_vars - self.nr_actual_vars])  # all layers + the padded layer
            logging.info("Create copying graph to copy intermediate weights into the original ones")
            self.copy_ops = list()
            for layer_index in range(self.nr_layers):
                original_fixed_weights, original_uncompressed_mask = self.layers_fw_um[layer_index]
                copied_fw = split_fw[layer_index]
                copied_um = split_um[layer_index]

                fw_copy_op = original_fixed_weights.assign(copied_fw)
                um_copy_op = original_uncompressed_mask.assign(copied_um)
                copy_op = tf.group(fw_copy_op, um_copy_op)

                self.copy_ops.append(copy_op)

    def _pretrain(self, pretrain_iterations):
        pass

    def load(self, file_path):
        pass
