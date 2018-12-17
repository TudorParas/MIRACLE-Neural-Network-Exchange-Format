import os

import numpy as np
import tensorflow as tf

from examples.linear_regression_experiment.mnist_data.mnist_data import MnistData
from utils.file_io import dump_to_file, load_from_file

# Parameters for defining the graph
DTYPE = tf.float32
LOG_INITIAL_SIGMA = -10.
LOG_P_INITIAL_SIGMA = 0
TRAIN_P_SIGMA = True
DIMENSION = [784, 10]  # Dimension of the linear regression matrix before any block or hashing tricks.
NUMPY_SEED = 53  # Used for deterministic generating of permutations and of samples. Needed for loading

# Training parameters
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
INITIAL_KL_PENALTY = 1e-08
KL_PENALTY_STEP = 1.0002  # Should be > 1

# Logging parameters
SUMMARIES_DIR = 'out/miracle'
MESSAGE_FREQUENCY = 1000  # output message this many iterations

np.random.seed(NUMPY_SEED)

# Debugging parameters
WEIGHTS_TO_SHOW = 4


class MnistMiracle(object):
    def __init__(self, bits_per_block, block_size_vars, hash_group_size_vars, out_dir=None):
        """
        Define the graph for doing linear regression using the basic algorithm

        Parameters
        ----------
        bits_per_block: int
            Bits we allocate for each block
        block_size_vars: int
            How many variables there'll be in a block
        hash_group_size_vars: int
            How many variables we set to have the same value. Atm it is a divisor of nr of vars in uncompressed model
        out_dir: str
            Where do we store the compressed models
        """
        self.dataset = MnistData()

        self.bits_per_block = bits_per_block
        self.block_size_vars = block_size_vars
        self.hash_group_size_vars = hash_group_size_vars
        self.out_dir = out_dir
        try:
            os.makedirs(out_dir)
        except FileExistsError:
            pass

        self.kl_block_target = tf.constant(bits_per_block * np.log(2), dtype=DTYPE)  # Transform from bits to nats.
        self.kl_penalty_step = KL_PENALTY_STEP  # How much we increase/decrease penalty if exceeding/not target KL

        self._create_graph()
        self._initialize_session()

    def _create_graph(self):
        """Create a graph of the linear regression matrix which we'll compress and the bias which we won't"""
        x, y = self.dataset.get_data()
        with tf.name_scope('Linear_regression'):
            self.weight = self._create_w()
            self.bias = tf.Variable(initial_value=0.)  # Do not compress the bias in order to keep things simple
            logits = tf.matmul(x, self.weight) + self.bias

        with tf.name_scope('Loss'):
            self._create_prior()
            self._create_kl_loss()
            self.loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y)) + self.kl_loss

        with tf.name_scope('accuracy'):
            with tf.name_scope('correct_prediction'):
                correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
            with tf.name_scope('accuracy'):
                self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        with tf.name_scope('Training'):
            optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
            self.train_op = optimizer.minimize(self.loss)

            # Operation that does not train the standard deviation for the prior
            no_scales_list = [v for v in tf.trainable_variables() if v is not self.p_scale_var]
            self.train_op_no_scales = optimizer.minimize(self.loss, var_list=no_scales_list)

        with tf.name_scope('P_Sample'):
            self._generate_p_sample()
        with tf.name_scope('Compression'):
            self._initialize_compressor()
        with tf.name_scope('Loader'):
            self._initialize_loader()

    def _create_w(self):
        """Create a gaussian for each variable. Weigh their variance by their size"""
        self._create_w_metainformation()
        with tf.name_scope("weights"):
            with tf.name_scope("mu"):
                # Create the mu initializer by sampled from a random distribution with sigma based on size of the layer.
                # It should have one value for every trainable variable. It is flattened because we will use the
                # permutations to get the blocks
                # ToDo: If KL fuck up to bad then generate num_actual_vars randomly and pad the rest with 0
                mu_init = np.random.normal(size=self.num_train_vars, scale=1 / DIMENSION[0])
                permutted_mu_init = mu_init[self.permutation]  # Apply the permutation.
                permutted_mu_init = permutted_mu_init.reshape(self.shape)  # Reshape it into the actual shape we want

                # Mean of the learned distribution
                self.mu = tf.Variable(permutted_mu_init, dtype=DTYPE, name='mu')

            with tf.name_scope("Sigma"):
                sigma_init = tf.fill(self.shape, tf.cast(LOG_INITIAL_SIGMA, dtype=DTYPE))
                # Variance of the learned distribution initilized to 1e-10.
                # We use log sigma as a Variable because we want sigma to always be positive
                self.sigma_var = tf.Variable(sigma_init, name='sigma')
                self.sigma = tf.exp(self.sigma_var)

            with tf.name_scope("Variational_Weights"):
                # We have to apply the reparametrisation trick ourselves bc tf doesn't support multi-mean normals
                epsilon = tf.random_normal(self.shape, name="epsilon")  # N(0, 1) used for sampling
                self.variational_weights = self.mu + epsilon * self.sigma

            with tf.name_scope("Fixed_weights"):
                # Create the fixed weights, which we'll update during compression.
                # This will be used because we retrain after the compression of
                # each block, so fixed weights are from this
                self.fixed_weights = tf.Variable(tf.zeros_like(self.variational_weights), trainable=False,
                                                 name="fixed_weights")
                # Value of 1 if the block is uncompressed
                self.uncompressed_mask = tf.Variable(tf.ones(shape=self.num_blocks), trainable=False)

            with tf.name_scope("Combined_Weights"):
                # Combine the variational weights with the fixed weights using the mask
                expanded_mask = tf.expand_dims(self.uncompressed_mask, 1)  # expand dims so that maths works out
                # Combined the fixed weights with the variational weights
                combined_weights = tf.reshape(
                    expanded_mask * self.variational_weights + (1. - expanded_mask) * self.fixed_weights,
                    shape=[-1], name='combied_weights'
                )
                # Apply the inverse permutation
                permuted_weights = tf.gather(combined_weights, self.permutation_inv, name='permuted_weights')
                # Get only the actual weights, no padding
                actual_weights = permuted_weights[:self.num_actual_vars]
                # Undo the hashing trick, expanding the weights into the dimensions
                # ToDo experiment with multiplying with random.choice([-1, 1]) instead of just ones.
                flattened_weights = tf.multiply(tf.expand_dims(actual_weights, axis=1),
                                                np.ones(shape=self.hash_group_size_vars),
                                                name='flattened_weights')
                self.restored_weights = tf.reshape(flattened_weights, DIMENSION, name='weights')

        return self.restored_weights

    def _create_w_metainformation(self):
        """Create variables that will help in defining w"""
        # Get how many varibles we're actually working with. Currently only supporting a hash group which is a
        # divisor of the nr of dimensions
        if np.prod(DIMENSION) % self.hash_group_size_vars != 0:
            raise ValueError("The hash group size should be a divisor of the nr of variables.")
        self.num_actual_vars = np.prod(DIMENSION) // self.hash_group_size_vars
        # Get the number of blocks
        self.num_blocks = int(np.ceil(self.num_actual_vars / self.block_size_vars))
        # Make the shape of the trainable variables we'll generate
        self.shape = [self.num_blocks, self.block_size_vars]
        # Some vars might not make a whole block. In this case we just pad out the last block with nothing
        self.num_train_vars = np.prod(self.shape)

        # Define the permutations used to put variables into blocks
        self.permutation = np.random.permutation(self.num_train_vars)
        # Get the inverse of the permutation. Used to rearrange the created variables into the desired shape.
        self.permutation_inv = np.argsort(self.permutation)

    def _create_prior(self):
        """ Create the "prior" and the shared source of randomness. The log stddev of the prior is trainable"""
        with tf.name_scope('prior'):
            self.p_scale_var = tf.Variable(LOG_P_INITIAL_SIGMA, dtype=DTYPE, trainable=TRAIN_P_SIGMA)
            self.p_scale = tf.exp(self.p_scale_var)
            self.p = tf.distributions.Normal(loc=0., scale=self.p_scale)

    def _create_kl_loss(self):
        """Create the KL loss which states how 'far away' our distribution is from the prior"""
        with tf.name_scope("KL_LOSS"):
            # Create the distribution which we use to compute the KL loss
            self.w_dist = tf.distributions.Normal(loc=self.mu, scale=self.sigma)
            # One kl penalty for each block
            self.kl_penalties = tf.Variable(tf.fill([self.num_blocks], INITIAL_KL_PENALTY), trainable=False)
            # Compute the current kl for each block. We use reduce sum as kl is additive along dimensions
            self.block_kls = tf.reduce_sum(tf.distributions.kl_divergence(self.w_dist, self.p), axis=1)
            self.mean_kl = tf.reduce_mean(self.block_kls)  # Used for logging purposes

            # Set it as variable so that we can easily enable / disable kl loss
            self.enable_kl_loss = tf.Variable(1., dtype=DTYPE, trainable=False)  # 0 or 1
            # Define the KL loss as the actual loss multiplied by the penalty and the enable. Only consider the kl loss
            # from the uncompressed variables.
            self.kl_loss = tf.reduce_sum(self.block_kls * self.uncompressed_mask * self.kl_penalties) * \
                           self.enable_kl_loss
            # Operation to update the penalty in case the kl_loss exceeds the target kl
            self.kl_penalty_update = self.kl_penalties.assign(
                tf.where(
                    condition=tf.logical_and(  # Block is uncompressed and the block's KL is greater than the target
                        tf.cast(self.uncompressed_mask, tf.bool),
                        tf.greater(self.block_kls, self.kl_block_target)),
                    x=self.kl_penalties * self.kl_penalty_step,  # Increase penalty
                    y=self.kl_penalties / self.kl_penalty_step),  # Decrease penalty
                name='KL_penalty_updata')

    def pretrain(self, iterations):
        """Pretrain, without enforcing kl loss"""
        print("Strating pretraining for {0} iterations ".format(iterations))
        self.sess.run(self.enable_kl_loss.assign(0.))
        for i in range(iterations):
            self.sess.run(self.train_op)
            accuracy, _ = self.sess.run([self.accuracy, self.train_op])
            if i % MESSAGE_FREQUENCY == 0:
                print("Pretrain accuracy at iteration {}: {}".format(i, accuracy))
                self.test(kl_loss=False)

        mean_kl, scale, accuracy = self.sess.run([self.mean_kl, self.p_scale, self.accuracy])
        print("Finished pretraining with:\n"
              "\t Mean KL {0} bits\n"
              "\t Prior with scale {1}\n"
              "\t Accuracy: {2}\n".format(
            mean_kl / np.log(2), scale, accuracy))

    def _show_weights(self):
        print("Mu: \n{}".format(self.sess.run(self.mu)[:WEIGHTS_TO_SHOW, :WEIGHTS_TO_SHOW]))
        print("Sigma: \n{}".format(self.sess.run(self.sigma)[:WEIGHTS_TO_SHOW, :WEIGHTS_TO_SHOW]))
        print("Variational: \n{}".format(self.sess.run(self.variational_weights)[:WEIGHTS_TO_SHOW, :WEIGHTS_TO_SHOW]))
        print("Restored: \n{}".format(self.sess.run(self.restored_weights)[:WEIGHTS_TO_SHOW, :WEIGHTS_TO_SHOW]))

    def train(self, iterations, verbose=True):
        """Train until the kl converges at a value smaller than the target kl"""
        self.sess.run(self.enable_kl_loss.assign(1.))
        # with tf.control_dependencies([self.train_op]):
        #     kl_penalty_update = tf.identity(self.kl_penalty_update)
        for i in range(iterations):
            if verbose:
                self._log_training(i)
            self.sess.run([self.train_op, self.kl_penalty_update])

    def _log_training(self, iteration):
        """Function called while training for logging"""
        # Add summaries for tensorboard
        summaries = self.sess.run(self.merged)
        self.train_writer.add_summary(summaries, iteration)

        if iteration % MESSAGE_FREQUENCY == 0:
            print("Iteration {0} ".format(iteration), end='\n\n')

            loss, mean_kl, accuracy = self.sess.run([self.loss, self.mean_kl, self.accuracy])
            print("Training Data: \n"
                  "Loss--{0}, Mean KL--{1}, Accuracy--{2}".format(loss, mean_kl, accuracy),
                  end='\n\n')

            self.test()

    def test(self, kl_loss=True):
        self.dataset.initialize_test_data(self.sess)
        tmp_enable = self.sess.run(self.enable_kl_loss)
        if kl_loss:
            self.sess.run(self.enable_kl_loss.assign(1.))
            loss, mean_kl, accuracy = self.sess.run([self.loss, self.mean_kl, self.accuracy])

            print("Test Data: \n"
                  "Loss--{0}, Mean KL--{1}, Accuracy--{2}".format(loss, mean_kl, accuracy),
                  end='\n\n')
        else:
            self.sess.run(self.enable_kl_loss.assign(0.))
            loss, accuracy = self.sess.run([self.loss, self.accuracy])  #

            print("Test Data: \n"
                  "Loss--{0}, Accuracy--{1}".format(loss, accuracy),
                  end='\n\n')

        # Restore the enable_kl_loss
        self.sess.run(self.enable_kl_loss.assign(tmp_enable))
        # Revert back to train data
        self.dataset.initialize_train_data(self.sess, BATCH_SIZE)

    def _initialize_compressor(self):
        """Define the tensorflow nodes used during compression"""
        self.block_to_compress_index = tf.placeholder(tf.int32)
        with tf.name_scope('block_info'):
            # Get the mu and sigma for this block
            block_mu = self.mu[self.block_to_compress_index, :]
            block_sigma = self.sigma[self.block_to_compress_index, :]

        with tf.name_scope('probabilities'):
            # Compute the probabilities of the sample for the p and q distributions
            self.log_q_probs = tf.reduce_sum(
                tf.log(1 / tf.sqrt(2 * np.pi * tf.pow(block_sigma, 2))) - (
                        tf.square(self.p_sample - block_mu) / (2 * tf.square(block_sigma))
                ),
                axis=1)
            self.log_p_probs = tf.reduce_sum(tf.log(1 / tf.sqrt(2 * np.pi * tf.square(self.p_scale))) - (
                    tf.square(self.sample_block) / 2
            ),
                                             axis=1)

        with tf.name_scope('sampling'):
            # Create a Categorical distribution which we'll use for sampling.
            # By using softmax we turn the log probabilities into actual probabilities
            alphas = tf.nn.softmax(self.log_q_probs - self.log_p_probs)
            cat_distr = tf.distributions.Categorical(probs=alphas)

            self.chosen_seed = cat_distr.sample()
            # self.chosen_seed = tf.argmax(self.log_q_probs)  # Choose the value most like to have come from q
            self.chosen_sample = self.p_sample[self.chosen_seed, :]

        with tf.name_scope('compression_operations'):
            # Update the fixed weights and the mask
            self.fixed_weights_update = tf.scatter_update(ref=self.fixed_weights,
                                                          indices=[self.block_to_compress_index],
                                                          updates=[self.chosen_sample],
                                                          name='fixed_weights_update')
            self.mask_update = tf.scatter_update(ref=self.uncompressed_mask, indices=[self.block_to_compress_index],
                                                 updates=[0.], name='mask_update')

    def _generate_p_sample(self):
        """Generate the block sample from p that we'll use for all the compression"""
        sample_block = self._get_normal_sample_block()  # use this block to sample
        self.sample_block = tf.constant(sample_block, dtype=DTYPE)
        # Sample p using the sample vector. This means just multiplying it by p-s stf
        self.p_sample = sample_block * self.p_scale

    def _get_normal_sample_block(self):
        samples = np.power(2, self.bits_per_block)  # total nr of samples we consider
        sample_block = np.random.normal(size=[samples, self.block_size_vars])

        return sample_block

    def _initialize_loader(self):
        """Create the graph that will load model from file"""
        self.loaded_p_scale_var = tf.placeholder(DTYPE)
        self.loaded_block_index = tf.placeholder(tf.int32)
        self.loaded_block_seed = tf.placeholder(tf.int32)
        # Load the required p_scale to get the proper block
        self.load_p_scale = tf.assign(self.p_scale_var, self.loaded_p_scale_var)
        # Get the needed block
        loaded_block = self.p_sample[self.loaded_block_seed, :]
        # Update the fixed weights for this block
        self.load_block = tf.scatter_update(ref=self.fixed_weights,
                                            indices=[self.loaded_block_index],
                                            updates=[loaded_block],
                                            name='load_block')
        # Make the mask indicate that this is now a compressed block.
        self.load_mask_update = tf.scatter_update(ref=self.uncompressed_mask, indices=[self.loaded_block_index],
                                                  updates=[0.], name='load_mask_update')

    def _initialize_session(self):
        """Define the session"""
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.dataset.initialize_train_data(self.sess, BATCH_SIZE)
        self._create_summaries()
        # Write output so that it can be viewed by tensorflow
        self.train_writer = tf.summary.FileWriter(SUMMARIES_DIR + '/train',
                                                  self.sess.graph)

    def _create_summaries(self):
        summaries = {'Loss': self.loss, 'Mean_KL': self.mean_kl, 'KL Loss': self.kl_loss, "Accuracy": self.accuracy}
        for name, summary in summaries.items():
            tf.summary.scalar(name, summary)

        self.merged = tf.summary.merge_all()

    def compress(self, retrain_iter, out_name):
        """Compress the linear regression matrix and store it in out_file

        Parameters
        ----------
        retrain_iter: int
            For how many iterations we retrain after each block is compressed
        out_name: str
            Name of the file containing the compressed model
        """
        mean_kl, scale = self.sess.run([self.mean_kl, self.p_scale])
        print("Starting compression with:\n"
              "\t Mean KL {0} bits\n"
              "\t Prior with scale {1}\n".format(
            mean_kl / np.log(2), scale))

        self.sess.run(self.enable_kl_loss.assign(1.))
        chosen_seeds = list()
        for block_index in range(self.num_blocks):
            block_seed, _, _ = self.sess.run([self.chosen_seed, self.fixed_weights_update, self.mask_update],
                                             feed_dict={self.block_to_compress_index: block_index})
            chosen_seeds.append(block_seed)

            print('Block {0} of {1} compressed'.format(block_index, self.num_blocks))
            print('Retraining for {} iterations'.format(retrain_iter))
            self.sess.run([self.train_op_no_scales, self.kl_penalty_update])
            print("P_scale_var: {}".format(self.sess.run(self.p_scale_var)))
            self.test(kl_loss=True)

        p_scale_var = self.sess.run(self.p_scale_var)
        out_file = os.path.join(self.out_dir, out_name)
        dump_to_file(p_scale_vars=[p_scale_var], seeds=chosen_seeds, bits_per_block=self.bits_per_block,
                     out_file=out_file)

    def load_model(self, model_file):
        """Load the model from the file. This involves:
            Setting the p scale
            Setting the fixed_weights from the seeds
            Making the uncompressed mask 0

        """
        print('Loading model from {}'.format(model_file))
        p_scale_var, seeds = self._get_from_file(model_file)

        self.sess.run(self.load_p_scale, feed_dict={self.loaded_p_scale_var: p_scale_var})
        for block_index, block_seed in enumerate(seeds):
            self.sess.run([self.load_block, self.load_mask_update],
                          feed_dict={self.loaded_block_index: block_index,
                                     self.loaded_block_seed: block_seed})

    def _get_from_file(self, model_file):
        """Get the seeds and the pscale from file"""
        p_scale_vars, seeds = load_from_file(model_file, bits_per_block=self.bits_per_block, p_scale_number=1,
                                             seeds_number=self.num_blocks)
        return p_scale_vars[0], seeds
