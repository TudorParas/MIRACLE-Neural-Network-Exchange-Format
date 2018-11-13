import numpy as np
import tensorflow as tf

from linear_regression_experiment.toy_data.toy_data import ToyData

# Data parameters
SAMPLES = 10000
SLOPE = 2

# Parameters for defining the graph
DTYPE = tf.float32
LOG_INITIAL_SIGMA = -10.
LOG_P_INITIAL_SIGMA = 0
TRAIN_P_SIGMA = True

NUMPY_SEED = 2  # seed if we use a normally sampled vector

# Training parameters
BATCH_SIZE = 128
LEARNING_RATE = 1e-2
INITIAL_KL_PENALTY = 1e-08
KL_PENALTY_STEP = 1.005  # Should be > 1

# Logging parameters
SUMMARIES_DIR = 'out/basic_miracle'
MESSAGE_FREQUENCY = 2000  # output message this many iterations

np.random.seed(NUMPY_SEED)


class BasicMiracle(object):
    def __init__(self, compressed_size):
        """
        Define the graph for doing linear regression using the basic algorithm

        Parameters
        ----------
        compressed_size: int
            Size of the final compressed model in bits
        """
        self.dataset = ToyData(samples=SAMPLES, slope=SLOPE)
        self.compressed_size = compressed_size
        self.kl_target = tf.constant(compressed_size * np.log(2), dtype=DTYPE)  # Transform from bits to nats.
        self.kl_penalty_step = KL_PENALTY_STEP  # How much we increase/decrease penalty if exceeding/not target KL

        self._create_graph()
        self._initialize_session()

    def _create_graph(self):
        """Create a graph of the model"""
        self.x, self.y = self.dataset.get_data()
        with tf.name_scope('Linear_regression'):
            self.weight = self._create_w()

            predictions = self.x * self.weight

        with tf.name_scope('Loss'):
            self._create_prior()
            self._create_kl_loss()
            self.loss = tf.reduce_mean(tf.square(predictions - self.y)) + self.kl_loss

        with tf.name_scope('Training'):
            optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
            self.train_op = optimizer.minimize(self.loss)

            # Operation that does not train the standard deviation for the prior
            no_scales_list = [v for v in tf.trainable_variables() if v is not self.p_scale_var]
            self.train_op_no_scales = optimizer.minimize(self.loss, var_list=no_scales_list)

        with tf.name_scope('Compression'):
            self._initialize_compressor()

    def _create_prior(self):
        """ Create the "prior" and the shared source of randomness. The log stddev of the prior is trainable"""
        with tf.name_scope('prior'):
            self.p_scale_var = tf.Variable(LOG_P_INITIAL_SIGMA, dtype=DTYPE, trainable=TRAIN_P_SIGMA)
            self.p_scale = tf.exp(self.p_scale_var)
            self.p = tf.distributions.Normal(loc=0., scale=self.p_scale)

    def _create_w(self):
        """Create a gaussian that is our weight"""
        with tf.name_scope("weights"):
            # Mean of the learned distribution
            self.mu = tf.Variable(np.random.normal(), dtype=DTYPE, name='mu')
            # Variance of the learned distribution initilized to 1e-10.
            # We use log sigma as a Variable because we want sigma to always be positive
            self.sigma_var = tf.Variable(tf.cast(LOG_INITIAL_SIGMA, dtype=DTYPE), name='sigma')
            self.sigma = tf.exp(self.sigma_var)

            variational_weight = tf.random_normal(shape=[1], mean=self.mu, stddev=self.sigma)

            return variational_weight

    def _create_kl_loss(self):
        """Create the KL loss which states how 'far away' our distribution is from the prior"""
        with tf.name_scope("KL_LOSS"):
            # Create the distribution which we use to compute the KL loss
            self.w_dist = tf.distributions.Normal(loc=self.mu, scale=self.sigma)
            # Only one KL penalty as we don't use blocks
            self.kl_penalty = tf.Variable(INITIAL_KL_PENALTY, trainable=False)
            # Compute the current kl. As we compare 2 multivariate gaussians the kl_divergence will return a 'shape'
            # shaped kl. We sum that bc kl is additive along different dimensions.
            self.current_kl = tf.reduce_sum(tf.distributions.kl_divergence(self.w_dist, self.p))

            # Set it as variable so that we can easily enable / disable kl loss
            self.enable_kl_loss = tf.Variable(1., dtype=DTYPE, trainable=False)  # 0 or 1
            # Define the KL loss as the actual loss multiplied by the penalty and the enable.
            self.kl_loss = self.current_kl * self.kl_penalty * self.enable_kl_loss
            # Operation to update the penalty in case the kl_loss exceeds the target kl
            self.kl_penalty_update = self.kl_penalty.assign(
                tf.where(condition=tf.greater(self.current_kl, self.kl_target),
                         x=self.kl_penalty * self.kl_penalty_step,  # Increase penalty
                         y=self.kl_penalty / self.kl_penalty_step),  # Decrease penalty
                name='KL_penalty_updata')

    def pretrain(self, iterations):
        """Pretrain, without enforcing kl loss"""
        print("Strating pretraining for {0} iterations ".format(iterations))
        self.sess.run(self.enable_kl_loss.assign(0.))
        for i in range(iterations):
            self.sess.run(self.train_op)

        kl, scale, mu, sigma = self.sess.run([self.current_kl, self.p_scale, self.mu, self.sigma])
        print("Finished pretraining with:\n"
              "\t KL {0} bits\n"
              "\t Prior with scale {1}\n"
              "\t W_mu: {2}\n"
              "\t W_sigma: {3}\n".format(
            kl / np.log(2), scale, mu, sigma))

    def train(self, iterations):
        """Train until the kl converges at a value smaller than the target kl"""
        self.sess.run(self.enable_kl_loss.assign(1.))
        # with tf.control_dependencies([self.train_op]):
        #     kl_penalty_update = tf.identity(self.kl_penalty_update)
        for i in range(iterations):
            self._log_training(i)
            self.sess.run([self.train_op, self.kl_penalty_update])

    def _log_training(self, iteration):
        """Function called while training for logging"""
        # Add summaries for tensorboard
        summaries = self.sess.run(self.merged)
        self.train_writer.add_summary(summaries, iteration)

        if iteration % MESSAGE_FREQUENCY == 0:
            loss, kl, kl_penalty, kl_loss = self.sess.run([self.loss, self.current_kl, self.kl_penalty, self.kl_loss])

            print("Iteration {0} ".format(iteration), end='\n\n')
            print("Training Data: \n"
                  "Loss--{0}, KL--{1}, KL penalty--{2}, KL loss--{3}".format(loss, kl, kl_penalty, kl_loss))

            self.test()

    def test(self, kl_loss=True):
        self.dataset.initialize_test_data(self.sess)
        if kl_loss:
            loss, kl, kl_penalty, kl_loss = self.sess.run([self.loss, self.current_kl, self.kl_penalty, self.kl_loss])

            print("Test Data: \n"
                  "Loss--{0}, KL--{1}, KL penalty--{2}, KL loss--{3}".format(loss, kl, kl_penalty, kl_loss),
                  end='\n\n')
        else:
            tmp_enable, _ = self.sess.run([self.enable_kl_loss, self.enable_kl_loss.assign(0.)])
            loss = self.sess.run(self.loss)  #

            print("Test Data: \n"
                  "Loss--{0}".format(loss),
                  end='\n\n')
            self.sess.run(self.enable_kl_loss.assign(tmp_enable))

        # Revert back to train data
        self.dataset.initialize_train_data(self.sess, BATCH_SIZE)

    def _initialize_compressor(self):
        """Define the tensorflow nodes used during compression"""
        # Generate a sequence of numbers sampled from a unit gaussian that we'll use to sample our prior
        nr_of_samples = np.power(2, self.compressed_size)
        sample_vector = self._get_normal_sample_vector(nr_of_samples)

        sample_vector = tf.constant(sample_vector, dtype=DTYPE)
        # Sample p using the sample vector. This means just multiplying it by p-s stf
        self.p_sample = sample_vector * self.p_scale
        with tf.name_scope('probabilities'):
            # Compute the probabilities of the sample for the p and q distributions
            self.log_q_probs = tf.log(1 / tf.sqrt(2 * np.pi * tf.pow(self.sigma, 2))) - (
                    tf.square(self.p_sample - self.mu) / (2 * tf.square(self.sigma))
            )
            self.log_p_probs = tf.log(1 / tf.sqrt(2 * np.pi * tf.square(self.p_scale))) - (
                    tf.square(sample_vector) / 2
            )

        with tf.name_scope('sampling'):
            # Create a Categorical distribution which we'll use for sampling.
            # By using softmax we turn the log probabilities into actual probabilities
            alphas = tf.nn.softmax(self.log_q_probs - self.log_p_probs)
            cat_distr = tf.distributions.Categorical(probs=alphas)

            self.chosen_seed = cat_distr.sample()
            self.max_seed = tf.argmax(self.log_q_probs)  # Choose the value most like to have come from q

            self.chosen_w = self.p_sample[self.chosen_seed]
            self.max_w = self.p_sample[self.max_seed]

    def _get_normal_sample_vector(self, samples):
        sample_vector = np.random.normal(size=samples)

        return sample_vector

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
        summaries = {'Loss': self.loss, 'KL': self.current_kl, 'KL Penalty': self.kl_penalty, 'KL Loss': self.kl_loss}
        for name, summary in summaries.items():
            tf.summary.scalar(name, summary)

        self.merged = tf.summary.merge_all()

    def compress(self, out_file=None):
        """Compress the linear regression matrix and store it in out_file"""
        kl, scale, mu, sigma = self.sess.run([self.current_kl, self.p_scale, self.mu, self.sigma])
        print("Starting compression with:\n"
              "\t KL {0} bits\n"
              "\t Prior with scale {1}\n"
              "\t W_mu: {2}\n"
              "\t W_sigma: {3}\n".format(
            kl / np.log(2), scale, mu, sigma))

        chosen_seed, chosen_w, max_seed, max_w, log_q_probs = self.sess.run(
            [self.chosen_seed, self.chosen_w, self.max_seed, self.max_w, self.log_q_probs])

        print("Compression finished with: \n"
              "\t Chosen seed: {0}\n"
              "\t Chosen w: {1}\n"
              "\t Chosen w log probability: {4}\n"
              "\t Max seed: {2}\n"
              "\t Max w: {3}\n"
              "\t Max w log probability: {5}\n".format(
            chosen_seed, chosen_w, max_seed, max_w, log_q_probs[chosen_seed], log_q_probs[max_seed]))

        print("Testing the chosen w")
        self._test_w(chosen_w)
        print("Testing the max w")
        self._test_w(max_w)

    def _test_w(self, chosen_w):
        """Test how good it is the w we have chosen"""
        # Store the values so they can be reloaded
        tmp_mu, tmp_sigma_var = self.sess.run([self.mu, self.sigma_var])
        # Make the matrix deterministic by making sigma really low and mu equal to the chosen w
        self.sess.run([self.mu.assign(chosen_w),
                       self.sigma_var.assign(tf.cast(LOG_INITIAL_SIGMA, dtype=DTYPE))])
        self.test(kl_loss=False)
        # Restore the values
        self.sess.run([self.mu.assign(tmp_mu),
                       self.sigma_var.assign(tmp_sigma_var)])
