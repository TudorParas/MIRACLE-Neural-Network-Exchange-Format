import numpy as np
import tensorflow as tf
from tqdm import tqdm

from linear_regression_experiment.toy_data.toy_data import ToyData

# Data parameters
SAMPLES = 60000
ROWS = 10
OUTPUTS = 2

# Parameters for defining the graph
DTYPE = tf.float32
WEIGHT_DECAY = 5e-4
LOG_INITIAL_SIGMA = -10.
LOG_P_INITIAL_SIGMA = -1.

# Training parameters
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
INITIAL_KL_PENALTY = 1e-08
KL_PENALTY_STEP = 1.005  # Should be > 1

# Logging parameters
SUMMARIES_DIR = 'out/basic_miracle'
MESSAGE_FREQUENCY = 1000  # output message this many iterations


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
        self.compressed_size = compressed_size
        self.kl_target = tf.constant(compressed_size * np.log(2), dtype=DTYPE)  # Transform from bits to nats.
        self.kl_penalty_step = KL_PENALTY_STEP  # How much we increase/decrease penalty if exceeding/not target KL

        self.shape = [ROWS, OUTPUTS]

        self._create_graph()

        self._initialize_session()

    def _create_graph(self):
        """Create a graph of the linear regression matrix which we'll compress and the bias which we won't"""
        self.x, self.y = self.dataset.get_data()
        with tf.name_scope('Linear_regression'):
            self.weight_matrix = self._create_linear_reg_matrix()
            self.bias = tf.Variable(tf.random_normal(shape=[OUTPUTS]), "bias")

            predictions = tf.matmul(self.x, self.weight_matrix) + self.bias
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

    def _create_prior(self):
        """ Create the "prior" and the shared source of randomness. The variance of the prior is trainable"""
        with tf.name_scope('prior'):
            self.p_scale_var = tf.Variable(LOG_INITIAL_SIGMA, dtype=DTYPE)
            self.p_scale = tf.exp(self.p_scale_var)
            self.p = tf.distributions.Normal(loc=0., scale=self.p_scale)

    def _create_linear_reg_matrix(self):
        """Create a gaussian for each variable. Weigh their variance by their size"""
        with tf.name_scope("weights"):
            # Mean of each variable
            mu_init = np.random.normal(size=self.shape, loc=0., scale=np.sqrt(1. / self.shape[0]))
            self.mu = tf.Variable(mu_init, dtype=DTYPE, name='mu')
            # Variance for all weight blocks initilized to 1e-10. We want the exponenet to be the trained variable.
            self.sigma_var = tf.Variable(tf.fill(self.shape, tf.cast(LOG_P_INITIAL_SIGMA, dtype=DTYPE)), name='sigma')
            self.sigma = tf.exp(self.sigma_var)

            variational_weights = tf.random_normal(self.shape, mean=self.mu, stddev=self.sigma)

            return variational_weights

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

    def train(self, iterations):
        """Train until the kl converges at a value smaller than the target kl"""
        self.sess.run(self.enable_kl_loss.assign(1.))
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
        tf.summary.scalar('Loss', self.loss)
        tf.summary.scalar('KL', self.current_kl)
        tf.summary.scalar('KL Penalty', self.kl_penalty)
        tf.summary.scalar('KL Loss', self.kl_loss)

        self.merged = tf.summary.merge_all()

    def compress(self, out_file=None):
        """Compress the linear regression matrix and store it in out_file"""
        kl, scale = self.sess.run([self.current_kl, self.p_scale])
        print("Starting compression with KL {0} bits, prior with scale {1}".format(kl / np.log(2), scale))
        with tf.name_scope("Compressor"):
            samples = int(np.power(2, self.compressed_size))
            alphas = list()
            for current_seed in tqdm(range(samples), desc="Trying samples"):
                sample_w = self.p.sample(self.shape, seed=current_seed)

                q_prob = tf.reduce_sum(self.w_dist.prob(sample_w))
                p_prob = tf.reduce_sum(self.p.prob(sample_w))
                sample_alpha = q_prob / p_prob

                alphas.append(sample_alpha)

            alphas = np.array(self.sess.run(alphas))
            probabilities = alphas / alphas.sum()

            # chosen_seed = np.random.choice(samples, p=probabilities)  # Sample over the seeds with the probbilities
            chosen_seed = np.argmax(probabilities)  # Pick the one with the highest probability
            print("Chosen seed is {0} with probability {1}".format(chosen_seed, probabilities[chosen_seed]))
            chosen_w = self.p.sample(self.shape, seed=chosen_seed)
            print("Chosen w is :\n {0}".format(self.sess.run(chosen_w)))
            self._test_chosen_w(chosen_w)

            if out_file:
                self._output_model_to_file(out_file, chosen_seed)

    def _test_chosen_w(self, chosen_w):
        """Test how good it is the w we have chosen"""
        print("Testing the chosen w")
        # Store the values so they can be reloaded
        tmp_mu, tmp_sigma_var = self.sess.run([self.mu, self.sigma_var])
        # Make the matrix deterministic by making sigma really low and mu equal to the chosen w
        self.sess.run([self.mu.assign(chosen_w),
                       self.sigma_var.assign(tf.fill(self.shape, tf.cast(LOG_P_INITIAL_SIGMA, dtype=DTYPE)))])
        self.test(kl_loss=False)
        # Restore the values
        self.sess.run([self.mu.assign(tmp_mu),
                       self.sigma_var.assign(tmp_sigma_var)])

    def _output_model_to_file(self, out_file, chosen_seed):
        """Output the compressed model to a .mrcl file"""
        pass

    def load_model(self, out_file):
        """Load the compressed model from the .mrcl file"""
        pass
