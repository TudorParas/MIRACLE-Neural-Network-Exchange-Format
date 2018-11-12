from linear_regression_experiment.toy_data.basic_miracle import BasicMiracle
import tensorflow as tf

tf.InteractiveSession()
tf.global_variables_initializer().run()

NO_KL_ITERATIONS = 400
KL_ITERATIONS = 6000
COMPRESSION_SIZE = 5

bm = BasicMiracle(COMPRESSION_SIZE)

bm.pretrain(NO_KL_ITERATIONS)
bm.train(KL_ITERATIONS)

chosen_seed, chosen_w, max_seed, max_w, log_q_probs, p_sample = bm.compress()