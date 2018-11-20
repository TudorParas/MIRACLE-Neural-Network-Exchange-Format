from linear_regression_experiment.mnist_data.miracle import MnistMiracle

NO_KL_ITERATIONS = 4000
KL_ITERATIONS = 80000
RETRAIN_ITERATIONS = 4000

BITS_PER_BLOCK = 10
BLOCK_SIZE = 16
HASH_GROUP_SIZE = 1


mm = MnistMiracle(bits_per_block=BITS_PER_BLOCK, block_size_vars=BLOCK_SIZE, hash_group_size_vars=HASH_GROUP_SIZE)

mm.pretrain(NO_KL_ITERATIONS)
mm.train(KL_ITERATIONS)

mm.compress(RETRAIN_ITERATIONS)