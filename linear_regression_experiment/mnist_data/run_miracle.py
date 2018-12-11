from linear_regression_experiment.mnist_data.miracle import MnistMiracle
import os
NO_KL_ITERATIONS = 2000
KL_ITERATIONS = 80000
RETRAIN_ITERATIONS = 3000

BITS_PER_BLOCK = 10
BLOCK_SIZE = 30
HASH_GROUP_SIZE = 1
OUT_DIR = 'out/compressed'
OUT_FILE = 'trainp_bits{}_block{}_hash{}.mrcl'.format(BITS_PER_BLOCK, BLOCK_SIZE, HASH_GROUP_SIZE)

mm = MnistMiracle(bits_per_block=BITS_PER_BLOCK, block_size_vars=BLOCK_SIZE, hash_group_size_vars=HASH_GROUP_SIZE,
                  out_dir=OUT_DIR)

# mm.pretrain(NO_KL_ITERATIONS)
# mm.train(KL_ITERATIONS)
#
# mm.compress(RETRAIN_ITERATIONS, OUT_FILE)

mm.load_model(os.path.join(OUT_DIR, OUT_FILE))
mm.test(kl_loss=False)