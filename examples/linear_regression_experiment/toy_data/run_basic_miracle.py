from examples.linear_regression_experiment.toy_data.basic_miracle import BasicMiracle

NO_KL_ITERATIONS = 1000
KL_ITERATIONS = 4000
COMPRESSION_SIZE = 4

bm = BasicMiracle(COMPRESSION_SIZE)

bm.pretrain(NO_KL_ITERATIONS)
bm.train(KL_ITERATIONS)

bm.compress()
