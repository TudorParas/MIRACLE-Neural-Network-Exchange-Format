import numpy as np
import tensorflow as tf

import miracle

np.random.seed(53)

tf.InteractiveSession()
a = miracle.create_variable(shape=(5, 40, 40, 20), hash_group_size=13)
b = miracle.create_variable(shape=(4, 4), hash_group_size=2)
loss_mock = tf.constant(0)
miracle.graph._initialize_train_compression_graph(loss=loss_mock, compressed_size_bytes=10, block_size_vars=None,
                                                  bits_per_block=None, optimizer=None, initial_kl_penalty=1e-08,
                                                  kl_penalty_step=1.0002)


def test_create_variable():
    tf.global_variables_initializer().run()
    ev = a.eval()
    assert ev.shape == (5, 40, 40, 20)
    # Check that the hashed values are actually equal
    flat_ev = np.reshape(ev, [-1])
    assert all(flat_ev[:12] == flat_ev[1:13])  # first 13 elements are equal
    assert all(flat_ev[13:25] == flat_ev[14:26])  # next 13 elements are equal
    assert not all(flat_ev[:-1] == flat_ev[1:])  # not all elements are equal
