import miracle
import tensorflow as tf
import numpy as np


def test_create_variable():
    tf.InteractiveSession()
    a = miracle.create_variable(shape=(5, 40, 40, 20), hash_group_size=13)

    tf.global_variables_initializer().run()

    ev = a.eval()
    assert ev.shape == (5, 40, 40, 20)
    # Check that the hashed values are actually equal
    flat_ev = np.reshape(ev, [-1])
    assert all(flat_ev[:12] == flat_ev[1:13])  # first 13 elements are equal
    assert all(flat_ev[13:25] == flat_ev[14:26])  # next 13 elements are equal
    assert not all(flat_ev[:-1] == flat_ev[1:])  # not all elements are equal
