import miracle.miracle_graph_utils as mgu
import tensorflow as tf

import numpy as np

def test_parse_compressed_size():
    nr_actual_vars = 23
    block_size_vars = 5
    bits_per_block = 10
    size = 6
    call_args = [(None, nr_actual_vars, block_size_vars, bits_per_block), (size, nr_actual_vars, None, bits_per_block),
                 (size, nr_actual_vars, block_size_vars, None), (size, nr_actual_vars, None, None)]
    expected_outputs = [(block_size_vars, bits_per_block), (5, bits_per_block), (block_size_vars, 10), (5, 10)]

    for arguments, expected_out in zip(call_args, expected_outputs):
        actual_out = mgu.parse_compressed_size(*arguments)
        assert actual_out == expected_out


def test_expand_variable():
    var = tf.constant([1, 2, 3, 4, 5])
    shape = (4, 2)
    nr_hashed_vars = 3
    hash_group_size = 2

    new_var = mgu.expand_variable(var, shape=shape, nr_hashed_vars=nr_hashed_vars, hash_group_size=hash_group_size)

    sess = tf.Session()
    ev_var = sess.run(new_var)
    print(ev_var)
    assert ev_var.shape == shape


def test_generate_quasi_sample():
    block = mgu.generate_quasi_sample(30, 10)
    assert block.shape == (np.power(2, 10), 30)

def test_generate_normal_sample():
    block = mgu.generate_normal_sample(30, 10)
    assert block.shape == (np.power(2, 10), 30)
