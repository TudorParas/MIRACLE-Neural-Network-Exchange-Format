"""Static functions declared here so as not to complicate the miracle_graph class"""
import numpy as np
import tensorflow as tf
import logging

from sobol_seq import i4_sobol_generate
from scipy.stats import norm

def parse_compressed_size(compressed_size_bytes, nr_actual_vars, block_size_vars, bits_per_block):
    """
    Infer the block_size_vars and the bits_per_block from the user defined size. Always round up
    If the user specifies block_size_vars and bits_per_block then use that.

    We use a default size of 12 bits per block unless the user's arguments override it.
    """
    if block_size_vars is not None and bits_per_block is not None:
        if compressed_size_bytes is not None:
            logging.warning("Using user specified block_size_vars {}, bits_per_block {}. Ignoring specified size.".format(
                block_size_vars, bits_per_block))
        else:
            logging.info("Using block_size_vars {}, bits_per_block {}".format(block_size_vars, bits_per_block))
        return block_size_vars, bits_per_block

    if compressed_size_bytes is None:
        raise ValueError("Either specify a final compression size, or specify the number "
                         "of variables and the number of bits per block.")
    if nr_actual_vars is None:
        raise ValueError("Please specify the number of variables")
    compressed_size_bits = compressed_size_bytes * 8

    if block_size_vars is not None:
        nr_total_blocks = np.ceil(nr_actual_vars / block_size_vars)
        bits_per_block = np.ceil(compressed_size_bits / nr_total_blocks)
    else:
        if bits_per_block is None:
            bits_per_block = 10
        nr_total_blocks = np.ceil(compressed_size_bits / bits_per_block)
        block_size_vars = np.ceil(nr_actual_vars / nr_total_blocks)

    logging.info("Using block_size_vars {}, bits_per_block {}".format(block_size_vars, bits_per_block))
    return int(block_size_vars), int(bits_per_block)


def expand_variable(var, shape, nr_hashed_vars, hash_group_size):
    """
    Given a flattened variable var, do:
        - repeat the first 'nr_hashed_vars' in 'var' a 'hash_group_size' nrber of times
        - append to this the last variables
        - reshape the created variable into 'shape'
    """
    hashed_vars = var[:nr_hashed_vars]
    expanded_hashed = tf.multiply(tf.expand_dims(hashed_vars, axis=1), np.ones(shape=hash_group_size),
                                  name='var_expansion')
    expanded_hashed = tf.reshape(expanded_hashed, shape=[-1], name='flatten')  # flatten them

    expanded_vars = tf.concat([expanded_hashed, var[nr_hashed_vars:]], axis=0, name='expanded_vars')

    return tf.reshape(expanded_vars, shape=shape)


def generate_quasi_sample(block_size_vars, bits_per_block):
    """Generate a quasi sample that is more evently distributed that if we were to use randomness"""
    samples = np.power(2, bits_per_block)
    uni_quasi = np.array(i4_sobol_generate(block_size_vars, samples, skip=1))  # generate samples that are evenly spaces. sobol_generate(block_size, 2**bits_per_block)
    normal_quasi = norm.ppf(uni_quasi)

    return normal_quasi


def generate_normal_sample(block_size_vars, bits_per_block):
    """Generate the block that we will use to sample. Generating it now for all functions saves time"""
    samples = np.power(2, bits_per_block)  # total nr of samples we consider
    sample_block = np.random.normal(size=[samples, block_size_vars])

    return sample_block