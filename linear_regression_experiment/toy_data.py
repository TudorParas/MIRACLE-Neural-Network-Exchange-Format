"""This file is meant to create the toy data for training"""

import numpy as np



def create_data(samples=1000, rows=10, num_outputs=None):
    """
    Create toy data consisting of

    Parameters:
        samples: int
            Number of input-output pairs
        rows: int
            Number of rows of the input and of the output
        num_outputs: int
            Number of outputs we want
    Returns:
        tuple
            Tuple of array of input vectors and array of output vectors
    """

    input_data = np.random.random(size=(samples, rows))
    output_data = create_output_data(input_data, num_outputs)

    return input_data, output_data


def _sum_max_function(vector):
    """Given a numpy array return a numpy array of its sum and its average"""
    return np.array([vector.sum(), vector.mean()])


def _avg_function(vector):
    """Just return the sum"""
    return np.array([vector.mean()])


def create_output_data(input_data, num_outputs):
    """
    Create the output data by appying the function to all the output data

    Parameters:
        input_data: np.array
        func: np.array -> np.array

    Returns:
        np.array
            Apply func to each input vector
    """
    if num_outputs == 1:
        func = _avg_function
    else:
        func = _sum_max_function
    return np.array(list(map(func, input_data)))



