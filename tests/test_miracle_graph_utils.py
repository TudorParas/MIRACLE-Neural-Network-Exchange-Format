import miracle.miracle_graph_utils as mgu

def test_parse_compressed_size():
    nr_actual_vars = 23
    block_size_vars = 5
    bits_per_block = 10
    size = 6
    call_args = [(None, nr_actual_vars, block_size_vars, bits_per_block), (size, nr_actual_vars, None, bits_per_block),
                 (size, nr_actual_vars, block_size_vars, None), (size, nr_actual_vars, None, None)]
    expected_outputs = [(block_size_vars, bits_per_block), (5, bits_per_block), (block_size_vars, 10), (6, 12)]

    for arguments, expected_out in zip(call_args, expected_outputs):
        actual_out = mgu.parse_compressed_size(*arguments)
        assert actual_out == expected_out
