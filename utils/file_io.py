from bitstring import BitStream

P_SCALE_LENGTH = 32  # How many bits we use to represent pscale


def dump_to_file(p_scale_vars, seeds, bits_per_block, out_file):
    """Write the p_scale and the seeds to file in a bit-efficient way

    Parameters
    ---------

    p_scale_vars: list
    seeds: list
    bits_per_block: int
    out_file: str
    """
    # Create the bitstream,
    stream = BitStream()
    # Add the p_scale_vars to the bitstream
    for p_scale_var in p_scale_vars:
        stream.append(BitStream(float=p_scale_var, length=P_SCALE_LENGTH))
    # Add the seeds to the bitstream
    for seed in seeds:
        stream.append(BitStream(uint=seed, length=bits_per_block))

    # Dump the reads to file
    with open(out_file, mode='wb+') as f:
        stream.tofile(f)


def load_from_file(model_file, bits_per_block, p_scale_number, seeds_number):
    """Retrieve the p_scales and the seeds from file

    Parameters
    ---------

    model_file: str
    bits_per_block: int
    p_scale_number: int
        How many p scales do we read
    seeds_number: int
        How many seeds do we read
    """
    with open(model_file, mode='rb') as f:
        stream = BitStream(f)

    p_scale_vars = list()
    for index in range(p_scale_number):
        p_scale_vars += stream.readlist('float: {}'.format(P_SCALE_LENGTH))
    seeds = list()
    for index in range(seeds_number):
        seeds += stream.readlist('uint: {}'.format(bits_per_block))

    return p_scale_vars, seeds
