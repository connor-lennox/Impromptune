import numpy as np


def _get_angles(pos, i, depth):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(depth))
    return pos * angle_rates


def position_encoding(position, depth):
    angle_rads = _get_angles(np.arange(position)[:, np.newaxis],
                             np.arange(depth)[np.newaxis, :],
                             depth)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return pos_encoding
