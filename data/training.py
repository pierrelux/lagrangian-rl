import numpy as onp

from jax import tree_util


def batched_epoch(perm, data, batch_size, drop_remainder):
    start_idx = 0
    range_stop = perm.shape[0]
    if not drop_remainder:
        range_stop += batch_size - 1

    for end_idx in range(batch_size, range_stop, batch_size):
        idx = perm[start_idx:end_idx]
        batch = tree_util.tree_map(lambda x: x[idx], data)

        yield batch
        start_idx = end_idx


def generate_batches(data, batch_size, drop_remainder=True, shuffle=True):
    num_samples = len(data[0])
    while True:
        if shuffle:
            perm = onp.random.permutation(num_samples)
        else:
            perm = onp.arange(num_samples)
        yield from batched_epoch(perm, data, batch_size, drop_remainder)
