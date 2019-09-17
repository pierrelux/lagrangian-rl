import collections

from jax import lax
from jax import random
import jax.numpy as np

LQR = collections.namedtuple("LQR", "A B Q R")


def uniform_plus_identity(scale=1e-2):
    def init(key, shape, dtype=np.float32):
        return np.eye(shape) + random.uniform(key, shape, dtype) * scale
    return init


def chol_uniform_plus_identity(scale=1e-2):
    _init = uniform_plus_identity(scale)

    def init(key, shape, dtype=np.float32):
        x = _init(key, shape, dtype)
        return x.T @ x
    return init


def lqr(amat_initializer=uniform_plus_identity(),
        bmat_initializer=uniform_plus_identity(),
        qmat_initializer=chol_uniform_plus_identity(),
        rmat_initializer=chol_uniform_plus_identity()):

    def init(key, shape, dtype=np.float32):
        # assume shape is (num_state_dims, num_action_dims)
        n, m = shape

        a_key, b_key, q_key, r_key = random.split(key, 4)
        return LQR(
            A=amat_initializer(a_key, (n, n), dtype=dtype),
            B=bmat_initializer(b_key, (n, m), dtype=dtype),
            Q=qmat_initializer(q_key, (n, n), dtype=dtype),
            R=rmat_initializer(r_key, (m, m), dtype=dtype),
        )

    return init


def timevarying_lqr(amat_initializer=uniform_plus_identity(),
                    bmat_initializer=uniform_plus_identity(),
                    qmat_initializer=chol_uniform_plus_identity(),
                    rmat_initializer=chol_uniform_plus_identity()):

    def init(key, shape, dtype=np.float32):
        # assume shape is (horizon, num_state_dims, num_action_dims)
        T, n, m = shape

        def generate_lqr(key, _):
            new_key, a_key, b_key, q_key, r_key = random.split(key, 5)
            return new_key, LQR(
                A=amat_initializer(a_key, (n, n), dtype=dtype),
                B=bmat_initializer(b_key, (n, m), dtype=dtype),
                Q=qmat_initializer(q_key, (n, n), dtype=dtype),
                R=rmat_initializer(r_key, (m, m), dtype=dtype),
            )

        return lax.scan(generate_lqr, key, lax.iota(T))[1]

    return init


def lqr_traj(lqr_initializer, xs_initializer, us_initializer):

    def init(key, shape, dtype=np.float32):
        # assume shape is (num_state_dims, num_action_dims, horizon)
        T, n, m = shape

        lqr_key, xs_key, us_key = random.split(key, 3)

        try:
            lqr = lqr_initializer(lqr_key, shape, dtype)
        except ValueError:
            # assume initializer failed because it was not time-varying
            lqr = lqr_initializer(lqr_key, (n, m), dtype)

        xs = xs_initializer(xs_key, (T, n), dtype)
        us = us_initializer(us_key, (), dtype)

        return lqr, xs, us

    return init
