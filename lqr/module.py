import collections

from jax import lax
from jax import random
import jax.numpy as np

LQR = collections.namedtuple("LQR", "A B Q R")


def uniform_plus_identity(scale=1e-2):
    def init(key, shape, dtype=np.float32):
        return (np.eye(shape[0], shape[1])
                + random.uniform(key, shape, dtype) * scale)
    return init


def lqr(pmat_initializer=uniform_plus_identity(),
        amat_initializer=uniform_plus_identity(),
        bmat_initializer=uniform_plus_identity(),
        qmat_initializer=uniform_plus_identity(),
        rmat_initializer=uniform_plus_identity()):

    def init(key, shape, dtype=np.float32):
        # assume shape is (num_state_dims, num_action_dims)
        n, m = shape

        p_key, a_key, b_key, q_key, r_key = random.split(key, 5)

        pmat = pmat_initializer(p_key, (n, n), dtype=dtype)
        lqr = LQR(
            A=amat_initializer(a_key, (n, n), dtype=dtype),
            B=bmat_initializer(b_key, (n, m), dtype=dtype),
            Q=qmat_initializer(q_key, (n, n), dtype=dtype),
            R=rmat_initializer(r_key, (m, m), dtype=dtype),
        )
        return pmat, lqr

    def apply(params, inputs=None, **kwargs):
        del inputs, kwargs
        pmat, lqr = params
        lqr = LQR(
            A=lqr.A,
            B=lqr.B,
            Q=lqr.Q.T @ lqr.Q,
            R=lqr.R.T @ lqr.R,
        )
        return pmat, lqr

    return init, apply


def timevarying_lqr(pmat_initializer=uniform_plus_identity(),
                    amat_initializer=uniform_plus_identity(),
                    bmat_initializer=uniform_plus_identity(),
                    qmat_initializer=uniform_plus_identity(),
                    rmat_initializer=uniform_plus_identity()):

    def init(key, shape, dtype=np.float32):
        # assume shape is (horizon, num_state_dims, num_action_dims)
        T, n, m = shape

        def generate_lqr(key, _):
            new_key, p_key, a_key, b_key, q_key, r_key = random.split(key, 6)

            pmat = pmat_initializer(p_key, (n, n), dtype=dtype)
            lqr = LQR(
                A=amat_initializer(a_key, (n, n), dtype=dtype),
                B=bmat_initializer(b_key, (n, m), dtype=dtype),
                Q=qmat_initializer(q_key, (n, n), dtype=dtype),
                R=rmat_initializer(r_key, (m, m), dtype=dtype),
            )
            return new_key, (pmat, lqr)

        return lax.scan(generate_lqr, key, lax.iota(T))[1]

    def apply(params, inputs=None, **kwargs):
        del inputs, kwargs
        pmat, lqr = params
        lqr = LQR(
            A=lqr.A,
            B=lqr.B,
            Q=lqr.Q.T @ lqr.Q,
            R=lqr.R.T @ lqr.R,
        )
        return pmat, lqr

    return init, apply


