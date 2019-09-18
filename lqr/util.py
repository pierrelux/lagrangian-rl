import jax.numpy as np


def policy(kmat, x_goal=None, kvec=None):
    def policy(x, t):
        kmat_t = kmat
        kvec_t = kvec

        if kmat.ndim > 2:
            kmat_t = kmat[t]
            if kvec_t is not None:
                kvec_t = kvec[t]

        dx = -x
        if x_goal is not None:
            dx = dx + x_goal

        u_t = np.dot(kmat_t, dx)

        if kvec_t is not None:
            u_t = u_t + kvec_t

        return u_t

    return policy
