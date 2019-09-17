import jax.numpy as np


def policy(kmat, x_goal, kvec=None):
    def policy(x, t):
        kmat_t = kmat
        kvec_t = kvec

        if kmat.ndim > 2:
            kmat_t = kmat[t]
            if kvec_t is not None:
                kvec_t = kvec[t]

        u_t = np.dot(kmat_t, x_goal - x)

        if kvec_t is not None:
            u_t = u_t + kvec_t

        return u_t

    return policy
