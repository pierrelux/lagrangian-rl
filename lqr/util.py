import jax.numpy as np


def riccati_operator(K, params):
    A, B, Q, R = params
    V = A.T @ K @ A
    W = A.T @ K @ B
    X = R + B.T @ K.T @ B
    Y = B.T @ K @ A
    Z = np.linalg.solve(X, Y)
    return V - K - W@Z + Q


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
