from jax import lax
from jax import ops
import jax.numpy as np
import jax.scipy

import dynamical


def forward_pass(x_init, K, k, F, f):

    def forward_step(x_t, params):
        (K_t, k_t, F_t, f_t) = params
        u_t = K_t @ x_t + k_t
        x_tp1 = F_t @ np.concatenate((x_t, u_t), axis=0) + f_t
        return x_tp1, (x_t, u_t)

    return lax.scan(forward_step, x_init, (K, k, F, f))[1]


def backwards_pass(C, c, F, f):

    def _lu_solve(p, l, u, b):
        pb = np.dot(p, b)
        y = jax.scipy.linalg.solve_triangular(l, pb, lower=True,
                                              unit_diagonal=True)
        return jax.scipy.linalg.solve_triangular(u, y, lower=False,
                                                 unit_diagonal=False)

    def backwards_step(carry, params):
        V_tp1, v_tp1 = carry
        (C_t, c_t, F_t, f_t) = params
        n = f_t.shape[0]

        FTV_tp1 = F_t.T @ V_tp1
        Q_t = C_t + FTV_tp1 @ F_t
        q_t = c_t + FTV_tp1 @ f_t + F_t.T @ v_tp1

        Q_x, Q_u = np.split(Q_t, (n,))
        Q_xx, Q_xu = np.split(Q_x, (n,), axis=1)
        Q_ux, Q_uu = np.split(Q_u, (n,), axis=1)
        q_x, q_u = np.split(q_t, (n,))

        p, l, u = jax.scipy.linalg.lu(Q_uu)
        K_t = -_lu_solve(p, l, u, Q_ux)
        k_t = -_lu_solve(p, l, u, q_u)

        KTQuu = K_t.T @ Q_uu
        V_t = Q_xx + Q_xu @ K_t + K_t.T @ Q_ux + KTQuu @ K_t
        v_t = q_x + Q_xu @ k_t + K_t.T @ q_u + KTQuu @ k_t
        return (V_t, v_t), (K_t, k_t)

    V_T = np.zeros((f.shape[1], f.shape[1]))
    v_T = np.zeros((f.shape[1],))
    K, kvec = lax.scan(backwards_step, (V_T, v_T), (C, c, F, f))[1]
    return lax.rev(K, (0,)), lax.rev(kvec, (0,))


def finite_horizon_lqr(amat, bmat, x_goal, u_goal, qmat, rmat, horizon):
    n = x_goal.shape[0]
    m = u_goal.shape[0]

    F = np.concatenate((amat, bmat), axis=1)
    F = np.repeat(F[None, :, :], horizon, axis=0)
    f = np.zeros((horizon, n))

    C = np.zeros((n + m, n + m))
    C = ops.index_update(C, ops.index[:n, :n], qmat)
    C = ops.index_update(C, ops.index[m:, m:], rmat)
    C = np.repeat(C[None, :, :], horizon, axis=0)
    c = np.zeros((horizon, n + m))

    K, k = backwards_pass(C, c, F, f)
    return K[0], k[0]
