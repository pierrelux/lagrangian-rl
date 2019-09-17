from jax import lax
from jax import ops
from jax import tree_util
import jax.numpy as np
import jax.scipy

from fax import lagrangian


def rollout(x_init, lqr, us):

    if lqr.A.ndim < 3:
        T = us.shape[0]
        lqr = tree_util.tree_map(lambda x: np.repeat(x[None, :, :], (T, 1, 1)),
                                 lqr)

    def forward_step(x_t, params):
        lqr, u_t = params
        x_tp1 = lqr.A @ x_t + lqr.B @ u_t
        return x_tp1, None

    return lax.scan(forward_step, x_init, (lqr, us))[1]


def forward_recursion(x_init, K, k, F, f):

    def forward_step(x_t, params):
        (K_t, k_t, F_t, f_t) = params
        u_t = K_t @ x_t + k_t
        x_tp1 = F_t @ np.concatenate((x_t, u_t), axis=0) + f_t
        return x_tp1, (x_t, u_t)

    return lax.scan(forward_step, x_init, (K, k, F, f))[1]


def backwards_recursion(C, c, F, f):

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

        Quxqu = np.concatenate((Q_ux, q_u[:, None]), axis=1)
        Ktkt = -jax.scipy.linalg.solve(Q_uu, Quxqu, sym_pos=True)
        K_t, k_t = np.split(Ktkt, (n,), axis=1)
        k_t = k_t[:, 0]

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

    K, k = backwards_recursion(C, c, F, f)
    return K[0], k[0]


def make_finite_horizon_lagrangian():

    def cost(params):
        lqr, xs, us = params

        einsum_op = "ij,ti->tj"
        if lqr.A.ndim == 3:
            einsum_op = "t" + einsum_op

        return (np.einsum(einsum_op, lqr.Q, xs)
                + np.einsum(einsum_op, lqr.R, us))

    def constraints(params):
        lqr, xs, us = params

        einsum_op = "ij,ti->tj"
        if lqr.A.ndim == 3:
            einsum_op = "t" + einsum_op

        return (np.einsum(einsum_op, lqr.A, xs)
                + np.einsum(einsum_op, lqr.B, us))

    return lagrangian.make_lagrangian(cost, constraints)
