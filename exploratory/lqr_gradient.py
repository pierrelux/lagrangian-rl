import jax
import jax.numpy as np
from fax.implicit.twophase import two_phase_solver
from jax import grad, jit, random


def solve_dare(A, B, Q, R):
    def _make_riccati_operator(params):
        A, B, Q, R = params

        def _riccati_operator(i, P):
            del i
            X = R + B.T @ P.T @ B
            Y = B.T @ P @ A
            return (A.T @ P @ A) - ((A.T @ P @ B) @ np.linalg.solve(X, Y)) + Q

        return _riccati_operator

    implicit_function = two_phase_solver(_make_riccati_operator)
    solution = implicit_function(np.eye(A.shape[0]), (A, B, Q, R))
    return solution.value


def lqr_evaluation(A, B, Q, R, K):
    def _make_lqr_eval_operator(params):
        A, B, Q, R, K = params

        def _lqr_eval_operator(i, P):
            del i
            return Q + (K.T @ R @ K) + ((A + B@K).T @ P @ (A + B@K))

        return _lqr_eval_operator

    implicit_function = two_phase_solver(_make_lqr_eval_operator)
    solution = implicit_function(np.eye(A.shape[0]), (A, B, Q, R, K))

    def _vf(x):
        return x.T @ solution.value @ x

    def _qf(x, u):
        return (x.T @ Q @ x) + (u.T @ R @ u) + _vf(A @ x + B @ u)

    return _vf, _qf


def riccati_policy(A, B, Q, R, P):
    del Q
    X = R + B.T @ P.T @ B
    Y = B.T @ P @ A
    K = -np.linalg.solve(X, Y)
    return K


def simulate(A, B, Q, R, K, x0):
    x = x0
    while True:
        u = K @ x
        cost = x.T @ Q @ x + u.T @ R @ u
        yield x, u, cost
        x = A @ x + B @ u


def spectral_radius(A):
    w, _ = np.linalg.eig(A)
    return np.max(np.abs(w))


if __name__ == "__main__":
    A = np.array([[1., 1.], [0., 1.]])
    B = np.array([[0.], [1.]])
    Q = np.array([[1., 0.], [0., 0.]])
    R = np.array([[1.]])
    x0 = np.array([-1, 0])

    P = solve_dare(A, B, Q, R)
    K = riccati_policy(A, B, Q, R, P)

    key = random.PRNGKey(0)
    for _ in range(100):
        key, subkey = random.split(key)
        K = jax.random.normal(subkey, shape=B.shape).T
        sr = spectral_radius(A + B@K)
        if sr < 1.:
            break

    @jit
    def objective(Khat):
        vf, _ = lqr_evaluation(A, B, Q, R, Khat)
        return vf(x0)

    gradfun = jit(grad(objective))
    for _ in range(100):
        K = K - 0.01*gradfun(K)
        print(objective(K))
