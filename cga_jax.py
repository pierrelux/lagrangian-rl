import jax
import jax.numpy as np
from jax import random, jvp, grad, jit


def make_mixed_jvp(f, x, y, reversed=False):
    """Make a mixed jacobian-vector product function

    Args:
        f (callable): Binary callable with signature f(x,y)
        x (numpy.ndarray): First argument to f
        y (numpy.ndarray): Second argument to f
        reversed (bool, optional): Take Dyx if False, Dxy if True. Defaults to False.

    Returns:
        callable: Unary callable 'jvp(v)' taking a numpy.ndarray as input.
    """
    if reversed is not True:
        given = y
        gradfun = grad(f, 0)

        def frozen_grad(y):
            return gradfun(x, y)
    else:
        given = x
        gradfun = grad(f, 1)

        def frozen_grad(x):
            return gradfun(x, y)

    def _jvp(v):
        return jvp(frozen_grad, (given,), (v,))[1]

    return _jvp


def basic_iterative_solver(A, b, x0=None, tol=1e-5):
    """Solve for x in (I-A)x=b using basic iterations without preconditioning/splitting.

    Args:
        A (callable): unary function evaluating Av for some given vector v.
        b (np.ndarray): Right hand side of the linear system.

    Returns:
        np.ndarray: The converged solution.
    """
    if not callable(A):
        def _linop(x):
            return np.dot(A, x)
    else:
        _linop = A

    if x0 is None:
        x0 = np.zeros_like(b)

    def f(x):
        return b + A(x)

    x, xprev = f(x0), x0
    while np.linalg.norm(x - xprev) <= tol:
        x, xprev = f(x), x

    return x


if __name__ == "__main__":
    key = random.PRNGKey(0)
    A = jax.random.uniform(key, shape=(2, 3))
    x = jax.random.uniform(key, shape=(2,))
    y = jax.random.uniform(key, shape=(3,))

    def f(x, y):
        return x.T @ A @ y

    def g(x, y):
        return -f(x, y)

    eta = 0.5
    grad_yg = jit(grad(g, 1))
    grad_xf = jit(grad(f, 0))

    for i in range(100):
        print(i, f(x, y), np.linalg.norm(np.concatenate((x, y))))

        jvp_xyf = make_mixed_jvp(f, x, y)
        jvp_yxg = make_mixed_jvp(g, x, y, reversed=True)

        bx = grad_xf(x, y) - eta*jvp_xyf(grad_yg(x, y))
        deltax = -basic_iterative_solver(lambda x: (eta**2)*jvp_xyf(jvp_yxg(x)), bx)

        by = grad_yg(x, y) - eta*jvp_yxg(grad_xf(x, y))
        deltay = -basic_iterative_solver(lambda x: (eta**2)*jvp_yxg(jvp_xyf(x)), by)

        x = x + eta*deltax
        y = y + eta*deltay
