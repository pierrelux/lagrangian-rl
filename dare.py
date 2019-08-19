import autograd.numpy as np
from autograd import make_vjp
from autograd.test_util import check_grads
from autograd.extend import primitive, defvjp
from autograd.builtins import tuple as atuple
from autograd.scipy.linalg import solve_discrete_are


def riccati_operator(K, params):
    A, B, Q, R = params
    V = A.T @ K @ A
    W = A.T @ K @ B
    X = R + B.T @ K.T @ B
    Y = B.T @ K @ A
    Z = np.linalg.solve(X, Y)
    return V - K - W@Z + Q


def euclidean_distance(x, y):
    """L2 distance
    """
    return np.linalg.norm(x-y)


def distance_predicate(tol=1e-8, distance=euclidean_distance):
    """ Returns True when the distance between two values is smaller than some tolerance.
    """
    def _p(x, y):
        return distance(x, y) <= tol
    return _p


def fixed_point(f, params, initial_value, termination_condition):
    """Iterates an operator up to a prescribed distance

    Args:
      f (callable): Unary callable producing a new iterate from an old one
      initial_value (np.ndarray): Initial value in the iteration sequence.
      termination_condition (callable): Binary callable taking current and previous iterates to
        decide (True) to terminate or not (False).
      operator_update (callable, optional): Callback function called for each new iterate.

    Returns:
      np.ndarray: Fixed-point value
    """
    x, x_prev = f(initial_value, params), initial_value

    while not termination_condition(x, x_prev):
        x, x_prev = f(x, params), x

    return x


def basic_iterative_solver(A, b):
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

    def f(x, params):
        _A, _b = params
        return _b + _A(x)

    return fixed_point(f, (_linop, b), np.zeros_like(b), distance_predicate())


def implicit_vjp(f, xstar, params):
    """Computes $\\bar{x}^\\top \\frac{dx^\\star}{d\\theta}$

    Args:
        f (callable): binary callable
        xstar (np.ndarray): fixed-point value
        params (tuple): parameters for the operator f

    Returns:
        np.ndarray: vector-Jacobian product at the fixed-point
    """
    vjp_xstar, _ = make_vjp(f, argnum=0)(xstar, params)
    vjp_params, _ = make_vjp(f, argnum=1)(xstar, params)

    def _vjp(xbar):
        ybar = basic_iterative_solver(vjp_xstar, xbar)
        return vjp_params(ybar)

    return _vjp


def fixed_point_vjp(ans, f, params, initial_value, termination_condition):
    """Equivalent to Christianson's (1994) two-phases reverse accumulation method.
    """
    del initial_value
    del termination_condition
    return implicit_vjp(f, ans, params)


fixed_point = primitive(fixed_point)
defvjp(fixed_point, None, fixed_point_vjp, None)

if __name__ == "__main__":
    A = np.array([[1., 1.], [0., 1.]])
    B = np.array([[0.], [1.]])
    Q = np.array([[1., 0.], [0., 0.]])
    R = 1.

    def F(K, params):
        return K + riccati_operator(K, params)

    K0 = np.zeros_like(A)

    def test(a):
        params = atuple((a, B, Q, R))
        return fixed_point(F, params, K0, distance_predicate())

    check_grads(test, modes=['rev', ], order=1)(A)
