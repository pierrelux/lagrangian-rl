import jax
import jax.numpy as np
from fax.lagrangian import cga
from fax import converge
from jax import random


def make_saddle_point_problem(objective_function, equality_constraint):
    """Make a saddle-point problem, amenable to CGA, from the Lagrangian

    Args:
        objective_function (callable): Unary scalar-valued callable 'J(x)' where 'x' can be a
            tuple of numpy.ndarray.
        equality_constraints (callable): Unary vector-valued callable 'h(x)' hwere 'x' can be a
            tuple of numpy.ndarray.

    Returns:
        tuple: Tuple of two binary scalar-valued callables with signature 'f(x,y)' and 'g(x,y)'.
    """
    def _lagrangian(x, multipliers):
        return objective_function(x) - np.dot(multipliers, equality_constraint(x))

    def _f(a, b):
        return _lagrangian(a, b)

    def _g(a, b):
        return -_f(a, b)

    return (_f, _g)


if __name__ == "__main__":

    rng = random.PRNGKey(42)
    rng_x, rng_y = random.split(rng)
    init_vals = (random.uniform(rng_x, shape=(3,)), random.uniform(rng_y))

    def make_objective_function(u):
        def _objective_function(x):
            return np.dot(x, u)
        return _objective_function

    def equality_constraint(x):
        return np.sum(x**2) - 1

    u = np.array([4, 5, 6])
    objective_function = make_objective_function(u)
    f, g = make_saddle_point_problem(objective_function, equality_constraint)

    eta = 0.1
    rtol = atol = 1e-8
    max_iter = 3000

    def convergence_test(x_new, x_old):
        return converge.max_diff_test(x_new, x_old, rtol, atol)

    solution = cga.cga_iteration(init_vals, f, g, convergence_test, max_iter, eta)

    print(solution, u/np.linalg.norm(u))
