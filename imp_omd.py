import pickle

import autograd.numpy as np
from autograd.builtins import tuple as atuple
from autograd.scipy.special import logsumexp
from autograd import make_vjp, grad
from autograd.extend import primitive, defvjp
from autograd.misc.optimizers import adam


import polytope


def softmax(vals, temp=1., axis=-1):
    """Batch softmax
    Args:
        vals (np.ndarray): Typically S x A array
        t (float, optional): Defaults to 1.. Temperature parameter
        axis (int, optional): Defaults to -1 (last axis). Reduction axis (eg. the "action" axis)
    Returns:
        np.ndarray: Array of same size as input
    """
    return np.exp((1./temp)*vals - logsumexp((1./temp)*vals, axis=axis, keepdims=True))


def euclidean_distance(x, y):
    """L2 distance
    """
    return np.linalg.norm(x-y)


def distance_predicate(tol=1e-5, distance=euclidean_distance):
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


def policy_evaluation(P, R, gamma, policy):
    """ Policy Evaluation Solver

    We denote by 'A' the number of actions, 'S' for the number of
    states.

    Args:
      P (numpy.ndarray): Transition function as (A x S x S) tensor
      R (numpy.ndarray): Reward function as a (S x A) tensor
      gamma (float): Scalar discount factor
      policies (numpy.ndarray): tensor of shape (S x A)

    Returns:
      tuple (vf, qf) where the first element is vector of length S and the second element contains
      the Q functions as matrix of shape (S x A).
    """
    nstates = P.shape[-1]
    expanded = False
    if policy.ndim < 3:
        policy = np.expand_dims(policy, axis=0)
        expanded = True

    R = np.expand_dims(R, axis=0)
    ppi = np.einsum('ast,nsa->nst', P, policy)
    rpi = np.einsum('nsa,nsa->ns', R, policy)
    vf = np.linalg.solve(np.eye(nstates) - gamma*ppi, rpi)
    qf = R + gamma*np.einsum('ast,nt->nsa', P, vf)

    if expanded is True:
        vf = np.squeeze(vf)
        qf = np.squeeze(qf)

    return vf, qf


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


def smooth_bellman_optimality_operator(qf, params):
    """Smooth Bellman optimality operator

    The 'hard' max is replaced by log-sum-exp.

  Args:
    qf (np.ndarray): (S x A) array
    params (tuple): Tuple (P, R, gamma, temperature)

  Returns:
    np.ndarray: Backed up values in an array of size (S x A)
  """
    P, R, gamma, temperature = params
    return R + gamma*np.einsum('ast,t->sa', P, temperature*logsumexp((1./temperature)*qf, axis=1))


def bellman_optimality_operator(qf, params):
    """Bellman optimality operator

    Args:
      qf (np.ndarray): (S x A) array
      params (tuple): Tuple (P, R, gamma)

    Returns:
      np.ndarray: Backed up values in an array of size (S x A)
    """
    P, R, gamma = params
    return R + gamma*np.einsum('ast,t->sa', P, np.max(qf, axis=1))


fixed_point = primitive(fixed_point)
defvjp(fixed_point, None, fixed_point_vjp, None)

if __name__ == "__main__":
    mdp = polytope.DadashiFig2d()

    temperature_planner = 1e-2
    temperature_transition = 1e-2

    rng = np.random.RandomState(1234)
    initial_distribution = np.ones(mdp.nstates)/mdp.nstates

    def performance_measure(model_params):
        transition_hat = softmax(model_params[0], temp=temperature_transition)
        reward_hat = model_params[1]

        params = atuple((transition_hat, reward_hat, mdp.discount, temperature_planner))
        qstar_smooth = fixed_point(smooth_bellman_optimality_operator, params,
                                   np.zeros_like(mdp.R), distance_predicate(tol=1e-5))

        policy = softmax(qstar_smooth, temperature_planner)
        vf, _ = policy_evaluation(mdp.P, mdp.R, mdp.discount, policy)

        return -(initial_distribution @ vf)

    qstar_hard = fixed_point(bellman_optimality_operator, mdp.mdp,
                             np.zeros_like(mdp.R), distance_predicate(1e-8))
    optimal_value = initial_distribution @ np.max(qstar_hard, axis=1)

    synthetic_reward = np.zeros_like(mdp.R)
    transition_logits = np.zeros_like(mdp.P)

    gradfun = grad(performance_measure, argnum=0)
    x0 = atuple((transition_logits, synthetic_reward))

    def callback(x, i, g):
        print(f"{i}, {performance_measure(x):.8f}, {optimal_value:.8f}")

    solution = adam(lambda x, i: gradfun(x), x0, callback=callback,
                    step_size=0.01, num_iters=100)

    with open('solution.pkl', 'wb') as file:
        pickle.dump((softmax(solution[0], temperature_transition), solution[1]), file)
