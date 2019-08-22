import autograd.numpy as np
from autograd.tracer import getval
from autograd import make_vjp, grad
from autograd.misc.optimizers import adam
from autograd.extend import primitive, defvjp
from autograd.builtins import tuple as atuple


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


def riccati_operator(K, params):
    """Discrete-Time Algebraic Riccati Operator

    Args:
        K (numpy.ndarray): Candidate solution as a square matrix (M x M)
        A (numpy.ndarray): Square matrix (M x M)
        B (numpy.ndarray): Input (M x N)
        Q (numpy.ndarray): Input (M x M)
        R (numpy.ndarray): Square matrix (N x N)

    Returns:
        numpy.ndarray: Application of the operator to the input
    """
    A, B, Q, R = params
    V = A.T @ K @ A
    W = A.T @ K @ B
    X = R + B.T @ K.T @ B
    Y = B.T @ K @ A
    Z = np.linalg.solve(X, Y)
    return V - K - W@Z + Q


def generate_trajectory(policy, transition_function, reward_function, initial_function):
    """Simulate a policy in a given environment

    Args:
        policy (callable): Unary callable taking a state (np.ndarray)
            and returning an action (np.ndarray)
        transition_function (callable): Binary callable taking a state
            as first argument and action as second argument
        reward_function (callable): Binary callable taking a state
            as first argument and action as second argument
        x0 ([type]): Initial state
    """
    x = initial_function()
    while True:
        u = policy(x)
        r = reward_function(x, u)
        x, x_prev = transition_function(x, u), x
        yield np.hstack((x_prev, u, r, x))


def make_lqr_env(A, B, Q, R, x0):
    """Wrap an LQR problem into a functional environment

    Args:
        A (numpy.ndarray): Square matrix (M x M)
        B (numpy.ndarray): Input (M x N)
        Q (numpy.ndarray): Input (M x M)
        R (numpy.ndarray): Square matrix (N x N)
        x0 (numpy.ndarray): Fixed initial state (M, )

    Returns:
        tuple: Triple consisting of a transition, reward and initial function (in that order).
    """
    def _lqr_transition_function(x, u):
        return A @ x + B @ u

    def _lqr_reward_function(x, u):
        return x.T @ Q @ x + u.T @ R @ u

    def _lqr_initial_function():
        return x0

    return (_lqr_transition_function, _lqr_reward_function, _lqr_initial_function)


def take_samples(trajectory_gen, steps=10):
    """Consume a number of elements from a generator/iterable

    Args:
        trajectory_gen (iterable): Trajectory generator
        steps (int, optional): Number of samples to obtain. Defaults to 10.

    Returns:
        list: list of 'steps' elements taken from the iterable
    """
    return np.vstack([data for _, data in zip(range(steps), trajectory_gen)])


def log_diagonal_normal_pdf(policy, scale, states, actions):
    """Log probability of taking some actions in some states

    Args:
        policy (callable): Unary callable taking a state and returning an action
        variance (numpy.ndarray): Variance
        states (numpy.ndarray): Batch of states (nbatch, ndim_states)
        actions (numpy.ndarray): Batch of actions (nbatch, ndim_states)

    Returns:
        numpy.ndarray: Array of size (nbatch, 1)
    """
    variance = scale**2
    log_scale = np.log(np.sqrt(2.*np.pi*variance))
    distance = actions - policy(states)
    return -log_scale - 0.5*(1./variance)*(distance**2)


fixed_point = primitive(fixed_point)
defvjp(fixed_point, None, fixed_point_vjp)


def make_riccati_policy(K, A, B, Q, R):
    """Optimal policy from static gain matrix, expressed as a closure

    Args:
        K (numpy.ndarray): Solution to the discrete algebraic Riccati equation, (M x M)
        A (numpy.ndarray): Square matrix (M x M)
        B (numpy.ndarray): Input (M x N)
        Q (numpy.ndarray): Input (M x M)
        R (numpy.ndarray): Square matrix (N x N)

    Returns:
        callable: Unary callable taking a state (np.ndarray) and returning an action (np.ndarray)
    """
    del Q
    X = R + B.T @ K.T @ B
    Y = B.T @ K @ A
    Z = -np.linalg.solve(X, Y).T

    def _policy(state):
        return np.dot(state, Z)

    return _policy


def make_smooth_policy(policy, scale, rng):
    def _diagonal_normal_policy(state):
        return rng.normal(getval(policy(state)), scale)
    return _diagonal_normal_policy


def solve_riccati(A, B, Q, R):
    params = atuple((A, B, Q, R))
    return fixed_point(lambda k, p: k + riccati_operator(k, p),
                       params, np.zeros_like(A), distance_predicate(tol=1e-5))


def take_rollouts(policy, env, nrollouts=1, trajectory_len=100):
    rollouts = [take_samples(generate_trajectory(policy, *env), trajectory_len)
                for _ in range(nrollouts)]
    return np.dstack(rollouts)


if __name__ == "__main__":
    A = np.array([[1., 1.], [0., 1.]])
    B = np.array([[0.], [1.]])
    Q = np.array([[1., 0.], [0., 0.]])
    R = np.array([[1.]])

    x0 = np.array([-1, 0])
    env = make_lqr_env(A, B, Q, R, x0)
    standard_deviation = 1e-2

    rng = np.random.RandomState(0)

    def performance_measure(Ahat):
        params = atuple((Ahat, B, Q, R))
        K = solve_riccati(*params)
        riccati_policy = make_riccati_policy(K, *params)

        # Take multiple rollouts
        smooth_riccati_policy = make_smooth_policy(riccati_policy, standard_deviation, rng)
        samples = take_samples(generate_trajectory(smooth_riccati_policy, *env), 100)
        states = samples[:, :2]
        actions = samples[:, 2]
        rewards = samples[:, 3]

        reward_accumulation = np.cumsum(rewards[::-1])[::-1]
        logpdf = log_diagonal_normal_pdf(
            riccati_policy, standard_deviation, states, actions[:, np.newaxis])

        return np.mean(reward_accumulation[:, np.newaxis]*logpdf)

    def callback(x, i, g):
        K = solve_riccati(x, B, Q, R)
        samples = take_rollouts(make_riccati_policy(K, x, B, Q, R), env, nrollouts=1)
        print(f"{i}, {performance_measure(x):.8f}, {np.mean(samples[:, 3, :])}")

    Ahat_init = np.array([[1., 0.8], [0., 0.]])

    gradfun = grad(performance_measure)
    solution = adam(lambda x, i: gradfun(x), Ahat_init,
                    callback=callback, step_size=0.01, num_iters=100)
