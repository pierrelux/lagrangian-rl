import time
import pickle
import argparse
import numpy as onp

import jax
import jax.numpy as np
from jax import grad, jit
from jax.scipy.special import logsumexp
from jax.experimental import optimizers
from jax.experimental.stax import softmax

from fax import converge
from fax.lagrangian import cga
from fax.lagrangian import util as lagr_util
from fax.implicit.twophase import two_phase_solver

import polytope


def policy_evaluation(P, R, gamma, policy):
    """ Direct policy evaluation

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
    ppi = np.einsum('ast,sa->st', P, policy)
    rpi = np.einsum('sa,sa->s', R, policy)
    vf = np.linalg.solve(np.eye(P.shape[-1]) - gamma*ppi, rpi)
    qf = R + gamma*np.einsum('ast,t->sa', P, vf)
    return vf, qf


def expected_discounted_loglikelihood(mdp, initial_distribution, policy, expert_policy):
    transition, reward, discount = mdp
    loglike_reward_ternary = np.log(np.einsum('sa,ast->sat', policy, transition))
    loglike_reward_binary = np.einsum('ast,sat->sa', transition, loglike_reward_ternary)
    vf, _ = policy_evaluation(transition, loglike_reward_binary, discount, expert_policy)

    return initial_distribution @ (np.log(initial_distribution) + vf)


def expected_return(mdp, initial_distribution, policy):
    vf, _ = policy_evaluation(*mdp, policy)
    return initial_distribution @ vf


def make_differentiable_planner(true_discount, temperature):
    def param_func(params):
        transition, reward = params

        def _smooth_bellman_operator(i, qf):
            del i
            return reward + true_discount * np.einsum(
                'ast,t->sa', transition, temperature * logsumexp((1. / temperature) * qf, axis=1))
        return _smooth_bellman_operator
    smooth_value_iteration = two_phase_solver(param_func)

    def _planner(params):
        transition_hat, reward_hat = params
        q0 = np.zeros_like(reward_hat)
        solution = smooth_value_iteration(q0, (transition_hat, reward_hat))
        return softmax((1./temperature)*solution.value)

    return _planner


def make_omd_objective(mdp, initial_distribution, temperature, temperature_logits):
    true_discount = mdp[-1]
    planner = make_differentiable_planner(true_discount, temperature)

    def _objective(params):
        transition_logits, reward_hat = params
        transition_hat = softmax((1./temperature_logits)*transition_logits)
        return -expected_return(mdp, initial_distribution, planner((transition_hat, reward_hat)))

    return _objective


def make_sep_objective(mdp, initial_distribution, temperature, temperature_logits, expert_policy):
    true_discount = mdp[-1]
    planner = make_differentiable_planner(true_discount, temperature)

    def _objective(params):
        return -expected_discounted_loglikelihood(
            mdp, initial_distribution, planner(params),
            expert_policy)
    return _objective


def save_solution(params, arguments, prefix='solution'):
    transition_logits, reward_hat = params
    transition_hat = softmax((1./arguments.temperature_transition)*transition_logits)

    solution = (onp.asarray(transition_hat), onp.asarray(reward_hat))
    data = {'solution': solution, 'args': arguments}

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"{prefix}-{timestamp}.pkl"

    with open(filename, 'wb') as file:
        pickle.dump(data, file)


def implicit_differentiation(mdp, arguments):
    nactions, nstates = mdp[0].shape[:2]
    true_transition, true_reward, true_discount = mdp
    initial_distribution = np.ones(nstates)/nstates
    expert_policy = np.array([[0.2, 0.8], [0.7, 0.3]])

    if arguments.structural is True:
        objective = jit(make_sep_objective(mdp, initial_distribution,
                                           arguments.temperature, arguments.temperature_transition,
                                           expert_policy))
    else:
        objective = jit(make_omd_objective(mdp, initial_distribution,
                                           arguments.temperature,
                                           arguments.temperature_transition))

    opt_init, opt_update, get_params = optimizers.adam(step_size=arguments.lr_performance)

    @jit
    def update(i, opt_state):
        params = get_params(opt_state)
        return opt_update(i, grad(objective)(params), opt_state)

    @jit
    def convergence_test(x_new, x_old):
        return converge.max_diff_test(x_new, x_old, arguments.rtol, arguments.atol)

    init_params = (np.zeros_like(true_transition), np.zeros_like(true_reward))
    opt_state = opt_init(init_params)

    for i in range(arguments.max_iter):
        params = get_params(opt_state)
        print(i, objective(params))
        opt_state = update(i, opt_state)
        if convergence_test(get_params(opt_state), params):
            break

    if arguments.discard is False:
        prefix = f"{'sep' if arguments.structural else 'omd'}-implicit"
        save_solution(params, arguments, prefix=prefix)


def smooth_bellman_optimality_operator(x, params):
    transition, reward, discount, temperature = params
    return reward + discount * np.einsum('ast,t->sa', transition, temperature *
                                         logsumexp((1. / temperature) * x, axis=1))


def competitive_differentiation(mdp, arguments):
    nstates = mdp[0].shape[-1]
    true_transition, true_reward, true_discount = mdp
    initial_distribution = np.ones(nstates)/nstates
    expert_policy = np.array([[0.2, 0.8], [0.7, 0.3]])
    expert_return = expected_return(mdp, initial_distribution, expert_policy)

    @jit
    def f(decision_variables):
        x, theta = decision_variables
        transition_logits, reward_hat = theta

        transition_hat = softmax((1./arguments.temperature_transition)*transition_logits)
        op_params = (transition_hat, reward_hat, true_discount, arguments.temperature)

        return smooth_bellman_optimality_operator(x, op_params)

    @jit
    def omd_objective(decision_variables):
        x, _ = decision_variables
        policy = softmax((1./arguments.temperature)*x)
        return -expected_return(mdp, initial_distribution, policy)

    @jit
    def sep_objective(decision_variables):
        x, _ = decision_variables
        policy = softmax((1./arguments.temperature)*x)
        return -expected_discounted_loglikelihood(mdp, initial_distribution, policy, expert_policy)

    @jit
    def equality_constraint(decision_variables):
        x, _ = decision_variables
        return f(decision_variables) - x

    # L((x, theta), lambda). Decision_variables are (x, theta)
    objective = omd_objective
    if arguments.structural is True:
        objective = sep_objective
    init_mult, lagrangian, get_decision_variables = lagr_util.make_lagrangian(
        objective, equality_constraint)

    opt_init, opt_update, get_lagrangian_variables = cga.cga_lagrange_min(
        arguments.lr_performance, lagrangian, lr_multipliers=arguments.lr_constraint)

    @jit
    def convergence_test(x_new, x_old):
        return converge.max_diff_test(x_new, x_old, arguments.rtol, arguments.atol)

    @jit
    def step(i, opt_state):
        lagrangian_variables = get_lagrangian_variables(opt_state)
        grads = jax.grad(lagrangian, (0, 1))(*lagrangian_variables)
        return opt_update(i, grads, opt_state)

    # (x, theta)
    init_decision_variables = (
        np.zeros_like(true_reward),
        (np.zeros_like(true_transition), np.zeros_like(true_reward))
    )
    lagrangian_variables = init_mult(init_decision_variables)
    opt_state = opt_init(lagrangian_variables)

    omd_objective = jit(make_omd_objective(mdp, initial_distribution,
                                           arguments.temperature, arguments.temperature_transition))

    for i in range(arguments.max_iter):
        old_lagrangian_variables = get_lagrangian_variables(opt_state)
        old_decision_variables = get_decision_variables(old_lagrangian_variables)
        print(
            i, objective(old_decision_variables),
            np.linalg.norm(equality_constraint(old_decision_variables)),
            omd_objective(old_decision_variables[-1]),
            expert_return
        )
        opt_state = step(i, opt_state)
        if convergence_test(get_lagrangian_variables(opt_state), old_lagrangian_variables):
            break

    lagrangian_variables = get_lagrangian_variables(opt_state)
    decision_variables = get_decision_variables(lagrangian_variables)
    if arguments.discard is False:
        prefix = f"{'sep' if arguments.structural else 'omd'}-competitive"
        save_solution(decision_variables[1], arguments, prefix=prefix)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--implicit', action='store_true')
    parser.add_argument('--structural', action='store_true')
    parser.add_argument('--temperature', type=float, default=1e-2)
    parser.add_argument('--temperature_transition', type=float, default=1e-2)
    parser.add_argument('--lr_performance', type=float, default=0.15)
    parser.add_argument('--lr_constraint', type=float, default=0.925)
    parser.add_argument('--discard', action='store_true')
    parser.add_argument('--rtol', type=float, default=1e-6)
    parser.add_argument('--atol', type=float, default=1e-6)
    parser.add_argument('--max_iter', type=int, default=1000)

    arguments = parser.parse_args()

    mdp = polytope.dadashi_fig2d()
    method = implicit_differentiation if arguments.implicit is True else competitive_differentiation

    t0 = time.time()
    method(mdp, arguments)
    print('Time: ', time.time() - t0)
