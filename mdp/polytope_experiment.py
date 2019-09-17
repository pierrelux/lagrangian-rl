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


def save_solution(params, temperature_transition, prefix='solution'):
    transition_logits, reward_hat = params
    transition_hat = softmax((1./temperature_transition)*transition_logits)

    solution = (onp.asarray(transition_hat), onp.asarray(reward_hat))
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"{prefix}-{timestamp}.pkl"

    with open(filename, 'wb') as file:
        pickle.dump(solution, file)


def implicit_differentiation(mdp, temperature, temperature_transition):
    nactions, nstates = mdp[0].shape[:2]
    true_transition, true_reward, true_discount = mdp
    initial_distribution = np.ones(nstates)/nstates
    expert_policy = np.array([[0.2, 0.8], [0.7, 0.3]])

    # objective = jit(make_sep_objective(mdp, initial_distribution,
    #                                   temperature, temperature_transition, expert_policy))
    objective = jit(make_omd_objective(mdp, initial_distribution,
                                       temperature, temperature_transition))

    opt_init, opt_update, get_params = optimizers.adam(step_size=1e-2)

    @jit
    def update(i, opt_state):
        params = get_params(opt_state)
        return opt_update(i, grad(objective)(params), opt_state)

    init_params = (np.zeros_like(true_transition), np.zeros_like(true_reward))
    opt_state = opt_init(init_params)

    for i in range(100):
        opt_state = update(i, opt_state)
        params = get_params(opt_state)
        objective_value = objective(params)
        print(objective_value)

    save_solution(params, temperature_transition)


def smooth_bellman_optimality_operator(x, params):
    transition, reward, discount, temperature = params
    return reward + discount * np.einsum('ast,t->sa', transition, temperature *
                                         logsumexp((1. / temperature) * x, axis=1))


def competitive_differentiation(mdp, temperature, temperature_transition):
    nstates = mdp[0].shape[-1]
    true_transition, true_reward, true_discount = mdp
    initial_distribution = np.ones(nstates)/nstates

    @jit
    def f(decision_variables):
        x, theta = decision_variables
        transition_logits, reward_hat = theta

        transition_hat = softmax((1./temperature_transition)*transition_logits)
        op_params = (transition_hat, reward_hat, true_discount, temperature)

        return smooth_bellman_optimality_operator(x, op_params)

    @jit
    def objective(decision_variables):
        x, _ = decision_variables
        policy = softmax((1./temperature)*x)
        vf, _ = policy_evaluation(*mdp, policy)
        return -(initial_distribution @ vf)

    @jit
    def equality_constraint(decision_variables):
        x, _ = decision_variables
        return f(decision_variables) - x

    # L((x, theta), lambda). Decision_variables are (x, theta)
    init_mult, lagrangian, get_decision_variables = lagr_util.make_lagrangian(
        objective, equality_constraint)

    step_size = 0.16
    opt_init, opt_update, get_lagrangian_variables = cga.cga_lagrange_min(
        step_size, lagrangian, lr_multipliers=0.925)

    rtol = atol = 1e-6

    @jit
    def convergence_test(x_new, x_old):
        return converge.max_diff_test(x_new, x_old, rtol, atol)

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
                                           temperature, temperature_transition))

    for i in range(1000):
        old_lagrangian_variables = get_lagrangian_variables(opt_state)
        old_decision_variables = get_decision_variables(old_lagrangian_variables)
        print(
            i, objective(old_decision_variables),
            np.linalg.norm(equality_constraint(old_decision_variables)),
            omd_objective(old_decision_variables[-1])
        )
        opt_state = step(i, opt_state)
        if convergence_test(get_lagrangian_variables(opt_state), old_lagrangian_variables):
            break

    lagrangian_variables = get_lagrangian_variables(opt_state)
    decision_variables = get_decision_variables(lagrangian_variables)
    save_solution(decision_variables[1], temperature_transition, prefix='competitive')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--implicit', action='store_true')
    parser.add_argument('--temperature', type=float, default=1e-2)
    parser.add_argument('--temperature_transition', type=float, default=1e-2)
    arguments = parser.parse_args()

    mdp = polytope.dadashi_fig2d()
    method = implicit_differentiation if arguments.implicit is True else competitive_differentiation

    t0 = time.time()
    method(mdp, arguments.temperature, arguments.temperature_transition)
    print('Time: ', time.time() - t0)
