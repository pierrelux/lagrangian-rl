import jax
import jax.numpy as np
from jax import grad, jit
from jax.scipy.special import logsumexp
from jax.experimental import optimizers

from jax.experimental.stax import softmax
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


if __name__ == "__main__":
    temperature = 1e-2
    temperature_transition = 1e-2

    mdp = polytope.dadashi_fig2d()
    nstates = mdp[0].shape[-1]
    true_transition, true_reward, true_discount = mdp

    initial_distribution = np.ones(nstates)/nstates

    def param_func(params):
        transition, reward = params

        def _smooth_bellman_operator(i, qf):
            del i
            return reward + true_discount * np.einsum(
                'ast,t->sa', transition, temperature * logsumexp((1. / temperature) * qf, axis=1))
        return _smooth_bellman_operator

    smooth_value_iteration = two_phase_solver(param_func)

    @jit
    def omd_objective(params):
        transition_logits, reward_hat = params
        transition_hat = softmax((1./temperature_transition)*transition_logits)

        q0 = np.zeros_like(reward_hat)
        solution = smooth_value_iteration(q0, (transition_hat, reward_hat))

        policy = softmax((1./temperature)*solution.value)
        vf, _ = policy_evaluation(*mdp, policy)

        return -(initial_distribution @ vf)

    opt_init, opt_update, get_params = optimizers.adam(step_size=1e-2)

    @jit
    def update(i, opt_state):
        params = get_params(opt_state)
        return opt_update(i, grad(omd_objective)(params), opt_state)

    p0 = (np.zeros_like(true_transition), np.zeros_like(true_reward))
    opt_state = opt_init(p0)

    for i in range(100):
        opt_state = update(i, opt_state)
        params = get_params(opt_state)
        objective_value = omd_objective(params)
        print(objective_value)
