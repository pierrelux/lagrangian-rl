import collections
from functools import partial

import jax
from jax import random
import jax.numpy as np
from jax.experimental import vectorize

from fax import converge
from fax import lagrangian
from fax.lagrangian import cga

from lqr import discrete
from lqr import dynamical
from lqr import module
from lqr import pendulum
from lqr import util

import data as data_util

import numpy as onp
import scipy.linalg as olinalg


Demos = collections.namedtuple("Demos", "xs us")


def solve_discrete_lqr(lqr):
    pmat = olinalg.solve_discrete_are(lqr.A, lqr.B, lqr.Q, lqr.R)
    print("True pmat:", pmat)
    return olinalg.solve(lqr.R + lqr.B.T @ pmat @ lqr.B,
                         lqr.B.T @ pmat @ lqr.A,
                         sym_pos=True)


def generate_lqr_demos(xs, dynamics, x_goal, u_goal, qmat, rmat):
    amat, bmat = dynamical.linearize_dynamics(dynamics, x_goal, u_goal, 0)
    lqr = module.LQR(A=amat, B=bmat, Q=qmat, R=rmat)

    kmat = solve_discrete_lqr(lqr)
    print("True kmat:", kmat)
    policy = vectorize.vectorize("(i),()->(j)")(util.policy(kmat, x_goal))

    return Demos(xs=xs, us=policy(xs, np.zeros((), dtype=np.int32)))


def loss(targets, estimates):
    return np.mean((targets - estimates)**2)


def main():
    # general experiment parameters
    n = 2
    m = 1
    batch_size = 10
    num_train_samples = 1000

    seed = 0
    rng = random.PRNGKey(seed)

    numpy_seed = 42
    onp.random.seed(numpy_seed)

    pendulum_params = pendulum.PendulumParams(length=1, mass=1, g=-9.8,
                                              drag=0.01)
    dynamics = pendulum.pendulum_dynamics(pendulum_params)

    x_goal = np.array([np.pi, 0.])
    u_goal = np.zeros((m,))
    qmat = np.array([[2, 0.],
                     [0., 1]])
    rmat = np.ones((1, 1))

    # learning parameters
    lr = 0.5
    rtol = 1e-3
    atol = 1e-5

    # generate data
    rng, xs_key = random.split(rng)
    demo_xs = np.stack((random.uniform(xs_key, (num_train_samples,),
                                       minval=np.pi - np.pi/6,
                                       maxval=np.pi + np.pi/6),
                        random.uniform(xs_key, (num_train_samples,),
                                       minval=-0.5,
                                       maxval=0.5)),
                       axis=-1)
    demos = generate_lqr_demos(demo_xs, dynamics, x_goal, u_goal, qmat, rmat)
    batch_gen = data_util.generate_batches(demos, batch_size,
                                           drop_remainder=True,
                                           shuffle=True)
    placeholder_batch = next(batch_gen)

    # reset the batch generator
    batch_gen = data_util.generate_batches(demos, batch_size,
                                           drop_remainder=True,
                                           shuffle=True)

    # set up lagrangian for the constrained optimization
    params_init, get_lqr = module.lqr()

    def batch_loss(params, data):
        pmat, lqr = get_lqr(params)

        kmat = discrete.gain_matrix(pmat, lqr)
        policy = vectorize.vectorize("(i),()->(j)")(util.policy(kmat, x_goal))
        us = policy(data.xs, np.zeros((), dtype=np.int32))

        return loss(data.us, us)

    def constraints(params, data):
        del data
        pmat, lqr = get_lqr(params)
        return discrete.riccati_operator(pmat, lqr) - pmat

    mult_init, lagr_func, get_params = lagrangian.make_lagrangian(batch_loss,
                                                                  constraints)

    # set up training functions
    opt_init, opt_update, get_lagr_params = cga.cga_lagrange_min(lr, lagr_func)

    def convergence_test(x_new, x_old):
        return converge.max_diff_test(x_new, x_old, rtol, atol)

    @jax.jit
    def step(i, opt_state, data):
        params = get_lagr_params(opt_state)

        grads = jax.grad(lagr_func, (0, 1))(*params, data=data)
        return opt_update(i, grads, opt_state, data=data)

    # initialize all parameters
    rng, params_key = random.split(rng)

    params = params_init(params_key, (n, m))
    lagr_params = mult_init(params, data=placeholder_batch)
    opt_state = opt_init(lagr_params)

    for i in range(500):
        old_params = get_lagr_params(opt_state)
        opt_state = step(i, opt_state, data=next(batch_gen))

        if convergence_test(get_lagr_params(opt_state), old_params):
            print("CONVERGED!! Step:", i)
            break

    params = get_lagr_params(opt_state)
    pmat, lqr = get_lqr(get_params(params))
    print(pmat)
    print(discrete.gain_matrix(pmat, lqr))
    print(lqr)


if __name__ == "__main__":
    main()
