import collections
import itertools
import multiprocessing
import pickle
import time

import jax
from jax import random
from jax import tree_util
import jax.numpy as np
from jax.experimental import vectorize

from fax import converge
from fax import lagrangian
from fax.lagrangian import cga

from lqr import cartpole
from lqr import discrete
from lqr import dynamical
from lqr import module
from lqr import pendulum
from lqr import util

import data as data_util

import numpy as onp
import scipy.linalg as olinalg

import matplotlib.pyplot as plt
import tikzplotlib


Demos = collections.namedtuple("Demos", "xs us")


def safe_block_until_ready(x):
    try:
        x.block_until_ready()
    except AttributeError as e:
        pass


def solve_discrete_lqr(lqr):
    pmat = olinalg.solve_discrete_are(lqr.A, lqr.B, lqr.Q, lqr.R)
    return olinalg.solve(lqr.R + lqr.B.T @ pmat @ lqr.B,
                         lqr.B.T @ pmat @ lqr.A,
                         sym_pos=True)


def generate_lqr_demos(xs, x_goal, lqr):
    lqr = tree_util.tree_map(lambda x: onp.array(x).astype(np.float64), lqr)
    kmat = solve_discrete_lqr(lqr)

    policy = vectorize.vectorize("(i),()->(j)")(util.policy(kmat, x_goal))

    return Demos(xs=xs, us=policy(xs, np.zeros((), dtype=np.int32)))


def evaluate_lqr_policy(x_init, x_goal, u_goal, kmat, lqr, num_steps):
    x_init = x_init - x_goal

    policy = util.policy(kmat, kvec=-u_goal)

    @vectorize.vectorize("(n)->(i,n),(i,m),(i)")
    def evaluate(x):
        return discrete.rollout(x, lqr, policy, num_steps)

    x_final, (xs, us, cs) = evaluate(x_init)
    return xs + x_goal, us + u_goal, cs


def loss(targets, estimates):
    return np.mean((targets - estimates)**2)


def run_experiment(n, m, batch_size, num_train_samples, num_test_samples,
                   num_eval_steps, seed, numpy_seed, dynamics, x_goal, u_goal,
                   qmat, rmat, lr_cost, lr_constraints, rtol, atol, dt,
                   state_sampler, num_train_iterations):
    rng = random.PRNGKey(seed)
    onp.random.seed(numpy_seed)

    amat, bmat = dynamical.linearize_dynamics(dynamics, x_goal, u_goal, 0)
    amat = amat * dt
    bmat = bmat * dt
    qmat = qmat * dt
    rmat = rmat * dt

    true_lqr = module.LQR(A=amat, B=bmat, Q=qmat, R=rmat)
    opt_pmat = olinalg.solve_discrete_are(true_lqr.A, true_lqr.B, true_lqr.Q,
                                          true_lqr.R)

    # generate data
    rng, key = random.split(rng)
    train_xs = state_sampler(rng, num_train_samples)
    train_demos = generate_lqr_demos(train_xs, x_goal, true_lqr)

    rng, key = random.split(rng)
    test_xs = state_sampler(rng, num_test_samples)
    test_demos = generate_lqr_demos(test_xs, x_goal, true_lqr)

    # create a dummy batch to get dimensions
    batch_gen = data_util.generate_batches(train_demos, batch_size,
                                           drop_remainder=True,
                                           shuffle=True)
    placeholder_batch = next(batch_gen)

    # reset the batch generator
    batch_gen = data_util.generate_batches(train_demos, batch_size,
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
    opt_init, opt_update, get_lagr_params = cga.cga_lagrange_min(
        lr_cost, lagr_func, lr_multipliers=lr_constraints)

    def convergence_test(x_new, x_old):
        return converge.max_diff_test(x_new, x_old, rtol, atol)

    @jax.jit
    def step(i, opt_state, data):
        params = get_lagr_params(opt_state)

        val, grads = jax.value_and_grad(lagr_func, (0, 1))(*params, data=data)
        logs = {
            "lagrangian": val,
            "train loss": batch_loss(get_params(params), data),
        }
        return opt_update(i, grads, opt_state, data=data), logs

    # initialize all parameters
    rng, params_key = random.split(rng)

    params = params_init(params_key, (n, m))
    lagr_params = mult_init(params, data=placeholder_batch)
    opt_state = opt_init(lagr_params)

    # run first step but ignore updates to force the jit to compile
    step(0, opt_state, data=placeholder_batch)

    all_params = []
    all_times = []
    all_train_loss = []
    all_lagrangian = []

    for i in range(num_train_iterations):
        old_params = get_lagr_params(opt_state)
        all_params.append(old_params)

        # wait for the async dispatch to finish
        tree_util.tree_map(safe_block_until_ready, all_params)
        all_times.append(time.perf_counter())

        opt_state, logs = step(i, opt_state, data=next(batch_gen))
        # print(logs)
        all_train_loss.append(logs["train loss"])
        all_lagrangian.append(logs["lagrangian"])

        if convergence_test(get_lagr_params(opt_state), old_params):
            # print("CONVERGED!! Step:", i)
            break

    all_params.append(get_lagr_params(opt_state))
    # wait for the async dispatch to finish
    tree_util.tree_map(safe_block_until_ready, all_params)
    all_times.append(time.perf_counter())

    opt_costs = np.einsum("ij,ki,kj->k", opt_pmat, test_demos.xs - x_goal,
                          test_demos.xs - x_goal)

    @jax.jit
    def evaluate_params(params):
        pmat, learned_lqr = get_lqr(get_params(params))
        kmat = discrete.gain_matrix(pmat, learned_lqr)

        _, _, cs = evaluate_lqr_policy(test_demos.xs, x_goal, u_goal, kmat,
                                       true_lqr, num_eval_steps)
        costs = np.sum(cs, axis=-1)
        diff = costs - opt_costs

        test_loss = batch_loss(get_params(params), test_demos)
        return np.mean(diff), test_loss

    avg_diff, test_loss = zip(*[evaluate_params(p) for p in all_params])

    return all_times, avg_diff, test_loss, all_train_loss, all_lagrangian


def plot(all_times, avg_diff, test_loss, train_loss, lagr, title):
    plt.figure()
    plt.plot(np.arange(len(avg_diff)), avg_diff)
    plt.ylabel("Mean cost-to-go difference")
    plt.xlabel("# iterations")
    tikzplotlib.save(title + "-cost-iter.tex")

    plt.figure()
    plt.plot(np.array(all_times) - all_times[0], avg_diff)
    plt.ylabel("Mean cost-to-go difference")
    plt.xlabel("Relative walltime")
    tikzplotlib.save(title + "-cost-time.tex")

    plt.figure()
    plt.plot(np.arange(len(avg_diff)), test_loss)
    plt.ylabel("Test loss")
    plt.xlabel("# iterations")
    tikzplotlib.save(title + "-loss-iter.tex")

    plt.figure()
    plt.plot(np.array(all_times) - all_times[0], test_loss)
    plt.ylabel("Test loss")
    plt.xlabel("Relative walltime")
    tikzplotlib.save(title + "-loss-time.tex")

    lagr_plus_loss = np.array(lagr) + np.array(train_loss)
    plt.figure()
    plt.plot(np.arange(1, len(avg_diff)), train_loss, label="Train loss")
    plt.plot(np.arange(1, len(avg_diff)), lagr_plus_loss,
             label="Train Lagrangian + loss")
    plt.xlabel("# iterations")
    plt.legend()
    tikzplotlib.save(title + "-train-lagrangian-iter.tex")


def run_and_export(run_input):
    seed, numpy_seed, name = run_input

    # run pendulum
    # pendulum_params = pendulum.PendulumParams(length=1, mass=1, g=-9.8,
    #                                           drag=0.1)
    # pendulum_dynamics = pendulum.pendulum_dynamics(pendulum_params)
    #
    # def pendulum_sampler(rng, num_samples):
    #     key1, key2 = random.split(rng, 2)
    #     return np.stack((random.uniform(key1, (num_samples,),
    #                                     minval=np.pi - np.pi / 6,
    #                                     maxval=np.pi + np.pi / 6),
    #                      random.uniform(key2, (num_samples,),
    #                                     minval=-0.5,
    #                                     maxval=0.5)),
    #                     axis=-1)
    #
    # n = 2
    # m = 1
    # x_goal = np.array([np.pi, 0.])
    # u_goal = np.zeros((m,))
    # qmat = np.array([[2, 0.],
    #                  [0., 1]])
    # rmat = np.ones((1, 1))
    #
    # all_times, avg_diff, test_loss, train_loss, lagr = run_experiment(
    #     lr_cost=0.1,
    #     lr_constraints=0.5,
    #     rtol=1e-3,
    #     atol=1e-6,
    #     seed=0,
    #     numpy_seed=42,
    #     n=n,
    #     m=m,
    #     batch_size=2,
    #     num_train_iterations=2000,
    #     num_train_samples=10,
    #     num_test_samples=500,
    #     num_eval_steps=2000,
    #     dynamics=pendulum_dynamics,
    #     x_goal=x_goal,
    #     u_goal=u_goal,
    #     qmat=qmat,
    #     rmat=rmat,
    #     dt=1.,
    #     state_sampler=pendulum_sampler,
    # )
    # print("final avg diff:", avg_diff[-1])
    # print("final test loss:", test_loss[-1])
    #
    # plot(all_times, avg_diff, test_loss, train_loss, lagr, "pendulum")
    # plt.show()

    # run cartpole
    cartpole_params = cartpole.CartPoleParams(length=1, cart_mass=1,
                                              pole_mass=1., g=-9.8)
    cartpole_dynamics = cartpole.cartpole_dynamics(cartpole_params)

    def cartpole_sampler(rng, num_samples):
        key1, key2, key3, key4 = random.split(rng, 4)
        return np.stack((random.uniform(key1, (num_samples,),
                                        minval=np.pi - np.pi / 6,
                                        maxval=np.pi + np.pi / 6),
                         random.uniform(key2, (num_samples,),
                                        minval=-0.2,
                                        maxval=0.2),
                         random.uniform(key3, (num_samples,),
                                        minval=-0.2,
                                        maxval=0.2),
                         random.uniform(key4, (num_samples,),
                                        minval=-0.5,
                                        maxval=0.5)),
                        axis=-1)

    n = 4
    m = 1
    x_goal = np.array([np.pi, 0., 0., 0.])
    u_goal = np.zeros((m,))

    qmat = np.array([[2., 0., 0., 0.],
                     [0., 1., 0., 0.],
                     [0., 0., 1., 0.],
                     [0., 0., 0., 1.]])
    rmat = np.ones((1, 1))

    all_times, avg_diff, test_loss, train_loss, lagr = run_experiment(
        lr_cost=0.005,
        lr_constraints=0.5,
        rtol=1e-3,
        atol=1e-4,
        seed=seed,
        numpy_seed=numpy_seed,
        n=n,
        m=m,
        batch_size=50,
        num_train_iterations=2000,
        num_train_samples=1000,
        num_test_samples=500,
        num_eval_steps=2000,
        dynamics=cartpole_dynamics,
        x_goal=x_goal,
        u_goal=u_goal,
        qmat=qmat,
        rmat=rmat,
        dt=1.,
        state_sampler=cartpole_sampler,
    )

    filename = name + "-{}-{}.pkl".format(seed, numpy_seed)
    with open(filename, "wb") as f:
        pickle.dump((all_times, avg_diff, test_loss, train_loss, lagr), f)

    # print("final avg diff:", avg_diff[-1])
    # print("final test loss:", test_loss[-1])

    # plot(all_times, avg_diff, test_loss, train_loss, lagr, "cartpole")
    # plt.show()


def main():
    num_seeds = 10
    seeds = onp.random.randint(0, 2**32 - 1, num_seeds)
    np_seeds = onp.random.randint(0, 2**32 - 1, num_seeds)

    with multiprocessing.Pool() as pool:
        async_results = pool.imap_unordered(
            run_and_export, zip(seeds, np_seeds,
                                itertools.repeat("cartpole-run")))
        for i, _ in enumerate(async_results):
            print("{}/{}".format(i, num_seeds))


if __name__ == "__main__":
    main()
