from functools import partial

import jax
from jax import random
import jax.numpy as np

from fax import converge
from fax import lagrangian
from fax.lagrangian import cga

from lqr import discrete
from lqr import initializers


def main():
    n = 2
    m = 1
    seed = 0
    lr = 0.5
    horizon = 100
    rtol = 1e-3
    atol = 1e-5

    x_goal = np.array([np.pi, 0.])
    u_goal = np.zeros((m,))

    params_init = initializers.lqr_traj(
        lqr_initializer=initializers.lqr(),
        xs_initializer=partial(random.uniform, minval=-1e2, maxval=1e-2),
        us_initializer=partial(random.uniform, minval=-1e2, maxval=1e-2),
    )
    init_param, kkt, get_lqr_traj = initializers.finite_horizon_kkt(params_init)
    lagrangian.make_lagrangian()

    opt_init, opt_update, get_params = cga.cga_lagrange_min(lr, lagrangian)


    def convergence_test(x_new, x_old):
        return converge.max_diff_test(x_new, x_old, rtol, atol)

    @jax.jit
    def step(i, opt_state):
        params = get_params(opt_state)
        grads = jax.grad(lagrangian, (0, 1))(*params)
        return opt_update(i, grads, opt_state)

    rng = random.PRNGKey(seed)
    rng, params_key = random.split(rng)

    params = params_init(params_key, (horizon, n, m))
    lagr_params = init_mult(params)
    opt_state = opt_init(lagr_params)

    for i in range(500):
        old_params = get_params(opt_state)
        opt_state = step(i, opt_state)

        if convergence_test(get_params(opt_state), old_params):
            break
