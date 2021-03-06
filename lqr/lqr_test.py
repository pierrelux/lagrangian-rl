from functools import partial

from absl.testing import absltest

import lqr
from lqr import cartpole
from lqr import dynamical
from lqr import pendulum

import scipy
import numpy as onp

import jax
import jax.numpy as np
import jax.test_util

from jax.config import config
config.update("jax_enable_x64", True)


def _pendulum_problem():
    params = pendulum.PendulumParams(length=1, mass=1, g=-9.8, drag=0.)
    dynamics = pendulum.pendulum_dynamics(params)

    x_goal = np.array([np.pi, 0.])
    qmat = np.array([[2, 0.],
                     [0., 1]])
    rmat = np.ones((1, 1))
    return dynamics, x_goal, qmat, rmat


def _cartpole_problem():
    params = cartpole.CartPoleParams(length=1, cart_mass=1, pole_mass=1.,
                                     g=-9.8)
    dynamics = cartpole.cartpole_dynamics(params)

    x_goal = np.array([np.pi, 0., 0., 0.])
    qmat = np.array([[2., 0., 0., 0.],
                     [0., 1., 0., 0.],
                     [0., 0., 1., 0.],
                     [0., 0., 0., 1.]])
    rmat = np.ones((1, 1))
    return dynamics, x_goal, qmat, rmat


def _integrate(dynamics, policy, x_init, duration=10.):
    final_dynamics = dynamical.controlled_dynamics(policy, dynamics)
    final_dynamics = jax.jit(final_dynamics)

    return scipy.integrate.odeint(
        final_dynamics,
        x_init,
        [0., duration],
    )


class LQRTest(jax.test_util.JaxTestCase):

    def testContinuousInfiniteHorizonPendulum(self):
        offsets = [onp.array([0., 0.]),
                   onp.array([-0.3, 0.]),
                   onp.array([-0.6, 0.2]),
                   onp.array([0.2, 0.2])]
        dynamics, x_goal, qmat, rmat = _pendulum_problem()

        kmat = lqr.continuous.infinite_lqr(dynamics, x_goal, np.zeros((1,)), 0,
                                           qmat, rmat)
        policy = lqr.policy(kmat, x_goal)

        for x_offset in offsets:
            x_init = x_goal + np.array(x_offset)
            xs = _integrate(dynamics, policy, x_init)
            self.assertAllClose(x_goal, xs[-1], check_dtypes=True)

    def testContinuousInfiniteHorizonPendulum(self):
        offsets = [onp.array([0., 0., 0., 0.]),
                   onp.array([-0.3, 0., 0., 0.]),
                   onp.array([-0.6, 0., 0.2, 0.]),
                   onp.array([0.2, 0., 0.2, 0.])]
        dynamics, x_goal, qmat, rmat = _cartpole_problem()

        kmat = lqr.continuous.infinite_lqr(dynamics, x_goal, np.zeros((1,)), 0,
                                           qmat, rmat)
        policy = lqr.policy(kmat, x_goal)

        for x_offset in offsets:
            x_init = x_goal + np.array(x_offset)
            xs = _integrate(dynamics, policy, x_init, duration=20.)
            self.assertAllClose(x_goal, xs[-1], check_dtypes=True)

    def testDiscreteFiniteHorizonPendulum(self):
        offsets = [onp.array([0., 0.]),
                   onp.array([-0.3, 0.]),
                   onp.array([-0.6, 0.2]),
                   onp.array([0.2, 0.2])]
        dynamics, x_goal, qmat, rmat = _pendulum_problem()

        horizon = 100
        dt = 10./horizon
        amat, bmat = dynamical.linearize_dynamics(dynamics, x_goal,
                                                  np.zeros((1,)), 0)
        amat = amat * dt
        bmat = bmat * dt

        kmat, kvec = lqr.discrete.finite_horizon_lqr(amat, bmat, x_goal,
                                                     np.zeros((1,)),
                                                     qmat, rmat, horizon)
        policy = lqr.policy(kmat, x_goal, kvec)

        @jax.jit
        def linear_dynamics(x, u, t):
            del t
            return x_goal + amat @ (x - x_goal) + bmat @ u

        for x_offset in offsets:
            x_init = x_goal + np.array(x_offset)
            x = x_init

            for t in range(horizon):
                x = linear_dynamics(x, policy(x, t), t)

            self.assertAllClose(x_goal, x, check_dtypes=True)

            def _solved_policy(A, B, Q, R):
                kmat, kvec = lqr.discrete.finite_horizon_lqr(A, B, x_goal,
                                                             np.zeros((1,)),
                                                             Q, R,
                                                             horizon)
                policy = lqr.policy(kmat, x_goal, kvec)
                return policy(x_init, 0)

            jax.test_util.check_vjp(_solved_policy,
                                    partial(jax.vjp, _solved_policy),
                                    (amat, bmat, qmat, rmat))


if __name__ == "__main__":
    absltest.main()
