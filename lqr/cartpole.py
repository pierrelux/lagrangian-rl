import collections

import jax.numpy as np

from lqr import dynamical

CartPoleParams = collections.namedtuple(
  "CartPoleParams", "length cart_mass pole_mass g"
)


def cartpole_dynamics(params):
    m_p = params.pole_mass
    m_c = params.cart_mass
    l = params.length

    def h(q, t):
        del t
        mlcos = m_p * l * np.cos(q[0])
        return np.array([[m_c + m_p, mlcos],
                         [mlcos, m_p * l**2]])

    def c(q, q_dot, t):
        del t
        return np.array([[0, -m_p * l * q_dot[0] * np.sin(q[0])],
                         [0, 0]])

    def g(q, t):
        del t
        return np.array([0, m_p * params.g * l * np.sin(q[0])])

    bmat = np.array([[1.], [0.]])

    return dynamical.dynmaics_from_manipulator(h, c, g, bmat)
