import collections

import jax.numpy as np

PendulumParams = collections.namedtuple(
  "PendulumParams", "length mass g"
)


def pendulum_dynamics(params):
    def dxdt(x, u, t):
        del t
        w = np.product(params)
        # assume point mass and massless arm
        inertia = params.mass * params.length ** 2
        return np.stack((x[1], (w*np.sin(x[0]) + u[0])/inertia))

    return dxdt
