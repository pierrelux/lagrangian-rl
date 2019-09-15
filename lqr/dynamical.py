import jax
import jax.numpy as np


def linearize_dynamics(dynamics, x, u, t):
    return jax.jacfwd(dynamics, argnums=(0, 1))(x, u, t)


def controlled_dynamics(policy, dynamics):
    def controlled_dxdt(x, t):
        u = policy(x, t)
        dxdt = dynamics(x, u, t)
        return dxdt

    return controlled_dxdt


def dynmaics_from_manipulator(h, c, g, bmat):
    def dynamics(x, u, t):
        q, q_dot = np.split(x, 2)
        q_dot_dot = np.linalg.solve(h(q, t),
                                    bmat @ u - c(q, q_dot, t) @ q_dot - g(q, t))
        return np.concatenate((q_dot, q_dot_dot), axis=0)

    return dynamics
