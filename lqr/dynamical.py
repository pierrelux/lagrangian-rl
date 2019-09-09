import jax


def linearize_dynamics(dynamics, x, u, t):
    return jax.jacfwd(dynamics, argnums=(0, 1))(x, u, t)


def controlled_dynamics(policy, dynamics):
    def controlled_dxdt(x, t):
        u = policy(x, t)
        dxdt = dynamics(x, u, t)
        return dxdt

    return controlled_dxdt
