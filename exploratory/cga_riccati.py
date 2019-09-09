# This version uses two Lagrange multipliers that were meant to pick up the 
# singular values of the residual. It doesn't seem to work and I'm only 
# keeping it in case one wants to revisit it.

import jax
import jax.numpy as np
import jax.random as jrandom
import random
from random import randint
from jax import grad, jacobian
from jax.flatten_util import ravel_pytree
from jax import random
from fax.lagrangian import cga
from fax.lagrangian import cg
from scipy.linalg import solve_discrete_are

key = random.PRNGKey(randint(1,10000))


linear_op_solver = cg.fixed_point_solve

def Dxy(fun):
    gradfun = grad(fun, argnum=0)
    return jacobian(gradfun, argnum=1)
def Dyx(fun):
    gradfun = grad(fun, argnum=1)
    return jacobian(gradfun, argnum=0)


if __name__ == "__main__":

    # Choosing matrices for Riccati equation
    # A = np.array([[1, 1], [0, 1]])
    # B = np.array([[0], [1]])
    # Q = np.array([[1,0],[0,0]])
    # R = np.array([[1]])
    A = jrandom.normal(key, [10, 10])
    B = jrandom.normal(key, [10, 5])
    Q = jrandom.normal(key, [10, 10])
    Q = Q @ Q.transpose()
    R = jrandom.normal(key, [5, 5])
    R = R @ R.transpose()


    At = A.transpose()
    Bt = B.transpose()

    def Lagrangian(K, Z, mu0, mu1, nu):
      return np.sum( 
            mu0 @ (At @ K @ A @ nu + Q @ nu - K @ nu + At @ K @ B @ Z @ nu) +
            mu1 @ (Bt @ K @ A @ nu + Bt @ K @ B @ Z @ nu + R @ Z @ nu) 
            )

    K0 = jrandom.normal(key, [A.shape[0], A.shape[0]])
    Z0 = jrandom.normal(key, [B.shape[1], A.shape[1]])
    mu00 = jrandom.normal(key, [3, A.shape[1]])
    mu10 = jrandom.normal(key, [3, B.shape[1]])
    nu0 = jrandom.normal(key, [A.shape[1], 3])

    mu0 = (At @ K0 @ A @ nu0 + Q @ nu0 - K0 @ nu0 + At @ K0 @ B @ Z0 @ nu0).transpose()
    mu1 = (Bt @ K0 @ A @ nu0 + Bt @ K0 @ B @ Z0 @ nu0 + R @ Z0 @ nu0).transpose()
    nu = ((mu00 @ (At @ K0 @ A + Q - K0 + At @ K0 @ B @ Z0)).transpose() + 
         (mu1 @ (Bt @ K0 @ A + Bt @ K0 @ B @ Z0 + R @ Z0)).transpose())


    x, unflatten_x = ravel_pytree([K0, Z0])
    y, unflatten_y = ravel_pytree([mu00, mu10, nu0])


    def f(x, y):
        K, Z = unflatten_x(x)
        mu0, mu1, nu = unflatten_y(y)
        return Lagrangian(K, Z, mu0, mu1, nu)

    def g(x, y):
        return -f(x, y)

    eta = 0.50
    num_iter = 500

    cga_init, cga_update, get_params = cga.full_solve_cga(
                step_size_f=eta,
                step_size_g=eta,
                f=f,
                g=g,
                # linear_op_solver=linear_op_solver,
                # linear_op_solver=None,
            )

    grad_yg = jax.grad(g, 1)
    grad_xf = jax.grad(f, 0)

    # print(grad_yg)

    # print(x)
    # print(y)

    @jax.jit
    def step(i, opt_state):
        x, y = get_params(opt_state)[:2]
        grads = (grad_xf(x, y), grad_yg(x, y))
        return cga_update(i, grads, opt_state)

    opt_state = cga_init((x,y))
    for i in range(num_iter):
        opt_state = step(i, opt_state)

    final_values = get_params(opt_state)[:2]
    x = final_values[0]
    K, Z = unflatten_x(x)
    y = final_values[1]
    mu00, mu10, nu = unflatten_y(y)

    print(unflatten_x(x)[0])

    trueSol = solve_discrete_are(A, B, Q, R)
    print(trueSol)

    print(y)
    

    # print(final_values)
    print(grad_xf(x,y))
    print(grad_yg(x,y))

    print(At @ K @ A - K - (At @ K @ B) @ np.linalg.inv(R + Bt @ K @ B) @ (Bt @ K @ A) + Q )


    