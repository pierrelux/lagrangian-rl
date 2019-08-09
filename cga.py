import autograd.numpy as np
from autograd import grad, jacobian


def mixed_hessian(fun, a0=0, a1=1):
    gradfun = grad(fun, argnum=a0)
    return jacobian(gradfun, argnum=a1)


if __name__ == "__main__":
    rng = np.random.RandomState(0)
    A = rng.uniform(size=(2, 3))
    x = rng.uniform(size=(2,))
    y = rng.uniform(size=(3,))

    def f(x, y):
        return x.T @ A @ y

    def g(x, y):
        return -f(x, y)

    gradfx = grad(f, argnum=0)
    gradgy = grad(g, argnum=1)

    Dxyf = mixed_hessian(f)
    Dyxf = mixed_hessian(f, 1, 0)
    Dxyg = mixed_hessian(g)
    Dyxg = mixed_hessian(g, 1, 0)

    eta = 0.5
    for i in range(1000):
        print(i, f(x, y), np.linalg.norm(np.concatenate((x, y))))

        B = np.linalg.inv(np.eye(x.shape[0]) - (eta**2)*np.dot(Dxyf(x, y), Dyxg(x, y)))
        C = np.linalg.inv(np.eye(y.shape[0]) - (eta**2)*np.dot(Dyxg(x, y), Dxyf(x, y)))

        deltax = -np.dot(B, (gradfx(x, y) - eta*np.dot(Dxyf(x, y), gradgy(x, y))))
        deltay = -np.dot(C, (gradgy(x, y) - eta*np.dot(Dyxg(x, y), gradfx(x, y))))

        x = x + eta*deltax
        y = y + eta*deltay
