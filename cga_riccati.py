import autograd.numpy as np
from autograd import grad, jacobian
from autograd.misc import flatten


def Dxy(fun):
    gradfun = grad(fun, argnum=0)
    return jacobian(gradfun, argnum=1)

def Dyx(fun):
    gradfun = grad(fun, argnum=1)
    return jacobian(gradfun, argnum=0)


if __name__ == "__main__":
    rng = np.random.RandomState(0)

    # Choosing matrices for Riccati equation
    A = np.array([[1, 1], [0, 1]])
    At = A.transpose()
    B = np.array([[0], [1]])
    Bt = B.transpose()
    Q = np.array([[1,0],[0,0]])
    R = np.array([[1]])

    def Lagrangian(K, Z, mu, nu):
      # return np.sum(
      #         np.dot(
      #           np.dot(mu,
      #             np.block([[At @ K @ A + Q - K, 
      #                       At @ K @ B],
      #                      [Bt @ K @ A,
      #                       Bt @ K @ B + R]])),
      #           np.dot(
      #             np.block([[np.eye(A.shape[1])], 
      #                      [Z]]),
      #             nu)
      #         )
      return np.sum(
                np.dot(mu,
                  np.block([[At @ A + Q - K, 
                            At @ B],
                           [Bt @ A,
                            Bt @  B + R]])),
                            )

    K0 = np.zeros([A.shape[0], A.shape[0]])
    Z0 = np.zeros([B.shape[1], A.shape[1]])
    mu0 = np.zeros([1, A.shape[1] + B.shape[1]])
    nu0 = np.zeros([A.shape[1], 1])

    x, unflatten_x = flatten([K0, Z0])
    y, unflatten_y = flatten([mu0, nu0])


    def f(x, y):
        K, Z = unflatten_x(x)
        mu, nu = unflatten_y(y)
        return Lagrangian(K, Z, mu, nu)

    def g(x, y):
        return -f(x, y)

    gradfx = grad(f, argnum=0)
    gradgy = grad(g, argnum=1)

    Dxyf = Dxy(f)
    Dyxf = Dyx(f)
    Dxyg = Dxy(g)
    Dyxg = Dyx(g)

    eta = 0.5
    for i in range(1000):
        print(i, f(x, y), np.linalg.norm(np.concatenate((x, y))))

        # print(np.dot(Dxyf(x, y)), Dyxg(x, y)))
        print(gradfx)
        print(gradgy)
        print(Dxyf(x, y))
        print(Dyxf(x, y))
        print(Dxyg(x, y))
        print(Dyxg(x, y))
        Mx = np.linalg.inv(np.eye(x.shape[0]) - (eta**2)*np.dot(Dxyf(x, y), Dyxg(x, y)))
        My = np.linalg.inv(np.eye(y.shape[0]) - (eta**2)*np.dot(Dyxg(x, y), Dxyf(x, y)))

        deltax = -np.dot(Mx, (gradfx(x, y) - eta*np.dot(Dxyf(x, y), gradgy(x, y))))
        deltay = -np.dot(My, (gradgy(x, y) - eta*np.dot(Dyxg(x, y), gradfx(x, y))))

        x = x + eta*deltax
        y = y + eta*deltay
