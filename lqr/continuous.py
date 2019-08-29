import dynamical

import scipy
import numpy as np


def infinite_lqr(dynamics, x_0, u_0, t, qmat, rmat):
    amat, bmat = dynamical.linearize_dynamics(dynamics, x_0, u_0, t)
    pmat = scipy.linalg.solve_continuous_are(amat, bmat, qmat, rmat)
    return scipy.linalg.solve(rmat, np.dot(bmat.T, pmat), sym_pos=True)
