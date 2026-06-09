import numpy as np
from algorithms.direct_solver import gauss_partial_pivoting

def norm1_estimation(B, x):
    '''
    Given B n*n, estimate ||B||_1
    '''
    n = len(x)
    omega = B @ x
    v = np.sign(omega)
    z = B.T @ v

    if np.max(np.abs(z)) <= z.T @ x:
        return np.sum(np.abs(omega))
    else: 
        j = np.argmax(np.abs(z))
        e = np.zeros(n)
        e[j] = 1
        return norm1_estimation(B, e)

def norm1_estimation_inv(H, x):
    '''
    Given H, estimate ||H^-1||_1
    '''
    n = len(x)
    omega = gauss_partial_pivoting(H.T, x)
    v = np.sign(omega)
    z = gauss_partial_pivoting(H, v)

    if np.max(np.abs(z)) <= z.T @ x:
        return np.sum(np.abs(omega))
    else: 
        j = np.argmax(np.abs(z))
        e = np.zeros(n)
        e[j] = 1
        return norm1_estimation_inv(H, e)
