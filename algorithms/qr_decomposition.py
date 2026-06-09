import numpy as np

def house(x):
    n = len(x)
    v = np.zeros(n)

    eta = np.max(np.abs(x))
    x = x / eta
    sigma = x[1:].T @ x[1:]
    v[1:] = x[1:]

    if sigma == 0:
        beta = 0.0
    else:
        alpha = np.sqrt(x[0]**2 + sigma)
        if x[0] <= 0:
            v[0] = x[0] - alpha
        else:
            v[0] = -sigma / (x[0] + alpha)
        beta = 2 * (v[0]**2) / (sigma + v[0]**2)
        v /= v[0]
    return v, beta

def qr_decomposition(A):
    A = np.array(A, dtype=np.float64, copy=True)

    m, n = A.shape
    for j in range(0, n):
        if j+1 < m:
            v, beta = house(A[j:, j])
            A[j:, j:] -= beta * v[:, None] * (v @ A[j:, j:])
            A[j+1:, j] = v[1:m-j+1]
    return A