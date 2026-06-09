import numpy as np


def house(x):
    """
    Householder 变换
    输入向量 x，返回 Householder 向量 v（满足 v[0]=1）和系数 beta，
    使得 (I - beta * v * v^T) * x = ||x||_2 * e_1

    采用缩放策略避免溢出：先令 x = x / max(|x|)，再计算 alpha 与 beta。
    """
    n = len(x)
    v = np.zeros(n)
    eta = np.max(np.abs(x))
    if eta == 0:
        return v, 0.0
    x = x / eta
    sigma = x[1:].T @ x[1:]
    v[1:] = x[1:]
    if sigma == 0:
        beta = 0.0
    else:
        alpha = np.sqrt(x[0] ** 2 + sigma)
        if x[0] <= 0:
            v[0] = x[0] - alpha
        else:
            v[0] = -sigma / (x[0] + alpha)
        beta = 2 * (v[0] ** 2) / (sigma + v[0] ** 2)
        v /= v[0]
    return v, beta
