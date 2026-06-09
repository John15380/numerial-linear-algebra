import numpy as np

def conjugate_gradient(A, b, tol=1e-10, max_iter=None):
    """
    共轭梯度法 (CG) 求解对称正定线性方程组 Ax = b
    停止准则: ||r_k||_2 < tol
    """
    n = len(b)
    if max_iter is None:
        max_iter = n

    x = np.zeros(n, dtype=float)
    r = b.astype(float) - A @ x   # 初始残差 r_0 = b - A x_0
    p = r.copy()
    rr = np.dot(r, r)

    for k in range(max_iter):
        Ap = A @ p
        alpha = rr / np.dot(p, Ap)   # 最优步长
        x = x + alpha * p
        r = r - alpha * Ap
        rr_new = np.dot(r, r)

        if np.sqrt(rr_new) < tol:
            return x, k + 1

        beta = rr_new / rr            # 共轭方向更新系数
        p = r + beta * p
        rr = rr_new

    return x, max_iter
