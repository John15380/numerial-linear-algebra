import numpy as np


def power_method_polynomial(coeffs, tol=1e-10, max_iter=10000):
    """
    幂法求多项式方程的模最大根

    对于首一多项式:
        f(x) = x^n + a_{n-1} x^{n-1} + ... + a_1 x + a_0 = 0

    输入 coeffs = [a_{n-1}, a_{n-2}, ..., a_1, a_0]
    （从最高次非首项系数到常数项）

    构造友矩阵 (companion matrix) C:
        [  0    0   ...   0   -a_0   ]
        [  1    0   ...   0   -a_1   ]
        [  0    1   ...   0   -a_2   ]
        [  :    :   .     :    :     ]
        [  0    0   ...   1   -a_{n-1}]

    对 C 应用幂法求模最大特征值，即模最大根。

    返回:
        root: 模最大根
        iters: 迭代次数
        converged: 是否收敛
    """
    n = len(coeffs)
    if n == 0:
        raise ValueError("多项式次数至少为 1")

    # 构造友矩阵
    C = np.zeros((n, n), dtype=float)
    C[0, -1] = -coeffs[-1] if n >= 1 else 0.0  # -a_0
    for i in range(1, n):
        C[i, i - 1] = 1.0
        C[i, -1] = -coeffs[n - 1 - i]

    # 幂法
    v = np.random.rand(n)
    v = v / np.linalg.norm(v, 2)

    lambda_old = 0.0
    for k in range(max_iter):
        w = C @ v
        norm_w = np.linalg.norm(w, 2)
        if norm_w == 0:
            return 0.0, k + 1, False
        v = w / norm_w

        # Rayleigh 商作为特征值近似
        lambda_new = v.T @ (C @ v)

        if abs(lambda_new - lambda_old) < tol:
            return lambda_new, k + 1, True
        lambda_old = lambda_new

    lambda_final = v.T @ (C @ v)
    return lambda_final, max_iter, False


def power_method_polynomial_direct(coeffs, tol=1e-10, max_iter=10000):
    """
    幂法求多项式方程的模最大根（直接迭代，避免显式构造矩阵）

    利用友矩阵的结构，直接对向量进行迭代：
    设 y = (y_0, y_1, ..., y_{n-1})^T
    则 C*y 的分量为:
        z_0 = -a_0 * y_{n-1}
        z_1 = y_0 - a_1 * y_{n-1}
        ...
        z_{n-1} = y_{n-2} - a_{n-1} * y_{n-1}

    输入 coeffs = [a_{n-1}, a_{n-2}, ..., a_1, a_0]
    返回: (root, iters, converged)
    """
    n = len(coeffs)
    if n == 0:
        raise ValueError("多项式次数至少为 1")

    # coeffs 顺序: a_{n-1}, a_{n-2}, ..., a_1, a_0
    # 方便访问: coeffs[-1] = a_0, coeffs[-2] = a_1, ..., coeffs[0] = a_{n-1}
    a = coeffs  # alias

    # 随机初始向量
    y = np.random.rand(n)
    y = y / np.linalg.norm(y, 2)

    lambda_old = 0.0
    for k in range(max_iter):
        # 计算 z = C * y
        z = np.empty(n, dtype=float)
        z[0] = -a[-1] * y[-1]           # -a_0 * y_{n-1}
        for i in range(1, n):
            z[i] = y[i - 1] - a[n - 1 - i] * y[-1]  # y_{i-1} - a_i * y_{n-1}

        norm_z = np.linalg.norm(z, 2)
        if norm_z == 0:
            return 0.0, k + 1, False
        y = z / norm_z

        # Rayleigh 商: lambda = y^T C y = y^T z_norm = y^T (C y)
        # 先计算 C y（未归一化）用于 Rayleigh 商
        cy = np.empty(n, dtype=float)
        cy[0] = -a[-1] * y[-1]
        for i in range(1, n):
            cy[i] = y[i - 1] - a[n - 1 - i] * y[-1]
        lambda_new = np.dot(y, cy)

        if abs(lambda_new - lambda_old) < tol:
            return lambda_new, k + 1, True
        lambda_old = lambda_new

    # 最终估计
    cy = np.empty(n, dtype=float)
    cy[0] = -a[-1] * y[-1]
    for i in range(1, n):
        cy[i] = y[i - 1] - a[n - 1 - i] * y[-1]
    lambda_final = np.dot(y, cy)
    return lambda_final, max_iter, False
