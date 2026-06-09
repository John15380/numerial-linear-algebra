import numpy as np


def create_tridiagonal_matrix(n, a, b):
    """
    构造 n 阶对称三对角矩阵，主对角线元素为 a，次对角线元素为 b
    """
    A = np.zeros((n, n), dtype=np.float64)
    np.fill_diagonal(A, a)
    np.fill_diagonal(A[1:, :], b)
    np.fill_diagonal(A[:, 1:], b)
    return A


def companion_matrix(coeffs):
    """
    构造首一多项式的友矩阵 (companion matrix)

    对于多项式:
        p(x) = x^n + a_{n-1} x^{n-1} + ... + a_1 x + a_0

    输入 coeffs = [a_{n-1}, a_{n-2}, ..., a_1, a_0]
    返回 n×n 友矩阵 C
    """
    n = len(coeffs)
    if n == 0:
        return np.zeros((0, 0))

    C = np.zeros((n, n), dtype=np.float64)
    # 友矩阵：最后一列为 [-a_0, -a_1, ..., -a_{n-1}]^T，次对角线为 1
    for i in range(n):
        C[i, -1] = -coeffs[n - 1 - i]
    np.fill_diagonal(C[1:, :], 1.0)  # 次对角线
    return C
