# 输入的矩阵必须非奇异
import numpy as np

def forward_substitution(L, b):
    """
    下三角形方程组求解: Ly = b（前代法）
    """
    n = len(b)
    y = np.zeros(n) # 初始化y

    for i in range(n):
        y[i] = (b[i] - L[i, :i] @ y[:i]) / L[i, i]
    return y

def back_substitution(U, y):
    """
    上三角形方程组求解: Ux = y（回代法）
    """
    n = len(y)
    x = np.zeros(n) # 初始化x

    for i in range(n-1, -1, -1):
        x[i] = (y[i] - U[i, i+1:] @ x[i+1:]) / U[i, i]
    return x

def gauss_no_pivoting(A, b):
    """
    不选主元的Gauss消元法
    """
    n = len(b)
    A = np.array(A, dtype=np.float64, copy=True)
    b = np.array(b, dtype=np.float64, copy=True)

    # 将A化为U
    for k in range(n-1):
        m = (A[k+1:, k] / A[k, k])
        A[k+1:, k+1:] -= m[:, np.newaxis] * A[k, k+1:]
        b[k+1:] -= m * b[k]
        A[k+1:, k] = 0

    return back_substitution(A, b)

def gauss_partial_pivoting(A, b):
    """
    列主元Gauss消元法
    """
    n = len(b)
    A = np.array(A, dtype=np.float64, copy=True)
    b = np.array(b, dtype=np.float64, copy=True)

    for k in range(n-1):
        # 寻找列主元
        max_idx = np.argmax(np.abs(A[k:, k])) + k
        A[[k, max_idx], :] = A[[max_idx, k], :]
        b[[k, max_idx]] = b[[max_idx, k]]

        # 将A化为U
        m = (A[k+1:, k] / A[k, k])
        A[k+1:, k+1:] -= m[:, np.newaxis] * A[k, k+1:]
        b[k+1:] -= m * b[k]
        A[k+1:, k] = 0

    return back_substitution(A, b)

def gauss_complete_pivoting(A, b):
    """
    全主元Gauss消元法
    """
    n = len(b)
    col_indices = np.arange(n)
    A = np.array(A, dtype=np.float64, copy=True)
    b = np.array(b, dtype=np.float64, copy=True)

    for k in range(n-1):
        # 寻找全主元
        flat_idx = np.argmax(np.abs(A[k:, k:]))
        rel_idx, rel_idy = np.unravel_index(flat_idx, A[k:, k:].shape)

        max_idx = rel_idx + k
        max_idy = rel_idy + k

        # 行交换
        A[[k, max_idx], :] = A[[max_idx, k], :]
        b[[k, max_idx]] = b[[max_idx, k]]

        # 列交换
        A[:, [k, max_idy]] = A[:, [max_idy, k]]
        col_indices[[k, max_idy]] = col_indices[[max_idy, k]] # 记录哪两列换位

        # 将A化为U
        m = (A[k+1:, k] / A[k, k])
        A[k+1:, k+1:] -= m[:, np.newaxis] * A[k, k+1:]
        b[k+1:] -= m * b[k]
        A[k+1:, k] = 0

    x = np.zeros(n)
    x[col_indices] = back_substitution(A, b) # 还原解的顺序
    return x

def cholesky_decomposition(A, b):

    n = len(b)
    A = np.array(A, dtype=np.float64, copy=True)
    b = np.array(b, dtype=np.float64, copy=True)
    x = np.zeros(n)
    L = np.zeros((n, n))

    for k in range(n):
        L[k, k] = np.sqrt(A[k, k] - L[k, :k] @ L[k, :k])
        L[k+1:n, k] = (A[k+1:n, k] - L[k+1:n, :k] @ L[k, :k]) / L[k, k] 
        
    y = forward_substitution(L, b)
    x = back_substitution(L.T, y)
    return x

def ldlt_decomposition(A, b):
    n = len(b)
    A = np.array(A, dtype=np.float64, copy=True)
    b = np.array(b, dtype=np.float64, copy=True)
    x = np.zeros(n)
    d = np.zeros(n)
    L = np.eye(n)

    for k in range(n):
        d[k] = A[k, k] - (L[k, :k]**2) @ d[:k]  
        if k < n - 1:
            L[k+1:n, k] = (A[k+1:n, k] - L[k+1:n, :k] @ (L[k, :k] * d[:k])) / d[k]

    y = forward_substitution(L, b)
    z = y / d
    x = back_substitution(L.T, z)
    return x