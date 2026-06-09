import numpy as np

def create_tridiagonal(n, main_val, upper_val, lower_val):

    main_diag = np.full(n, main_val)
    upper_diag = np.full(n-1, upper_val)
    lower_diag = np.full(n-1, lower_val)
    
    # 叠加矩阵
    A = (np.diag(main_diag, k=0) +
         np.diag(upper_diag, k=1) +
         np.diag(lower_diag, k=-1))

    x_true = np.random.uniform(-10, 10, n)
    
    # 根据 Ax = b 算出对应的右端项 b
    b = A @ x_true
    
    return A, b, x_true

def create_hilbert(n):
    # 1. 构造 Hilbert 矩阵 A: A[i,j] = 1/(i+j+1) (下标从0开始)
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            A[i, j] = 1.0 / (i + j + 1)
    
    # 2. 设定真解为全 1
    x_true = np.ones(n)
    
    # 3. 计算对应的 b
    b = A @ x_true
    
    return A, b, x_true

def create_an(n):
    # 1. 初始化一个全为 0 的矩阵
    matrix = np.zeros((n, n), dtype=np.float64)
    
    # 2. 设置下三角部分为 -1 (k=-1 表示对角线下方第一条开始)
    # np.tril_indices 返回下三角的索引
    lower_indices = np.tril_indices(n, k=-1)
    matrix[lower_indices] = -1
    
    # 3. 设置主对角线为 1
    np.fill_diagonal(matrix, 1)
    
    # 4. 设置最后一列为 1
    matrix[:, -1] = 1
    
    return matrix