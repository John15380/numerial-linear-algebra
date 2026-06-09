import numpy as np

def calculate_optimal_omega(A):
    """
    计算 SOR 方法的最佳松弛因子 omega_opt
    公式: 2 / (1 + sqrt(1 - rho(T_J)^2))
    """
    D = np.diag(np.diag(A))
    L_plus_U = A - D
    # Jacobi 迭代矩阵 T_J = -D^-1 * (L + U)
    D_inv = np.diag(1.0 / np.diag(A))
    T_J = -D_inv @ L_plus_U
    
    # 计算特征值并取绝对值的最大值（谱半径）
    eigenvalues = np.linalg.eigvals(T_J)
    rho_TJ = max(abs(eigenvalues))
    
    # 如果谱半径 >= 1，SOR 可能不收敛，通常设为 1
    if rho_TJ >= 1:
        return 1.0
        
    omega_opt = 2 / (1 + np.sqrt(1 - rho_TJ**2))
    return omega_opt

def jacobi_iteration(A, b, tol=5e-5, max_iter=50000):
    """
    Jacobi 迭代法求解线性方程组
    """
    D = np.diag(np.diag(A))
    L = np.tril(A, -1)
    U = np.triu(A, 1)
    
    # 转换为迭代格式 x = Tx + c 提高计算速度
    D_inv = np.diag(1.0 / np.diag(A))
    T = -D_inv @ (L + U)
    C = D_inv @ b
    
    x = np.zeros_like(b)
    for k in range(max_iter):
        x_new = T @ x + C
        # 满足 4 位有效数字要求
        if np.max(np.abs(x_new - x)) < tol:
            return x_new, k + 1
        x = x_new
    return x, max_iter

def gauss_seidel_iteration(A, b, tol=5e-5, max_iter=50000):
    """
    Gauss-Seidel 迭代法求解线性方程组
    """
    D = np.diag(np.diag(A))
    L = np.tril(A, -1)
    U = np.triu(A, 1)
    
    DL_inv = np.linalg.inv(D + L)
    T = -DL_inv @ U
    C = DL_inv @ b
    
    x = np.zeros_like(b)
    for k in range(max_iter):
        x_new = T @ x + C
        if np.max(np.abs(x_new - x)) < tol:
            return x_new, k + 1
        x = x_new
    return x, max_iter

def sor_iteration(A, b, omega=1.2, tol=5e-5, max_iter=50000):
    """
    SOR (逐次超松弛) 迭代法求解线性方程组
    """
    D = np.diag(np.diag(A))
    L = np.tril(A, -1)
    U = np.triu(A, 1)
    
    M_inv = np.linalg.inv(D + omega * L)
    T = M_inv @ ((1 - omega) * D - omega * U)
    C = M_inv @ (omega * b)
    
    x = np.zeros_like(b)
    for k in range(max_iter):
        x_new = T @ x + C
        if np.max(np.abs(x_new - x)) < tol:
            return x_new, k + 1
        x = x_new
    return x, max_iter
