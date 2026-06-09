import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from algorithms.direct_solver import gauss_no_pivoting, gauss_partial_pivoting, cholesky_decomposition, ldlt_decomposition
from algorithms.create_matrix import create_hilbert, create_tridiagonal

# Test 1
n = 100
A, b, x_true = create_tridiagonal(n, 10, 1, 1)
solvers = {
    "Gauss No Pivoting": gauss_no_pivoting,
    "Gauss Partial Pivoting": gauss_partial_pivoting,
    "Cholesky": cholesky_decomposition,
    "LDLT": ldlt_decomposition
}

print(f"{'Algorithm':<25} | {'Relative Error':<18} | {'Residual (||Ax-b||)':<18}")
print("-" * 75)

for name, solver in solvers.items():
    try:
        x_calc = solver(A, b)
        
        # 1. 计算相对误差: ||x_calc - x_true||_inf / ||x_true||_inf
        # 它可以衡量解与真实情况的接近程度
        error = np.linalg.norm(x_calc - x_true, np.inf) / np.linalg.norm(x_true, np.inf)
        
        # 2. 计算残差: ||A * x_calc - b||_inf
        # 它可以衡量解在代数方程上的自洽性
        residual = np.linalg.norm(A @ x_calc - b, np.inf)
        
        print(f"{name:<25} | {error:.6e}          | {residual:.6e}")
        
    except Exception as e:
        print(f"{name:<25} | Failed: {e}")

print("="*50 + '\n')

# Test 2
n = 40
A_h, b_h, x_true_h = create_hilbert(n)

print(f"\n{'Algorithm (Hilbert n=40)':<25} | {'Relative Error':<18} | {'Residual':<18}")
print("-" * 75)

solvers = {
    "Gauss No Pivoting": gauss_no_pivoting,
    "Gauss Partial Pivoting": gauss_partial_pivoting,
    "Cholesky": cholesky_decomposition,
    "LDLT": ldlt_decomposition
}

for name, solver in solvers.items():
    try:
        x_calc = solver(A_h, b_h)
        
        # 计算相对误差
        rel_err = np.linalg.norm(x_calc - x_true_h, np.inf) / np.linalg.norm(x_true_h, np.inf)
        # 计算残差
        residual = np.linalg.norm(A_h @ x_calc - b_h, np.inf)
        
        # 如果结果包含 NaN，手动抛出异常
        if np.isnan(x_calc).any():
            print(f"{name:<25} | {'NAN DETECTED':<18} | {'数值崩溃':<18}")
        else:
            print(f"{name:<25} | {rel_err:.6e}          | {residual:.6e}")
            
    except Exception as e:
        # Cholesky 在这里一定会报错，因为 Hilbert 的数值误差会导致矩阵看起来“非正定”
        print(f"{name:<25} | Error: {str(e)[:15]}... | (Numerical Instability)")