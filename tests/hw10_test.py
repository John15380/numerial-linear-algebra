import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from algorithms.iteration import jacobi_iteration, gauss_seidel_iteration, sor_iteration, calculate_optimal_omega
import numpy as np

def create_system(epsilon, a, n):
    """
    根据给定的差分方程建立线性方程组 Ax = b
    """
    h = 1.0 / n
    size = n - 1
    A = np.zeros((size, size))
    b = np.full(size, a * (h ** 2))

    # 填充三对角系数矩阵
    for i in range(size):
        # 主对角线元素
        A[i, i] = -(2 * epsilon + h)
        # 下对角线元素
        if i > 0:
            A[i, i - 1] = epsilon
        # 上对角线元素
        if i < size - 1:
            A[i, i + 1] = epsilon + h

    # 处理边界条件: y(0) = 0 对 b[0] 无影响
    # y(n) = 1，将常数项移到右边，修改 b 的最后一个分量
    b[-1] -= (epsilon + h) * 1.0

    return A, b

def get_exact_solution(x, epsilon, a):
    """
    计算解析精确解
    """
    # 避免 epsilon 过小时 e^(-1/epsilon) 溢出报错
    # 对于极小 epsilon (如0.0001)，np.exp(-1/epsilon) 会安全地变为 0
    term1 = (1 - a) / (1 - np.exp(-1.0 / epsilon))
    term2 = 1 - np.exp(-x / epsilon)
    y = term1 * term2 + a * x
    return y

def main():
    a_val, n_val = 0.5, 100
    eps_list = [1.0, 0.1, 0.01, 0.0001]
    h = 1.0 / n_val
    x_nodes = np.linspace(h, 1 - h, n_val - 1)
    
    print(f"{'Epsilon':<10} | {'Method':<10} | {'Omega':<8} | {'Iters':<8} | {'Max Error'}")
    print("-" * 65)
    
    for eps in eps_list:
        A, b = create_system(eps, a_val, n_val)
        y_exact = get_exact_solution(x_nodes, eps, a_val)
        
        # 计算当前 epsilon 下的最佳 omega
        omega_opt = calculate_optimal_omega(A)
        
        # 各方法求解
        results = [
            ("Jacobi", 1.0, *jacobi_iteration(A, b)),
            ("G-S", 1.0, *gauss_seidel_iteration(A, b)),
            ("SOR", omega_opt, *sor_iteration(A, b, omega_opt))
        ]
        
        for name, omega, x_sol, iters in results:
            error = np.max(np.abs(x_sol - y_exact))
            omega_str = f"{omega:.4f}" if name == "SOR" else "-"
            print(f"{eps:<10} | {name:<10} | {omega_str:<8} | {iters:<8} | {error:.6e}")
        print("-" * 65)

if __name__ == "__main__":
    main()