import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from algorithms.conjugate_gradient import conjugate_gradient
from algorithms.iteration import jacobi_iteration, gauss_seidel_iteration
import numpy as np

def create_problem2(n):
    """
    构造题目2的矩阵 A 和向量 b
    a_{ij} = 1/(i+j+1) (1-indexed), b_i = (1/3) * sum_j a_{ij}
    真解为 x = (1/3, ..., 1/3)
    """
    # 0-indexed: A[i,j] = 1/((i+1)+(j+1)+1) = 1/(i+j+3)
    A = np.array([[1.0 / (i + j + 3) for j in range(n)] for i in range(n)])
    b = A.sum(axis=1) / 3.0
    x_true = np.full(n, 1.0 / 3.0)
    return A, b, x_true

def spectral_radius(M):
    return max(abs(np.linalg.eigvals(M)))

def main():
    # -- 题目2: CG 法求解 Hilbert-like 矩阵 --
    print("题目2: 共轭梯度法求解 Hilbert-like 矩阵 (a_ij = 1/(i+j+1))")
    print(f"{'n':<6} | {'Iters':<8} | {'Rel Error':<14} | {'||Ax-b||_inf':<16} | {'cond(A)'}")
    print("-" * 68)

    for n in [5, 10, 20, 50, 100]:
        A, b, x_true = create_problem2(n)
        x, iters = conjugate_gradient(A, b, tol=1e-10, max_iter=10000)
        rel_err  = np.linalg.norm(x - x_true, np.inf) / np.linalg.norm(x_true, np.inf)
        residual = np.linalg.norm(A @ x - b, np.inf)
        cond     = np.linalg.cond(A)
        print(f"{n:<6} | {iters:<8} | {rel_err:.6e}     | {residual:.6e}      | {cond:.3e}")

    print()

    # -- 题目3: Jacobi / G-S / CG 求解 5x5 方程组 --
    print("题目3: 各迭代法求解 5x5 方程组")
    A = np.array([
        [10.,  1.,  2.,  3.,  4.],
        [ 1.,  9., -1.,  2., -3.],
        [ 2., -1.,  7.,  3., -5.],
        [ 3.,  2.,  3., 12., -1.],
        [ 4., -3., -5., -1., 15.]
    ])
    b = np.array([12., -27., 14., -17., 12.])

    x_ref = np.linalg.solve(A, b)
    print(f"参考解: {np.array_str(x_ref, precision=6)}\n")

    # 计算 Jacobi 和 G-S 的谱半径，说明收敛性
    D   = np.diag(np.diag(A))
    L   = np.tril(A, -1)
    U   = np.triu(A, 1)
    T_J  = -np.diag(1.0 / np.diag(A)) @ (L + U)
    T_GS = -np.linalg.inv(D + L) @ U
    print(f"Jacobi 谱半径 rho(T_J)  = {spectral_radius(T_J):.6f}")
    print(f"G-S    谱半径 rho(T_GS) = {spectral_radius(T_GS):.6f}\n")

    print(f"{'Method':<12} | {'Iters':<8} | {'Rel Error':<14} | {'||Ax-b||_inf'}")
    print("-" * 55)

    x_j,  iters_j  = jacobi_iteration(A, b)
    x_gs, iters_gs = gauss_seidel_iteration(A, b)
    x_cg, iters_cg = conjugate_gradient(A, b, tol=1e-10, max_iter=100)

    for name, x_sol, iters in [("Jacobi", x_j, iters_j),
                                ("G-S",    x_gs, iters_gs),
                                ("CG",     x_cg, iters_cg)]:
        rel_err  = np.linalg.norm(x_sol - x_ref, np.inf) / np.linalg.norm(x_ref, np.inf)
        residual = np.linalg.norm(A @ x_sol - b, np.inf)
        print(f"{name:<12} | {iters:<8} | {rel_err:.6e}     | {residual:.6e}")

if __name__ == "__main__":
    main()
