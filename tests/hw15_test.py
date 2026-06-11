import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from algorithms.jacobi_eigen import jacobi_eigen
from algorithms.matrix_utils import create_tridiagonal_matrix


def main():
    # ==================================================================
    # 题目：经典 Jacobi 方法求实对称矩阵的全部特征值和特征向量
    # 教材第 244 页 1(1) 和 1(2)
    # ==================================================================
    print("=" * 70)
    print("经典 Jacobi 方法")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1(2)：n 阶对称三对角矩阵，主对角线 4，次对角线 1，n = 50 到 100
    # ------------------------------------------------------------------
    print("\n矩阵：对称三对角 (主对角线 4, 次对角线 1)")
    print(f"{'n':<8} {'Sweeps':<8} {'λ_min':<16} {'λ_max':<16} "
          f"{'max_diff':<14} {'max_residual':<14}")
    print("-" * 80)

    for n in [50, 60, 70, 80, 90, 100]:
        A = create_tridiagonal_matrix(n, 4, 1)
        eigenvalues, eigenvectors, sweeps = jacobi_eigen(A, tol=1e-14, max_sweeps=200)

        np_eig = np.linalg.eigvalsh(A)
        max_diff = np.max(np.abs(eigenvalues - np_eig))
        max_residual = max(
            np.linalg.norm(A @ eigenvectors[:, i]
                         - eigenvalues[i] * eigenvectors[:, i])
            for i in range(n)
        )

        print(f"{n:<8} {sweeps:<8} {eigenvalues[0]:<16.10f} "
              f"{eigenvalues[-1]:<16.10f} {max_diff:<14.2e} {max_residual:<14.2e}")

    # ------------------------------------------------------------------
    # 详细展示 n=50 和 n=100 的结果
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("详细结果")
    print("=" * 70)

    for n in [50, 100]:
        A = create_tridiagonal_matrix(n, 4, 1)
        eigenvalues, eigenvectors, sweeps = jacobi_eigen(A, tol=1e-14, max_sweeps=200)

        np_eig = np.linalg.eigvalsh(A)
        max_diff = np.max(np.abs(eigenvalues - np_eig))
        orth_err = np.linalg.norm(eigenvectors.T @ eigenvectors - np.eye(n))

        print(f"\nn = {n}:")
        print(f"  Sweeps: {sweeps}")
        print(f"  最小特征值: {eigenvalues[0]:.10f}")
        print(f"  最大特征值: {eigenvalues[-1]:.10f}")
        print(f"  与 numpy.eigvalsh 最大差异: {max_diff:.2e}")
        print(f"  特征向量正交性误差 ||V^T V - I||_2: {orth_err:.2e}")

        # 理论特征值
        lam_1 = 4 + 2 * np.cos(n * np.pi / (n + 1))
        lam_n = 4 + 2 * np.cos(np.pi / (n + 1))
        print(f"  理论 λ_min = {lam_1:.10f}")
        print(f"  理论 λ_max = {lam_n:.10f}")

        # 输出前 3 个特征值
        print(f"  前 3 个特征值: {eigenvalues[:3]}")
        print(f"  后 3 个特征值: {eigenvalues[-3:]}")

        # 输出最小特征值对应的特征向量前 5 个分量
        print(f"\n  最小特征值对应特征向量 (前 5 分量):")
        print(f"    {eigenvectors[:, 0][:5]}")
        print(f"  最大特征值对应特征向量 (前 5 分量):")
        print(f"    {eigenvectors[:, -1][:5]}")


if __name__ == "__main__":
    main()
