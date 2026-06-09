import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from algorithms.symmetric_qr import symmetric_qr_eigen
from algorithms.implicit_qr import eigen_general
from algorithms.matrix_utils import create_tridiagonal_matrix, companion_matrix
import numpy as np


def verify_polynomial_root(root, coeffs):
    """
    验证 root 是否为多项式的根，计算 |p(root)|
    coeffs = [a_{n-1}, ..., a_1, a_0]
    p(x) = x^n + a_{n-1} x^{n-1} + ... + a_0
    """
    n = len(coeffs)
    val = root ** n
    for i, a in enumerate(coeffs):
        power = n - 1 - i
        val += a * (root ** power)
    return abs(val)


def main():
    # ==================================================================
    # 题目一：隐式对称 QR 算法求实对称矩阵的全部特征值和特征向量
    # 教材第 244 页 1(2) 和 2(2)
    # ==================================================================
    print("=" * 70)
    print("题目一：隐式对称 QR 算法")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1(2): n 阶对称三对角矩阵，主对角线为 4，次对角线为 1
    # ------------------------------------------------------------------
    print("\n--- 1(2): 对称三对角矩阵 (主对角线 4, 次对角线 1) ---")
    for n in [50, 100]:
        A = create_tridiagonal_matrix(n, 4, 1)
        eigenvalues, eigenvectors, iters = symmetric_qr_eigen(A)
        np_eig = np.linalg.eigvalsh(A)
        max_diff = np.max(np.abs(eigenvalues - np_eig))
        max_residual = max(
            np.linalg.norm(A @ eigenvectors[:, i] - eigenvalues[i] * eigenvectors[:, i])
            for i in range(n)
        )
        print(f"\nn = {n}:")
        print(f"  QR 迭代次数: {iters}")
        print(f"  最小特征值: {eigenvalues[0]:.10f}")
        print(f"  最大特征值: {eigenvalues[-1]:.10f}")
        print(f"  与 numpy 最大差异: {max_diff:.2e}")
        print(f"  最大残差 ||Av - λv||_2: {max_residual:.2e}")

    # ------------------------------------------------------------------
    # 2(2): 100 阶对称三对角矩阵，主对角线为 2，次对角线为 -1
    # ------------------------------------------------------------------
    print("\n--- 2(2): 100 阶对称三对角矩阵 (主对角线 2, 次对角线 -1) ---")
    A2 = create_tridiagonal_matrix(100, 2, -1)
    eigenvalues2, eigenvectors2, iters2 = symmetric_qr_eigen(A2)
    np_eig2 = np.linalg.eigvalsh(A2)
    max_diff2 = np.max(np.abs(eigenvalues2 - np_eig2))
    max_residual2 = max(
        np.linalg.norm(A2 @ eigenvectors2[:, i] - eigenvalues2[i] * eigenvectors2[:, i])
        for i in range(100)
    )
    print(f"\nQR 迭代次数: {iters2}")
    print(f"最小特征值: {eigenvalues2[0]:.10f}")
    print(f"最大特征值: {eigenvalues2[-1]:.10f}")
    print(f"与 numpy 最大差异: {max_diff2:.2e}")
    print(f"最大残差 ||Av - λv||_2: {max_residual2:.2e}")

    # 输出最大和最小特征值对应的特征向量（前5个分量）
    print(f"\n最小特征值对应的特征向量 (前 5 个分量):")
    print(f"  {eigenvectors2[:, 0][:5]}")
    print(f"最大特征值对应的特征向量 (前 5 个分量):")
    print(f"  {eigenvectors2[:, -1][:5]}")

    # ==================================================================
    # 题目二：隐式 QR 算法（Francis 双重位移）
    # 教材第 202 页 2(2) 和 2(3)
    # ==================================================================
    print("\n" + "=" * 70)
    print("题目二：隐式 QR 算法（一般实矩阵）")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 2(2): x^41 + x^3 + 1 = 0 的全部根
    # ------------------------------------------------------------------
    print("\n--- 2(2): x^41 + x^3 + 1 = 0 的全部根 ---")
    # 多项式: x^41 + x^3 + 1 = 0
    # coeffs = [a40, a39, ..., a0]
    # a40 = a39 = ... = a3 = 0, a2 = 1, a1 = 0, a0 = 1
    coeffs41 = [0.0] * 38 + [1.0, 0.0, 1.0]
    C41 = companion_matrix(coeffs41)
    roots41, _, its41 = eigen_general(C41, max_iter_factor=200)
    np_roots41 = np.roots([1.0] + coeffs41)
    max_diff41 = np.max(np.abs(np.sort(roots41) - np.sort(np_roots41)))

    print(f"\nQR 迭代次数: {its41}")
    print(f"与 numpy.roots 最大差异: {max_diff41:.2e}")

    # 输出前 10 个根
    print(f"\n前 10 个根（按实部排序）:")
    sorted_roots = sorted(roots41, key=lambda z: (np.real(z), np.imag(z)))
    for i in range(10):
        r = sorted_roots[i]
        val = abs(r**41 + r**3 + 1)
        print(f"  root_{i+1:2d}: {r:.6f}  |p(root)| = {val:.2e}")

    # ------------------------------------------------------------------
    # 2(3): 矩阵 A(x) 在 x = 0.9, 1.0, 1.1 时的全部特征值
    # ------------------------------------------------------------------
    print("\n--- 2(3): 矩阵 A(x) 的特征值 ---")

    def A_matrix(x):
        return np.array([
            [9.1, 3.0, 2.6, 4.0],
            [1.2, 5.3, 1.7, 1.0],
            [3.2, 1.7, 9.4, x],
            [6.1, 4.9, 3.5, 6.2]
        ], dtype=np.float64)

    for x in [0.9, 1.0, 1.1]:
        A = A_matrix(x)
        eigenvalues, eigenvectors, its = eigen_general(A)
        np_eig = np.linalg.eigvals(A)
        max_diff = np.max(np.abs(np.sort(eigenvalues) - np.sort(np_eig)))

        print(f"\nx = {x}:")
        print(f"  QR 迭代次数: {its}")
        print(f"  特征值: {np.sort(eigenvalues)}")
        print(f"  与 numpy 最大差异: {max_diff:.2e}")

    # ------------------------------------------------------------------
    # 汇总：观察特征值随 x 的变化
    # ------------------------------------------------------------------
    print("\n--- 特征值随 x 的变化情况 ---")
    print(f"{'x':<8} | {'λ1':<14} | {'λ2':<14} | {'λ3':<14} | {'λ4':<14}")
    print("-" * 70)
    for x in [0.9, 1.0, 1.1]:
        A = A_matrix(x)
        eigenvalues, _, _ = eigen_general(A)
        sorted_eig = np.sort(eigenvalues)
        print(f"{x:<8} | {sorted_eig[0]:<14.6f} | {sorted_eig[1]:<14.6f} | "
              f"{sorted_eig[2]:<14.6f} | {sorted_eig[3]:<14.6f}")


if __name__ == "__main__":
    main()
