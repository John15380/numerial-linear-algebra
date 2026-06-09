import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from algorithms.power_method_polynomial import power_method_polynomial, power_method_polynomial_direct
import numpy as np


def verify_root(coeffs, root):
    """
    验证 root 是否为多项式的根，计算 |f(root)|
    coeffs = [a_{n-1}, ..., a_1, a_0]
    f(x) = x^n + a_{n-1} x^{n-1} + ... + a_0
    """
    n = len(coeffs)
    val = root ** n
    for i, a in enumerate(coeffs):
        power = n - 1 - i
        val += a * (root ** power)
    return abs(val)


def all_roots_numpy(coeffs):
    """
    用 numpy 求所有根作为参考
    coeffs 为 [a_{n-1}, ..., a_1, a_0]
    numpy 需要 [1, a_{n-1}, ..., a_0]
    """
    poly = [1.0] + list(coeffs)
    return np.roots(poly)


def main():
    # ------------------------------------------------------------------
    # 题目 (i): x^3 + x^2 - 5x + 3 = 0
    # coeffs: a_2=1, a_1=-5, a_0=3
    # ------------------------------------------------------------------
    print("=" * 60)
    print("(i) x^3 + x^2 - 5x + 3 = 0")
    print("=" * 60)
    coeffs1 = [1.0, -5.0, 3.0]
    roots1_ref = all_roots_numpy(coeffs1)
    print(f"numpy 全部根: {roots1_ref}")
    print(f"模最大根 (numpy): {max(roots1_ref, key=abs)}")

    root1, iters1, conv1 = power_method_polynomial(coeffs1, tol=1e-12, max_iter=1000)
    print(f"\n幂法结果: {root1}")
    print(f"迭代次数: {iters1}, 收敛: {conv1}")
    print(f"验证 |f(root)| = {verify_root(coeffs1, root1):.6e}")

    # ------------------------------------------------------------------
    # 题目 (ii): x^3 - 3x - 1 = 0
    # coeffs: a_2=0, a_1=-3, a_0=-1
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("(ii) x^3 - 3x - 1 = 0")
    print("=" * 60)
    coeffs2 = [0.0, -3.0, -1.0]
    roots2_ref = all_roots_numpy(coeffs2)
    print(f"numpy 全部根: {roots2_ref}")
    print(f"模最大根 (numpy): {max(roots2_ref, key=abs)}")

    root2, iters2, conv2 = power_method_polynomial(coeffs2, tol=1e-12, max_iter=1000)
    print(f"\n幂法结果: {root2}")
    print(f"迭代次数: {iters2}, 收敛: {conv2}")
    print(f"验证 |f(root)| = {verify_root(coeffs2, root2):.6e}")

    # ------------------------------------------------------------------
    # 题目 (iii): 8次方程
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("(iii) 8次方程")
    print("=" * 60)
    coeffs3 = [
        101.0,       # a_7
        208.01,      # a_6
        10891.01,    # a_5
        9802.08,     # a_4
        79108.9,     # a_3
        -99902.0,    # a_2
        790.0,       # a_1
        -1000.0,     # a_0
    ]
    roots3_ref = all_roots_numpy(coeffs3)
    print(f"numpy 全部根:")
    for r in roots3_ref:
        print(f"  {r}")
    print(f"模最大根 (numpy): {max(roots3_ref, key=abs)}")

    root3, iters3, conv3 = power_method_polynomial(coeffs3, tol=1e-12, max_iter=5000)
    print(f"\n幂法结果: {root3}")
    print(f"迭代次数: {iters3}, 收敛: {conv3}")
    print(f"验证 |f(root)| = {verify_root(coeffs3, root3):.6e}")

    # ------------------------------------------------------------------
    # 汇总表
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("汇总")
    print("=" * 60)
    print(f"{'方程':<30} | {'模最大根':<18} | {'迭代次数':<8}")
    print("-" * 60)
    print(f"{'(i) x^3+x^2-5x+3=0':<30} | {root1:<18.10f} | {iters1:<8}")
    print(f"{'(ii) x^3-3x-1=0':<30} | {root2:<18.10f} | {iters2:<8}")
    print(f"{'(iii) 8次方程':<30} | {root3:<18.10f} | {iters3:<8}")


if __name__ == "__main__":
    main()
