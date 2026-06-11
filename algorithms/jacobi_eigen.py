import numpy as np


def jacobi_eigen(A, tol=1e-12, max_sweeps=100):
    """
    经典 Jacobi 方法求实对称矩阵 A 的全部特征值和特征向量

    每轮（sweep）扫描所有上三角非对角元，若 |a_pq| > threshold 则施以
    一重 Jacobi 旋转将其对消。threshold 逐轮缩小，保证收敛。

    旋转角度满足 tan(2θ) = 2*a_pq / (a_pp - a_qq):
        τ = (a_qq - a_pp) / (2*a_pq)
        t = sign(τ) / (|τ| + sqrt(1 + τ²))
        c = 1 / sqrt(1 + t²),   s = c * t

    参数:
        A: 实对称矩阵
        tol: 收敛容限
        max_sweeps: 最大扫描轮数

    返回:
        eigenvalues: 特征值数组 (n,)，按升序排列
        eigenvectors: 特征向量矩阵 (n, n)，每列为对应特征向量
        sweeps: 完成的扫描轮数
    """
    A = np.array(A, dtype=np.float64, copy=True)
    n = A.shape[0]
    if n == 0:
        return np.array([]), np.empty((0, 0)), 0

    # 初始化特征向量矩阵
    V = np.eye(n)

    # 初始阈值：取非对角元均方根的估计
    off_norm = np.sqrt(np.sum(np.abs(A) ** 2) - np.sum(np.diag(A) ** 2))
    threshold = off_norm / n if n > 1 else 0.0

    sweeps = 0
    for sweep in range(max_sweeps):
        # 如果阈值已低于 tol，收敛
        if threshold < tol:
            break
        sweeps += 1

        for p in range(n - 1):
            for q in range(p + 1, n):
                if abs(A[p, q]) < threshold:
                    continue

                # 计算 Jacobi 旋转参数
                tau = (A[q, q] - A[p, p]) / (2.0 * A[p, q])
                if tau >= 0:
                    t_val = 1.0 / (tau + np.sqrt(1.0 + tau ** 2))
                else:
                    t_val = -1.0 / (-tau + np.sqrt(1.0 + tau ** 2))
                c = 1.0 / np.sqrt(1.0 + t_val ** 2)
                s = c * t_val

                # 更新对角线上的 p, q
                app = A[p, p]
                aqq = A[q, q]
                apq = A[p, q]

                A[p, p] = c * c * app - 2.0 * c * s * apq + s * s * aqq
                A[q, q] = s * s * app + 2.0 * c * s * apq + c * c * aqq
                A[p, q] = A[q, p] = 0.0  # 被对消

                # 更新第 p 行和第 q 行的其他元素
                for i in range(n):
                    if i == p or i == q:
                        continue
                    aip = A[i, p]
                    aiq = A[i, q]
                    A[i, p] = A[p, i] = c * aip - s * aiq
                    A[i, q] = A[q, i] = s * aip + c * aiq

                # 累积特征向量
                for i in range(n):
                    vip = V[i, p]
                    viq = V[i, q]
                    V[i, p] = c * vip - s * viq
                    V[i, q] = s * vip + c * viq

        # 基于当前非对角范数更新阈值
        off_sq = np.sum(np.abs(A) ** 2) - np.sum(np.diag(A) ** 2)
        off_norm = np.sqrt(max(off_sq, 0.0))
        threshold = off_norm / n if off_norm > tol else 0.0

    eigenvalues = np.diag(A)
    eigenvectors = V.copy()

    # 按特征值升序排列
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    return eigenvalues, eigenvectors, sweeps


def jacobi_eigen_classical(A, tol=1e-12, max_iters=None):
    """
    经典 Jacobi 方法（逐次对消最大非对角元），作为上述扫描策略的参考实现。

    每次选取模最大的非对角元 A[p,q] 进行 Jacobi 旋转。

    返回:
        eigenvalues, eigenvectors, iterations
    """
    A = np.array(A, dtype=np.float64, copy=True)
    n = A.shape[0]
    if n == 0:
        return np.array([]), np.empty((0, 0)), 0

    V = np.eye(n)

    if max_iters is None:
        max_iters = 100 * n * n

    iters = 0
    for it in range(max_iters):
        iters += 1

        # 查找模最大的非对角元
        max_val = 0.0
        p, q = 0, 1
        for i in range(n - 1):
            for j in range(i + 1, n):
                if abs(A[i, j]) > max_val:
                    max_val = abs(A[i, j])
                    p, q = i, j

        if max_val < tol:
            break

        # 计算旋转参数
        tau = (A[q, q] - A[p, p]) / (2.0 * A[p, q])
        if tau >= 0:
            t_val = 1.0 / (tau + np.sqrt(1.0 + tau ** 2))
        else:
            t_val = -1.0 / (-tau + np.sqrt(1.0 + tau ** 2))
        c = 1.0 / np.sqrt(1.0 + t_val ** 2)
        s = c * t_val

        # 更新 A
        app, aqq, apq = A[p, p], A[q, q], A[p, q]
        A[p, p] = c * c * app - 2.0 * c * s * apq + s * s * aqq
        A[q, q] = s * s * app + 2.0 * c * s * apq + c * c * aqq
        A[p, q] = A[q, p] = 0.0

        for i in range(n):
            if i == p or i == q:
                continue
            aip, aiq = A[i, p], A[i, q]
            A[i, p] = A[p, i] = c * aip - s * aiq
            A[i, q] = A[q, i] = s * aip + c * aiq

        # 更新特征向量
        for i in range(n):
            vip, viq = V[i, p], V[i, q]
            V[i, p] = c * vip - s * viq
            V[i, q] = s * vip + c * viq

    eigenvalues = np.diag(A)
    eigenvectors = V.copy()

    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    return eigenvalues, eigenvectors, iters
