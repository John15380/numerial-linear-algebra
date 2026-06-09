import numpy as np
from algorithms.householder import house


def hessenberg(A):
    """
    用 Householder 变换将一般实矩阵 A 化为上 Hessenberg 形式 H，
    并累积正交变换矩阵 Q，使得 A = Q H Q^T

    返回:
        H: 上 Hessenberg 矩阵
        Q: 正交矩阵
    """
    A = np.array(A, dtype=np.float64, copy=True)
    n = A.shape[0]
    Q = np.eye(n)

    for k in range(n - 2):
        v, beta = house(A[k + 1 :, k])
        if beta == 0:
            continue

        # 更新 A 的行: A[k+1:, k:] = (I - beta * v @ v^T) @ A[k+1:, k:]
        A[k + 1 :, k:] -= beta * np.outer(v, v @ A[k + 1 :, k:])
        # 更新 A 的列: A[:, k+1:] = A[:, k+1:] @ (I - beta * v @ v^T)
        A[:, k + 1 :] -= beta * np.outer(A[:, k + 1 :] @ v, v)
        # 累积 Q: Q[:, k+1:] = Q[:, k+1:] @ (I - beta * v @ v^T)
        Q[:, k + 1 :] -= beta * np.outer(Q[:, k + 1 :] @ v, v)

    # 清零数值噪音
    for i in range(n):
        for j in range(i + 2, n):
            A[j, i] = 0.0

    return A, Q


def francis_double_shift_step(H, Q, m):
    """
    对 H 的左上角 m×m 子矩阵执行一步 Francis 双重位移 QR 迭代

    采用显式双重位移策略：
    M = (H - sigma1*I)(H - sigma2*I) = H^2 - s*H + t*I
    对 M 进行 QR 分解得到 Q_step，然后 H_new = Q_step^T @ H @ Q_step

    参数:
        H: 上 Hessenberg 矩阵，会被就地修改
        Q: 正交变换累积矩阵，会被就地修改
        m: 当前处理的子矩阵维数
    """
    if m <= 2:
        return

    # 位移 sigma1, sigma2 为 H[m-2:m, m-2:m] 的特征值
    s = H[m - 2, m - 2] + H[m - 1, m - 1]
    t = H[m - 2, m - 2] * H[m - 1, m - 1] - H[m - 2, m - 1] * H[m - 1, m - 2]

    # M = H^2 - s*H + t*I
    M = H[:m, :m] @ H[:m, :m] - s * H[:m, :m] + t * np.eye(m)

    # QR 分解
    Q_step, R = np.linalg.qr(M)

    # 更新 H 和 Q
    H[:m, :m] = Q_step.T @ H[:m, :m] @ Q_step
    Q[:, :m] = Q[:, :m] @ Q_step

    # 清零 Hessenberg 结构外的数值噪音
    for i in range(m):
        for j in range(i + 2, m):
            H[j, i] = 0.0


def has_complex_eigenvalue_2x2(H, i):
    """
    判断 H 的第 i 和 i+1 行/列构成的 2x2 块是否有复特征值
    """
    a, b, c, d = H[i, i], H[i, i + 1], H[i + 1, i], H[i + 1, i + 1]
    trace = a + d
    det = a * d - b * c
    disc = trace ** 2 - 4 * det
    return disc < 0


def francis_qr_eigenvalues(H, tol=1e-12, max_iter_factor=100):
    """
    Francis 双重位移 QR 算法求上 Hessenberg 矩阵 H 的全部特征值

    返回实 Schur 形式，然后提取特征值。
    对于 1×1 对角块，特征值为实数；
    对于 2×2 对角块，特征值为一对共轭复数。

    返回:
        eigenvalues: 特征值数组（可能包含复数）
        H_schur: 实 Schur 形式的上 Hessenberg 矩阵
        Q: 正交矩阵，使得 A = Q H_schur Q^T
        iterations: QR 迭代次数
    """
    n = H.shape[0]
    Q = np.eye(n)
    max_iter = max_iter_factor * n
    its = 0
    m = n

    while m > 2 and its < max_iter:
        its += 1

        # 检查次对角线元素是否收敛到零
        if abs(H[m - 1, m - 2]) < tol * (
            abs(H[m - 1, m - 1]) + abs(H[m - 2, m - 2])
        ):
            H[m - 1, m - 2] = 0.0
            m -= 1
            continue

        # 如果右下角 2×2 块有复特征值，且次对角线已经比较稳定，
        # 则接受当前的 2×2 块，不再继续迭代
        if m >= 3 and has_complex_eigenvalue_2x2(H, m - 2):
            # 检查次对角线是否足够小或已稳定
            if abs(H[m - 1, m - 2]) < 1e-8 or its > max_iter // 2:
                m -= 2
                continue

        francis_double_shift_step(H, Q, m)

    # 提取特征值
    eigenvalues = []
    i = 0
    while i < n:
        if i == n - 1 or abs(H[i + 1, i]) < tol:
            # 1×1 块
            eigenvalues.append(H[i, i])
            i += 1
        else:
            # 2×2 块
            a, b, c, d = H[i, i], H[i, i + 1], H[i + 1, i], H[i + 1, i + 1]
            trace = a + d
            det = a * d - b * c
            disc = trace ** 2 - 4 * det
            if disc >= 0:
                lam1 = (trace + np.sqrt(disc)) / 2.0
                lam2 = (trace - np.sqrt(disc)) / 2.0
                eigenvalues.append(lam1)
                eigenvalues.append(lam2)
            else:
                lam1 = (trace + 1j * np.sqrt(-disc)) / 2.0
                lam2 = (trace - 1j * np.sqrt(-disc)) / 2.0
                eigenvalues.append(lam1)
                eigenvalues.append(lam2)
            i += 2

    return np.array(eigenvalues), H, Q, its


def inverse_iteration(A, eigenvalue, tol=1e-12, max_iter=100):
    """
    反幂法（带位移）求矩阵 A 对应于近似特征值 eigenvalue 的特征向量

    对 (A - mu * I) 应用逆迭代，收敛到最接近 mu 的特征向量

    返回:
        eigenvector: 特征向量（复数数组如果特征值为复数）
        converged: 是否收敛
    """
    n = A.shape[0]
    mu = eigenvalue
    is_complex = np.iscomplex(mu)

    if is_complex:
        A_shifted = A.astype(np.complex128) - mu * np.eye(n, dtype=np.complex128)
        x = np.random.rand(n) + 1j * np.random.rand(n)
    else:
        A_shifted = A - mu * np.eye(n)
        x = np.random.rand(n)

    x = x / np.linalg.norm(x)

    for _ in range(max_iter):
        try:
            y = np.linalg.solve(A_shifted, x)
        except np.linalg.LinAlgError:
            # 如果 A - mu*I 接近奇异，添加微小扰动
            if is_complex:
                A_shifted += 1e-14 * np.eye(n, dtype=np.complex128)
            else:
                A_shifted += 1e-14 * np.eye(n)
            y = np.linalg.solve(A_shifted, x)

        norm_y = np.linalg.norm(y)
        if norm_y == 0:
            break
        x_new = y / norm_y

        if np.linalg.norm(x_new - x) < tol:
            return x_new, True
        x = x_new

    return x, False


def eigen_general(A, tol=1e-12, max_iter_factor=100):
    """
    隐式 QR 算法（Francis 双重位移）求一般实矩阵 A 的全部特征值和特征向量

    步骤:
        1. Householder 变换化为上 Hessenberg 形式 H = Q^T A Q
        2. Francis 双重位移 QR 迭代求实 Schur 形式
        3. 提取特征值
        4. 对每个特征值用反幂法求特征向量

    返回:
        eigenvalues: 特征值数组
        eigenvectors: 特征向量矩阵（每列为对应特征值的特征向量）
        iterations: QR 迭代次数
    """
    A = np.array(A, dtype=np.float64, copy=True)
    n = A.shape[0]

    # 步骤 1: Hessenberg 化
    H, Q_hess = hessenberg(A)

    # 步骤 2: Francis QR 迭代
    eigenvalues, H_schur, Q_schur, its = francis_qr_eigenvalues(
        H, tol=tol, max_iter_factor=max_iter_factor
    )

    # 总正交变换: A = Q_total H_schur Q_total^T
    Q_total = Q_hess @ Q_schur

    # 步骤 4: 反幂法求特征向量
    eigenvectors = np.zeros((n, len(eigenvalues)), dtype=np.complex128)
    for i, lam in enumerate(eigenvalues):
        vec, conv = inverse_iteration(A, lam, tol=tol)
        eigenvectors[:, i] = vec

    return eigenvalues, eigenvectors, its
