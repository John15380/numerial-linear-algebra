import numpy as np
from algorithms.householder import house
from algorithms.givens import givens


def tridiagonalize(A):
    """
    用 Householder 变换将实对称矩阵 A 化为对称三对角矩阵 T，
    并累积正交变换矩阵 Q，使得 A = Q T Q^T

    返回:
        d: T 的对角线元素 (n,)
        e: T 的次对角线元素 (n-1,)，e[i] = T[i, i+1]
        Q: 正交矩阵 (n, n)
    """
    A = np.array(A, dtype=np.float64, copy=True)
    n = A.shape[0]
    Q = np.eye(n)

    for k in range(n - 2):
        x = A[k + 1 :, k].copy()
        v, beta = house(x)
        if beta == 0:
            continue

        # 更新 A[k+1:, k+1:]（对称更新）
        p = beta * A[k + 1 :, k + 1 :] @ v
        K = beta * (p.T @ v) / 2.0
        w = p - K * v
        A[k + 1 :, k + 1 :] -= np.outer(v, w) + np.outer(w, v)

        # 更新第 k 列的次对角线部分
        A[k, k + 1 :] = A[k + 1 :, k] = x - beta * (v @ x) * v

        # 累积正交变换 Q = Q * H_k，其中 H_k = diag(I_k, I - beta*v*v^T)
        # 即只更新 Q 的后 n-k-1 列
        q_part = Q[:, k + 1 :]
        q_part -= beta * np.outer(q_part @ v, v)

    d = np.diag(A)
    e = np.diag(A, k=1)
    return d, e, Q


def symmetric_implicit_qr_step(d, e, Q, m):
    """
    对对称三对角矩阵 T 的前 m×m 主子矩阵执行一步隐式 QR 迭代（带 Wilkinson 位移）
    T 由对角线 d 和次对角线 e 表示

    位移 mu 取右下角 2x2 块的靠近 d[m-1] 的特征值
    通过 Givens 旋转 chase the bulge，恢复三对角形式

    同时累积正交变换到 Q

    参数:
        d: 对角线元素，会被就地修改
        e: 次对角线元素，会被就地修改
        Q: 正交矩阵，会被就地修改
        m: 当前处理的子矩阵维数
    """
    if m <= 1:
        return

    # Wilkinson 位移：右下角 2x2 块 [d[m-2], e[m-2]; e[m-2], d[m-1]]
    # 选择靠近 d[m-1] 的特征值
    delta = (d[m - 2] - d[m - 1]) / 2.0
    if delta == 0:
        sign_delta = 1.0
    else:
        sign_delta = np.sign(delta)
    mu = d[m - 1] - sign_delta * e[m - 2] ** 2 / (
        abs(delta) + np.sqrt(delta ** 2 + e[m - 2] ** 2)
    )

    # 隐式 QR 步：构造第一个 Givens 旋转，作用于 (T - mu I) 的第 0,1 行
    x = d[0] - mu
    y = e[0]
    c, s = givens(x, y)

    # 应用 Givens 旋转到 T 的 0,1 行和列
    G = np.array([[c, -s], [s, c]])
    T_block = np.array([[d[0], e[0]], [e[0], d[1]]])
    T_block = G.T @ T_block @ G
    d[0], d[1] = T_block[0, 0], T_block[1, 1]
    e[0] = T_block[0, 1]
    # bulge 出现在 (2, 0) 和 (0, 2)
    bulge = s * e[1] if m > 2 else 0.0
    if m > 2:
        e[1] = c * e[1]

    # 累积到 Q
    q_cols = Q[:, [0, 1]]
    Q[:, [0, 1]] = q_cols @ G

    # Chase the bulge
    for i in range(m - 2):
        x = e[i]
        y = bulge
        c, s = givens(x, y)

        # 更新 e[i]
        e[i] = c * x + s * y

        # 应用 Givens 旋转 G(i+1, i+2) 到 T 的 2x2 块
        if i + 2 < m:
            T_sub = np.array([
                [d[i + 1], e[i + 1]],
                [e[i + 1], d[i + 2]]
            ])
            G = np.array([[c, -s], [s, c]])
            T_sub = G.T @ T_sub @ G
            d[i + 1], d[i + 2] = T_sub[0, 0], T_sub[1, 1]
            e[i + 1] = T_sub[0, 1]

            # 累积到 Q
            q_cols = Q[:, [i + 1, i + 2]]
            Q[:, [i + 1, i + 2]] = q_cols @ G

            # 更新 bulge 到下一个位置
            if i + 3 < m:
                old_e = e[i + 2]
                bulge = s * old_e
                e[i + 2] = c * old_e
            else:
                bulge = 0.0
        else:
            # 边界情况
            e[i] = c * x + s * y


def symmetric_qr_eigen(A, tol=1e-12, max_iter=1000):
    """
    隐式对称 QR 算法求实对称矩阵 A 的全部特征值和特征向量

    步骤:
        1. Householder 变换化为对称三对角矩阵 T = Q^T A Q
        2. 对 T 应用带 Wilkinson 位移的隐式 QR 迭代
        3. 累积所有正交变换，得到特征向量矩阵

    返回:
        eigenvalues: 特征值数组 (n,)，按升序排列
        eigenvectors: 特征向量矩阵 (n, n)，每列为对应特征值的特征向量
        iterations: 总 QR 迭代次数
    """
    A = np.array(A, dtype=np.float64, copy=True)
    n = A.shape[0]
    if n == 0:
        return np.array([]), np.empty((0, 0)), 0

    # 步骤 1: 三对角化
    d, e, Q = tridiagonalize(A)

    # 确保 d 和 e 是可写的
    d = d.copy()
    e = e.copy()

    # 步骤 2 & 3: 隐式 QR 迭代
    total_iters = 0
    m = n
    while m > 1:
        # 检查次对角线元素是否收敛到零
        if abs(e[m - 2]) < tol * (abs(d[m - 1]) + abs(d[m - 2])):
            e[m - 2] = 0.0
            m -= 1
            continue

        # 检查是否需要进行隐式 QR 步
        if total_iters >= max_iter:
            break

        symmetric_implicit_qr_step(d, e, Q, m)
        total_iters += 1

    eigenvalues = d.copy()
    eigenvectors = Q.copy()

    # 按特征值升序排列
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    return eigenvalues, eigenvectors, total_iters
