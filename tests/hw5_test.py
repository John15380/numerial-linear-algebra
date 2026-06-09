import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from algorithms.norm_estimation import norm1_estimation, norm1_estimation_inv
from algorithms.create_matrix import create_hilbert, create_an
from algorithms.direct_solver import gauss_partial_pivoting
import numpy as np
import matplotlib.pyplot as plt

# test1
n_list_1 = []
cond_list = []

for i in range(5, 20):
    H, _, _ = create_hilbert(i)
    x = np.ones(i) / i

    cond = norm1_estimation(H, x) * norm1_estimation_inv(H, x)
    print(f"{i}阶Hilbert矩阵的∞范数条件数为{cond}")
    n_list_1.append(i)
    cond_list.append(cond)
print("=========")

plt.plot(n_list_1, cond_list, marker='o')
plt.yscale("log")
plt.show()

# test2
n_list_2 = []
true_err_list = []
est_err_list = []

for i in range(5, 30):
    x = np.random.randn(i)
    A = create_an(i)
    b = A @ x
    x_hat = gauss_partial_pivoting(A, b)

    n_list_2.append(i)

    # 真实相对误差
    true_err = np.max(np.abs(x_hat - x)) / np.max(np.abs(x))
    true_err_list.append(true_err)
    print(f"{i}阶真实误差{true_err}")

    # 估计相对误差
    v_tilde = norm1_estimation_inv(A.T, np.ones(i)/i)
    r_tilde = np.max(np.abs(b - A @ x_hat))
    beta_tilde = np.max(np.abs(b))
    mu_tilde = norm1_estimation(A.T, np.ones(i)/i)

    est_err = v_tilde * mu_tilde * r_tilde / beta_tilde
    est_err_list.append(est_err)
    print(f"{i}阶估计误差{est_err}")

plt.plot(n_list_2, true_err_list, marker='o')
plt.plot(n_list_2, est_err_list, marker='o')
plt.yscale("log")
plt.show()