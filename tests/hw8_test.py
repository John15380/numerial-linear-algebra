import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from algorithms.create_matrix import create_tridiagonal, create_hilbert
from algorithms.qr_decomposition import qr_decomposition
from algorithms.direct_solver import back_substitution
import numpy as np

def qr_solve(A_packed, b):
    m, n = A_packed.shape
    b = b.copy().astype(np.float64)
    for j in range(n):
        v = np.zeros(m - j)
        v[0] = 1.0
        if j + 1 < m:
            v[1:] = A_packed[j+1:, j]
        vv = np.dot(v, v)
        if vv > 1.0:
            beta = 2.0 / vv
            b[j:] -= beta * v * (v @ b[j:])
    R = np.triu(A_packed[:n, :n])
    return back_substitution(R, b[:n])

# 1
print(f"{'Algorithm':<30} | {'Relative Error':<18} | {'Residual':<18}")
print("-" * 75)

A84, _, _ = create_tridiagonal(84, 6, 1, 8)
x_true84 = np.ones(84)
b84 = A84 @ x_true84
with np.errstate(divide='ignore', invalid='ignore'):
    x84 = qr_solve(qr_decomposition(A84.copy()), b84)
if np.any(np.isnan(x84)):
    print(f"{'QR Tridiag(84,6,1,8)':<30} | {'ill-conditioned':<18} | {'N/A':<18}")
else:
    err = np.linalg.norm(x84 - x_true84, np.inf) / np.linalg.norm(x_true84, np.inf)
    res = np.linalg.norm(A84 @ x84 - b84, np.inf)
    print(f"{'QR Tridiag(84,6,1,8)':<30} | {err:.6e}          | {res:.6e}")

A100, b100, x_true100 = create_tridiagonal(100, 10, 1, 1)
x100 = qr_solve(qr_decomposition(A100.copy()), b100)
err = np.linalg.norm(x100 - x_true100, np.inf) / np.linalg.norm(x_true100, np.inf)
res = np.linalg.norm(A100 @ x100 - b100, np.inf)
print(f"{'QR Tridiag(100,10,1,1)':<30} | {err:.6e}          | {res:.6e}")

A, b, x_true = create_hilbert(40)
x = qr_solve(qr_decomposition(A), b)
err = np.linalg.norm(x - x_true, np.inf) / np.linalg.norm(x_true, np.inf)
res = np.linalg.norm(A @ x - b, np.inf)
print(f"{'QR Hilbert(40)':<30} | {err:.6e}          | {res:.6e}")

# 2
data2 = np.loadtxt("../data/table3_2.csv", delimiter=",", skiprows=1)
t = data2[:, 0]
y2 = data2[:, 1]
A2 = np.column_stack([t**2, t, np.ones(len(t))])
a, b_coef, c = qr_solve(qr_decomposition(A2), y2)
print(f"\ny = {a:.4f}t^2 + {b_coef:.4f}t + {c:.4f}")
print(f"残量2范数: {np.linalg.norm(y2 - (a*t**2 + b_coef*t + c)):.4e}")

# 3
data3 = np.loadtxt("../data/table3_3_4.csv", delimiter=",", skiprows=1)
y3 = data3[:, 0]
X = data3[:, 1:]
m3 = X.shape[0]
A3 = np.hstack([np.ones((m3, 1)), X])
x_sol = qr_solve(qr_decomposition(A3.copy()), y3)
print(f"\n常数项 x0: {x_sol[0]:.6f}")
for i in range(1, len(x_sol)):
    print(f"x{i}: {x_sol[i]:.6f}")
