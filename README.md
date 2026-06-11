# 数值代数上机作业

本仓库包含《数值代数》课程的上机作业代码与报告，涵盖直接解法、迭代方法、特征值计算等核心内容。

## 目录结构

```
.
├── algorithms/        # 算法程序（通用子程序）
├── tests/             # 各次作业的测试/主程序
├── data/              # 数据文件（CSV）
├── reports/           # LaTeX 报告与 PDF
├── Makefile           # 报告编译管理
└── README.md          # 本文件
```

## 算法程序说明（algorithms/）

| 文件 | 功能 |
|------|------|
| `householder.py` | 通用的 Householder 变换，返回归一化向量 `v`（`v[0]=1`）与系数 `beta`，用于将对称矩阵化为三对角、一般矩阵化为 Hessenberg 等 |
| `givens.py` | Givens 平面旋转，返回参数 `c, s`，用于 QR 分解中的消元与隐式 QR 的 bulge chasing |
| `matrix_utils.py` | 矩阵构造工具：对称三对角矩阵 `create_tridiagonal_matrix`、首一多项式友矩阵 `companion_matrix` |
| `symmetric_qr.py` | **隐式对称 QR 算法**：先 Householder 三对角化，再带 Wilkinson 位移的隐式 QR 迭代求全部特征值和特征向量 |
| `implicit_qr.py` | **Francis 双重位移 QR 算法**：先 Householder Hessenberg 化，再对 $M=H^2-sH+tI$ 做 QR 分解实现双重位移步，求实 Schur 形式与特征值；含反幂法求特征向量 |
| `jacobi_eigen.py` | **经典 Jacobi 方法**：逐轮扫描非对角元，施以平面旋转对消，求实对称矩阵的全部特征值和特征向量 |
| `direct_solver.py` | 直接解法：Gauss 消去（无选主元 / 列主元）、Cholesky 分解、$LDL^T$ 分解、回代法 |
| `iteration.py` | 迭代方法：Jacobi、Gauss-Seidel、SOR 迭代，以及最佳松弛因子计算 |
| `conjugate_gradient.py` | 共轭梯度法（CG）求解对称正定线性方程组 |
| `power_method_polynomial.py` | 幂法求多项式模最大根（构造友矩阵后做幂迭代） |
| `qr_decomposition.py` | Householder QR 分解（将矩阵化为上三角_packed 形式） |
| `create_matrix.py` | 矩阵生成：三对角矩阵、Hilbert 矩阵等 |
| `norm_estimation.py` | 1-范数估计（$\|B\|_1$ 与 $\|B^{-1}\|_1$），用于条件数估计 |

## 各次作业内容

### 作业 2（hw2_test.py）
- **内容**：Gauss 消去法（无选主元 vs 列主元）求解三对角方程组
- **算法**：`direct_solver.py`

### 作业 3（hw3_test.py）
- **内容**：比较 Gauss 消去、Cholesky、$LDL^T$ 分解在良态三对角矩阵与病态 Hilbert 矩阵上的表现
- **算法**：`direct_solver.py`, `create_matrix.py`

### 作业 5（hw5_test.py）
- **内容**：1-范数估计与矩阵条件数计算
- **算法**：`norm_estimation.py`

### 作业 8（hw8_test.py）
- **内容**：Householder QR 分解求解最小二乘问题（多项式拟合与多元线性回归）
- **算法**：`qr_decomposition.py`, `direct_solver.py`
- **数据**：`data/table3_2.csv`, `data/table3_3_4.csv`

### 作业 10（hw10_test.py）
- **内容**：差分格式对应的线性方程组，用 Jacobi、Gauss-Seidel、SOR 迭代求解，比较收敛速度与最佳松弛因子
- **算法**：`iteration.py`

### 作业 11（hw11_test.py）
- **内容**：共轭梯度法求解 Hilbert-like 大型稀疏矩阵，并与 Jacobi/GS 比较
- **算法**：`conjugate_gradient.py`, `iteration.py`

### 作业 12（hw12_test.py）
- **内容**：幂法求多项式方程的模最大根（ companion 矩阵 + 幂迭代）
- **算法**：`power_method_polynomial.py`

### 作业 14（hw14_test.py）
- **内容**：
  1. **隐式对称 QR 算法**：求对称三对角矩阵（教材 p.244 1(2)、2(2)）的全部特征值与特征向量
  2. **Francis 双重位移 QR 算法**：求 $x^{41}+x^3+1=0$ 的全部根，以及矩阵 $A(x)$ 在 $x=0.9,1.0,1.1$ 时的特征值
- **算法**：`symmetric_qr.py`, `implicit_qr.py`, `matrix_utils.py`

### 作业 15（hw15_test.py）
- **内容**：经典 Jacobi 方法求实对称矩阵（教材 p.244 1(2)，$n=50\sim100$）的全部特征值和特征向量
- **算法**：`jacobi_eigen.py`, `matrix_utils.py`

## 使用方法

### 运行测试程序

在仓库根目录下执行：

```bash
# 运行第 14 次作业
python3 tests/hw14_test.py

# 运行其他作业
python3 tests/hw10_test.py
python3 tests/hw12_test.py
```

### 编译报告

```bash
# 编译指定作业的报告（如第 14 次）
make 14

# 编译所有报告
make

# 清理编译生成的辅助文件（保留 .tex 和 .pdf）
make clean
```

报告 PDF 将生成在 `reports/` 目录下。

## 依赖

- Python 3.x
- NumPy
- matplotlib（仅作业 5 用到）
- XeLaTeX（编译报告用）
