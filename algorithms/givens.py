import numpy as np


def givens(a, b):
    """
    计算 Givens 旋转参数 c, s，使得
    G = [c  -s; s  c] 满足 G^T [a; b] = [r; 0]
    即 c*a + s*b = r, -s*a + c*b = 0
    """
    if b == 0:
        return 1.0, 0.0
    r = np.hypot(a, b)
    c = a / r
    s = b / r
    return c, s
