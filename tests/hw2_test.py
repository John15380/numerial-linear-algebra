import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from algorithms.direct_solver import gauss_no_pivoting, gauss_partial_pivoting
from algorithms.create_matrix import create_tridiagonal
import numpy as np

A, b, x = create_tridiagonal(84, 6, 1, 8)
x = gauss_no_pivoting(A, b)
print(np.array_str(x, max_line_width=1000, precision=4, suppress_small=True))
x = gauss_partial_pivoting(A, b)
print(np.array_str(x, max_line_width=1000, precision=4, suppress_small=True))