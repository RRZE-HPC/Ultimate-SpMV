import sys
import numpy as np
from scipy.io.mmio import mmread
from scipy.sparse.linalg import norm

def get_matrix_norm(matrixname):
    matrix = mmread(matrixname).tolil()
    return norm(matrix, ord=np.inf)

def get_threshold(norm, tol, roundoff):
    return (tol * norm) / roundoff

float_roundoff = 0.5 * 2**-23 

matrixname = sys.argv[1]
tol = float(sys.argv[2])

# matrixnorm = get_matrix_norm(matrixname)
matrix = mmread(matrixname).tolil()
matrixnorm = norm(matrix, np.inf)

print(f"threshold of {matrixname} under tolerance {tol} is {get_threshold(matrixnorm, tol, float_roundoff)}")