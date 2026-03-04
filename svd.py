import numpy as np
import scipy.linalg as la

A=np.array([[3,0],[4,0]])
U, s, Vh = la.svd(A)
print("U Matrix:\n", U)
print("Singular Values:\n", s)
print("Vh Matrix:\n", Vh)
 