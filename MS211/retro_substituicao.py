import numpy as np
A = np.array([
    [.0004, 15.73],
    [.423, -24.72],
    
], dtype=float)

c = np.array([15.77,-20.49], dtype=float)

def triangular_superior(A, c):
    n = A.shape[0]
    x = np.zeros(n)
    x[n-1] = c[n-1] / A[n-1, n-1]

    for i in range(n-2, -1, -1):
        soma = c[i]
        for j in range(i+1, n):
            soma -= A[i, j] * x[j]
        x[i] = soma / A[i, i]

    return x


def triangular_inferior(A, c):
    n = A.shape[0]
    x = np.zeros(n)
    x[0] = c[0] / A[0, 0]

    for i in range(1, n):
        soma = c[i]
        for j in range(0, i):
            soma -= A[i, j] * x[j]
        x[i] = soma / A[i, i]
    return x
def eliminacao_gaussiana(A, C):
    A = np.array(A, dtype=float)
    C = np.array(C, dtype=float)

    n = len(C)
    for i in range(n):
        if A[i][i] == 0:
            for k in range(i+1, n):
                if A[k][i] != 0:
                    A[[i, k]] = A[[k, i]]
                    C[i], C[k] = C[k], C[i]
                    break
        for j in range(i+1, n):
            fator = A[j][i] / A[i][i]
            A[j] = A[j] - fator * A[i]
            C[j] = C[j] - fator * C[i]

    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = (C[i] - np.dot(A[i, i+1:], x[i+1:])) / A[i][i]

    return x
def eh_triangular_superior(A):
    n = A.shape[0]
    for i in range(1,n):
        for j in range (i):
            if A[i][j] != 0 and i>j:
                return False
    return True
    
def eh_triangular_inferior(A):
    n = A.shape[0]
    for i in range(n):
        for j in range(i+1,n):
            if A[i][j] != 0 and j>i:
                return False
    return True
if eh_triangular_inferior(A):
    print("Solução inferior:", triangular_inferior(A,c))
elif eh_triangular_superior(A):
    print("Solução superior:", triangular_superior(A, c))
else:
    print("Solução com eliminação gaussiana:", eliminacao_gaussiana(A,c))
