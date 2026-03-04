import numpy as np
import time
from scipy.sparse import diags, lil_matrix
from scipy.sparse.linalg import norm as sparse_norm

def criar_matriz_problema(h):
    """
    Cria a matriz do sistema discretizado para o problema modelo
    com condições de contorno dadas por g(x,y)
    """
    n = int(1/h) - 1  # número de pontos internos em cada direção
    N = n * n  # dimensão total da matriz
    
    # Criar matriz esparsa
    A = lil_matrix((N, N))
    b = np.zeros(N)
    
    # Preencher a matriz A e o vetor b
    for i in range(n):
        for j in range(n):
            idx = i * n + j  # índice no vetor solução
            
            x = (j + 1) * h
            y = (i + 1) * h
            
            # Coeficiente diagonal: 4
            A[idx, idx] = 4.0
            
            # Vizinho à esquerda (j-1)
            if j > 0:
                A[idx, idx - 1] = -1.0
            else:
                # Condição de contorno em x = 0
                # g(0, y) = 0 se x = 0
                b[idx] += 0.0
            
            # Vizinho à direita (j+1)
            if j < n - 1:
                A[idx, idx + 1] = -1.0
            else:
                # Condição de contorno em x = 1
                # g(1, y) = y se x = 1
                b[idx] += y
            
            # Vizinho abaixo (i-1)
            if i > 0:
                A[idx, idx - n] = -1.0
            else:
                # Condição de contorno em y = 0
                # g(x, 0) = (x-1)*sin(x) se y = 0
                b[idx] += (x - 1) * np.sin(x)
            
            # Vizinho acima (i+1)
            if i < n - 1:
                A[idx, idx + n] = -1.0
            else:
                # Condição de contorno em y = 1
                # g(x, 1) = x(2-x) se y = 1
                b[idx] += x * (2 - x)
    
    # Multiplicar b por h^2 (vem da discretização)
    b = b * (h * h)
    
    return A.tocsr(), b, n

def metodo_jacobi(A, b, tol=1e-8, max_iter=100000):
    """
    Implementa o método de Jacobi para resolver Ax = b
    """
    n = len(b)
    x = np.zeros(n)  # chute inicial u^(0) = 0
    x_new = np.zeros(n)
    
    # Extrair diagonal de A
    D = A.diagonal()
    
    iteracoes = 0
    inicio = time.time()
    
    for k in range(max_iter):
        # x^(k+1) = D^(-1) * (b - (A - D) * x^(k))
        for i in range(n):
            soma = 0.0
            # Calcular A[i,:] * x, exceto o termo diagonal
            for j in A.getrow(i).nonzero()[1]:
                if j != i:
                    soma += A[i, j] * x[j]
            
            x_new[i] = (b[i] - soma) / D[i]
        
        # Calcular norma da mudança relativa
        diff_norm = np.linalg.norm(x_new - x, 2)
        x_norm = np.linalg.norm(x_new, 2)
        
        if x_norm > 0:
            rel_change = diff_norm / x_norm
            
            if rel_change < tol:
                iteracoes = k + 1
                break
        
        x[:] = x_new
        iteracoes = k + 1
    
    tempo = time.time() - inicio
    
    return x, iteracoes, tempo

def main():
    print("=" * 70)
    print("Exemplo 8.2.8 - Método de Jacobi para Problema Modelo")
    print("=" * 70)
    print("\nProblema: -∇²u = f com f = 0")
    print("\nCondições de contorno g(x,y):")
    print("  g(x, 0) = (x-1)sin(x)  se y = 0")
    print("  g(x, 1) = x(2-x)       se y = 1")
    print("  g(0, y) = 0            se x = 0")
    print("  g(1, y) = y            se x = 1")
    print("\nChute inicial: u^(0) = 0")
    print("Critério de parada: ||u^(k+1) - u^(k)||₂ / ||u^(k+1)||₂ < 10^(-8)")
    print("\n" + "=" * 70)
    
    # Valores de h para testar
    valores_h = [1/20, 1/40, 1/80, 1/160]
    
    # Tentar valores menores se o computador permitir
    try:
        valores_h.extend([1/320, 1/640])
    except:
        pass
    
    resultados = []
    
    print("\n{:<15} {:<20} {:<20} {:<15}".format(
        "h", "Dimensão Matriz", "Iterações", "Tempo (s)"))
    print("-" * 70)
    
    for h in valores_h:
        try:
            # Criar o sistema
            A, b, n = criar_matriz_problema(h)
            dimensao = A.shape[0]
            
            # Resolver com Jacobi
            x, iteracoes, tempo = metodo_jacobi(A, b)
            
            resultados.append({
                'h': h,
                'h_frac': f"1/{int(1/h)}",
                'dimensao': dimensao,
                'iteracoes': iteracoes,
                'tempo': tempo
            })
            
            print(f"{resultados[-1]['h_frac']:<15} {dimensao:<20} {iteracoes:<20} {tempo:<15.2f}")
            
        except MemoryError:
            print(f"{1/h:<15.0f} Memória insuficiente")
            break
        except Exception as e:
            print(f"{1/h:<15.0f} Erro: {str(e)}")
            break
    
    print("\n" + "=" * 70)
    print("\nTabela 8.1 - Método de Jacobi aplicado ao problema modelo")
    print("=" * 70)
    print(f"\n{'h':<15} {'Dimensão da':<20} {'Iterações para':<20}")
    print(f"{'':15} {'Matriz':<20} {'Convergência':<20}")
    print("-" * 55)
    
    for r in resultados:
        print(f"{r['h_frac']:<15} {r['dimensao']:<20} {r['iteracoes']:<20}")
    
    print("\n" + "=" * 70)
    print("\nObservações:")
    print("- A convergência é lenta, como esperado para o método de Jacobi")
    print("- O número de iterações cresce significativamente conforme h diminui")
    print("- A matriz se torna cada vez maior (n² onde n = 1/h - 1)")
    print("=" * 70)
    
    return resultados

if __name__ == "__main__":
    resultados = main()