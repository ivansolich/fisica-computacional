import numpy as np

def mJacobi(A, tol): # probl
    n = np.shape(A)[0] # orden de A
    V = np.identity(n) # inicializo la matriz de autovectores
    k = 0 # nro de iteraciones

    while True:
        # Reviso los elementos diagonales superiores de A
        for i in range(n): # recorro filas
            for j in range(i+1, n): # recorro columnas
                if abs(A[i,j]) > tol:
                    p, q = i, j

        if abs(A[p,q]) < tol or k > 1e5:
            break
        else:
            k += 1

        # Calculo el angulo de rotacion
        phi = 0.5 * np.arctan2(2 * A[p,q], A[q,q] - A[p,p])
        c = np.cos(phi)
        s = np.sin(phi)

        # Defino la matriz de Jacobi
        P = np.identity(n) # reinicializo en cada step
        P[p,p] = c
        P[p,q] = s
        P[q,p] = -s
        P[q,q] = c

        # Calculo autovalores y autovectores
        A = P.T @ A @ P # actualizo A
        V = V @ P # acumulo P

    return A, V, k # A: matriz diagonal; V: matriz cuyas columnas son los autovectores normalizados


def mHouseholder(A): # prob2
    n = np.shape(A)[0]
    I = np.identity(n)
    # Q = np.copy(I); R = np.copy(A)
    for i in range(n-2):
        x = np.zeros(n) # vector de n componentes
        x[i+1:] = A[i+1:,i] # últimas n-1 componentes
        x[i+1] = x[i+1] - np.linalg.norm(x) # x - |x| e1

        # Construyo la Householder’s matrix P
        u = np.array([x]).T # redefino como matriz
        H = (u.T @ u) / 2 # factor de normalización * 1/2
        P = I - (u @ u.T) / H # Householder’s matrix

        # Cantidades útiles (computacionalmente baratas)
        # p = (A @ u) / H
        # K = (u.T @ p) / (2*H)
        # q = p - K * u

        # Calculo (y defino) las matrices Q y R
        if i == 0: # defino Q y R
            Q = P # matriz unitaria
            R = P @ A # matriz no unitaria
        else: # actualizo R y Q
            Q = Q @ P # here too!
            R = P @ R # importa el orden
        # A = A - q @ u.T - u @ q.T # actualizo A
        A = R @ Q # actualizo A
    return Q, R
