import numpy as np
np.set_printoptions(suppress=True, precision=6)

from math import factorial
import matplotlib.pyplot as plt
from methods import mJacobi, mHouseholder
import scipy as sc

hbarra = 6.582119569e-16  # const. Planck [eV*s]
eV = 1  # = 1 J
w = eV/hbarra  # frecuencia angular de oscilacion [1/s]
M = 9.1093837015e-31  # masa de un electron

D0 = np.sqrt(hbarra/(M*w))  # unidad de distancia [Angstrom]
Eef = hbarra * w  # unidad de energia [eV]
M = 1  # unidad de masa [kg] (masa de un electron)
V0 = 1  # [eV]
tolJacobi = 1e-10  # tolerancia
tol = 0.1/100  # precision buscada en E0, E1

def deltaK(n, m):  # delta de Kronecker
    if n == m:
        return 1
    else:
        return 0

def potencial(n, m, D, V1):
    # operador posicion**1
    x1 = (
        np.sqrt(n) * deltaK(m, n-1)
        + np.sqrt(n+1) * deltaK(m, n+1)
    )

    # operador posicion**2
    x2 = (
        np.sqrt(n*(n-1)) * deltaK(m, n-2)
        + (2*n+1) * deltaK(m, n)
        + np.sqrt((n+1)*(n+2)) * deltaK(m, n+2)
    )

    # operador posicion**4
    x4 = (
        np.sqrt(n*(n-1)*(n-2)*(n-3)) * deltaK(m, n-4)
        + (4*n - 2) * np.sqrt(n*(n-1)) * deltaK(m, n-2)
        + (6*n**2 + 6*n + 3) * deltaK(m, n)
        + ((4*n + 6) * np.sqrt((n+1)*(n+2))) * deltaK(m, n+2)
        + np.sqrt((n+1)*(n+2)*(n+3)*(n+4)) * deltaK(m, n+4)
    )

    Vmol = 4 * V0 * (x4 * ((1/D)**4) - x2 * ((1/D)**2))  # V molecular
    Velec = (V1/(np.sqrt(2))) * (x1 * (1/D))  # V electrico

    return Vmol + Velec

def hamiltoniano(dim, D, V1):
    H = np.zeros([dim, dim])
    for n in range(dim):
        for m in range(dim):
            # operador momento cuadrado
            p2 = (
                np.sqrt((n+1)*(n+2)) * deltaK(m, n+2)
                - (2*n+1) * deltaK(m, n)
                + np.sqrt(n*(n-1)) * deltaK(m, n-2)
            )
            V = potencial(n, m, D, V1)
            H[n, m] = -p2/4 + V/Eef
    return H

def funcionOnda(x, n, x0=D0):
    psi = (1/(np.sqrt(np.sqrt(np.pi) * (2**n) * (factorial(n) * x0)))) * np.exp((-1/2) *
          (x/x0)**2) * sc.special.hermite(n)(x/x0)
    return psi

def pot2(x, D, V1=0):
    Vmol = 4 * V0 * (4 * (x * (1/D))**4 - 2 * (x * (1/D))**2)
    Velec = V1 * (x/D)
    return Vmol + Velec

""" Inciso a) """
V1 = 0
D = 1  # D ~ D/D0, distancia relativa
dim = 5  # dimension inicial de H
H = hamiltoniano(dim, D, V1)
Q, R = mHouseholder(H)  # tridiagonalizo
H, V, k = mJacobi(R @ Q, tolJacobi)  # diagonalizo
print(H, "Autovalores")
print("")
print(V, "Autovectores")

""" Inciso b) Diagonalizar H y hallar la dim(H) necesaria para tener una precision relativa del 0.1 % de la energia del estado fundamental."""
fig = plt.figure()
ax = fig.add_subplot()
V1 = 0
D = np.array([0.3, 1.0, 12])  # = D/D0
n = 0  # estado de interes

for i in range(len(D)):
    dim = 5  # dimension inicial de H (la minima posible)
    H = hamiltoniano(dim, D[i], V1)  # H inicial
    H, V = np.linalg.eigh(H)  # determino autovalores y autovectores utilizando Numpy
    E = []
    Dim = []
    while True:
        dim += 2
        Hnew = hamiltoniano(dim, D[i], V1)  # H "nuevo"
        Hnew, V = np.linalg.eigh(Hnew)
        eps = abs((Hnew[n] - H[n])/Hnew[n])
        if eps > tol:
            E.append(eps)
            Dim.append(dim)
            H = Hnew
        else:
            E.append(eps)
            Dim.append(dim)
            H = Hnew
            break
    plt.plot(Dim, E, "o", markersize="3", label=f"D/D0 = {D[i]}")
    np.savetxt(f"incisoB{i}.dat", np.array([Dim, E]).T, fmt=" %s")

x = np.linspace(0, 500)
y = tol * np.ones(len(x))  # recta: y = tol
plt.plot(x, y, "--", color="gray")
plt.xlabel("dim(H)")
plt.ylabel("Error relativo porcentual, ϵ0")
plt.xlim(5, 90)
plt.legend()
plt.savefig(f"figurasPNGs/incisoB.png")

""" Inciso c) """
fig = plt.figure()
ax = fig.add_subplot()
V1 = 0
D = np.zeros(200)  # D/D0
I = 2
J = 9  # solo indices
for i in range(1):
    for j in range(I*J):  # 18 puntos entre 0.1 y 1
        D[j+9*i] = (j/2 + 1)*10**(-1)
    D[I*J:] = np.linspace(1, 12, len(D)-I*J)

Dim = np.zeros(len(D))
n = 0
while n <= 2:
    for i in range(len(D)):
        dim = 5
        H = hamiltoniano(dim, D[i], V1)
        H, V = np.linalg.eigh(H)
        while True:
            dim += 2
            Hnew = hamiltoniano(dim, D[i], V1)
            Hnew, Vnew = np.linalg.eigh(Hnew)
            E = abs((Hnew[n] - H[n])/Hnew[n])
            if E > tol:
                H = Hnew
            else:
                H = Hnew
                break
        Dim[i] = dim
    plt.plot(D, Dim, "o", markersize="1", label=f"En")
    if n == 0:
        Dim0 = np.copy(Dim)
    elif n == 1:
        Dim1 = np.copy(Dim)
    elif n == 2:
        Dim2 = np.copy(Dim)
    n += 1

np.savetxt(f"data/incisoC.dat", np.array([D, Dim0, Dim1, Dim2]).transpose(), fmt=" %s")
plt.xlabel("D/D0")
plt.ylabel("dim(H)")
plt.legend()
plt.savefig(f"figurasPNGs/incisoC.png")

# Parametro (D) bajo estudio
V1 = 0
D = np.array([D[0], D[-1]])
Dim = np.zeros(len(D), dtype=np.int64)

# Hallo las dimensiones opticas para D
n = 0  # nro de estado
while n <= 1:
    dim = 5  # dimension inicial (la minima posible)
    H = hamiltoniano(dim, D[n], V1)  # H inicial
    H, V = np.linalg.eigh(H)  # determino autovalores y autoV1ctores de H

    while True:
        dim += 2  # actualizo la dimension
        Hnew = hamiltoniano(dim, D[n], V1)  # H "nuevo"
        Hnew, Vnew = np.linalg.eigh(Hnew)

        E = abs((Hnew[n] - H[n]) / Hnew[n])  # calculo la precision con la dimension actual
        if E > tol:
            H = Hnew
        else:
            H = Hnew
            break
    Dim[n] = dim
    n += 1
print(f"Inciso c). Si D = {D}, dim(H) = {Dim}.")
print("")

for i in range(len(D)):
    # Plotteo
    fig = plt.figure()
    ax = fig.add_subplot()

    n = 0  # nro de estado
    while n <= 1:
        H = hamiltoniano(Dim[i], D[i], V1)
        H, phi = np.linalg.eigh(H)

        #color = next(ax._get_lines.prop_cycler)['color']

        a = -8 * D[i];
        b = 8 * D[i];
        N = 300
        x = np.linspace(a, b, N)
        psi = np.zeros(len(x))  # funcion de onda
        for j in range(len(x)):
            for k in range(len(phi)):
                psi[j] += phi[k, n] * funcionOnda(x[j], k)
        if n == 0:
            psi = abs(psi)

        ax.plot(x, psi, label=f"ψn")

        # Integro para verificar que la probabilidad en los pozos (region de estos) es 1/2
        sum = 0
        dx = (b - a) / N
        for j in range(int(N / 2), N):
            sum += psi[j] ** 2 * dx
        if n == 0:
            psi0 = np.copy(psi)
            sum0 = sum
        else:
            psi1 = np.copy(psi)
            sum1 = sum
        n += 1
    print(f"Inciso d). Probabilidades: P0 = {sum0}, P1 = {sum1}")

    # Potencial
    V = pot2(x, D[i])
    ax.plot(x, V, color="gray", label="V ")

    # Detalles del plotteo
    ax.set_xlabel("D/D0");
    ax.set_ylabel("ψn, V ")
    ax.set_ylim(-abs(V0 + 1), abs(V0 + 1))
    plt.legend()

    if D[i] < 1:
        plt.savefig(f"figurasPNGs/incisoD1.png")
        np.savetxt(f"data/psiD1.dat", np.array([x, psi0, psi1, V]).T, fmt=" %s")

    if D[i] > 1:
        plt.savefig(f"figurasPNGs/incisoD2.png")
        np.savetxt(f"data/psiD2.dat", np.array([x, psi0, psi1, V]).T, fmt=" %s")

# Plotteo
fig = plt.figure()
ax = fig.add_subplot()

# Parametro (D) bajo estudio
D = 10
V1 = np.array([0, 0.05, 0.8])

# Hallo las dimensiones optima para D
n = 0  # nro de estado
for i in range(len(V1)):
    dim = 5
    H = hamiltoniano(dim, D, V1[i])
    H, phi = np.linalg.eigh(H)

    while True:
        dim += 2
        Hnew = hamiltoniano(dim, D, V1[i])
        Hnew, phi = np.linalg.eigh(Hnew)

        E = abs((Hnew[n] - H[n]) / Hnew[n])
        if E > tol:
            H = Hnew
        else:
            H = Hnew
            break
    print(f"Inciso e). Si D = {D} y V1 = {V1[i]}, dim(H) = {dim}.")

    #color = next(ax._get_lines.prop_cycler)['color']

    a = -1.5 * D;
    b = 1.5 * D;
    N = 100
    x = np.linspace(a, b, N)
    psi = np.zeros(len(x))
    for j in range(len(x)):
        for k in range(len(phi)):
            psi[j] += phi[k, n] * funcionOnda(x[j], k)
    psi = abs(psi)
    ax.plot(x, psi, label=f"V1 = V1[i]")

    # Potencial
    V = pot2(x, D, V1=V1[i])
    ax.plot(x, V, ":")

    # Integracion
    sum = 0;
    dx = (b - a) / N
    for j in range(int(N / 2), N):
        sum += psi[j] ** 2 * dx
    if i == 0:
        psi0 = np.copy(psi)
        sum0 = sum
        pot0 = V
    elif i == 1:
        psi1 = np.copy(psi)
        sum1 = sum
        pot1 = V
    else:
        psi2 = np.copy(psi)
        sum2 = sum
        pot2 = V

print(f"Inciso e). Probabilidades: P0 = {sum0}, P1 = {sum1}, P2 = {sum2}.")

# Detalles del plotteo
ax.set_xlabel("D/D0")
ax.set_ylim(-2, 2)
plt.legend()
plt.savefig(f"figurasPNGs/IncisoE.png")

if V1[-1] > 0:
    np.savetxt(f"data/incisoE1.dat", np.array([x, psi0, pot0, psi1, pot1, psi2, pot2]).T, fmt=" %s")
else:
    np.savetxt(f"data/incisoE2.dat", np.array([x, psi0, pot0, psi1, pot1, psi2, pot2]).T, fmt=" %s")
