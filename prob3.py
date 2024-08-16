import numpy as np
import matplotlib.pyplot as plt

def sumaUp(N):
    suma = 0
    for i in range(1, N+1):
        reciproco = 1 / i
        suma += reciproco
    return suma

def sumaDown(N):
    suma = 0
    for i in range(N, 0, -1):
        reciproco = 1 / i
        suma += reciproco
    return suma

def diferencia(N):
    lista = []
    arr = np.empty(N)
    for i in range(1, N+1):
        dif = (sumaUp(i) - sumaDown(i))/ (abs(sumaUp(i)) + abs(sumaDown(i)))
        np.append(arr, dif)
        lista.append(dif)
    return arr, lista

N = 1000
valoresX = np.arange(1, N+1, 1)
arr, lista = diferencia(N)


fig, ax = plt.subplots()

plt.plot(valoresX, arr)

plt.yscale('log')
plt.xscale('log')

plt.show()

