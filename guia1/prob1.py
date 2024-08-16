import numpy as np
import matplotlib.pyplot as plt


def f(x):
    return np.exp(x) * np.log(x) - x ** 2


def derivada(x):
    return np.exp(x) * np.log(x) + np.exp(x) + 1 / x - 2 * x


def Newton_Raphson(f, dx,x0, N, tol):
    x = x0
    for i in range(N):
        x_new = x - f(x) / dx(x)

        if abs(x_new - x) < tol:
            return x

        x = x_new

raiz = Newton_Raphson(f, derivada, 2,100000,1e-9)
print(raiz)

x = np.linspace(0.5, 2.5, 1000)

fig, ax = plt.subplots()

plt.plot(x, f(x))
plt.scatter(raiz,f(raiz),color='r')
plt.axhline(y=0, color='grey', linestyle='--', linewidth=1)

plt.show()