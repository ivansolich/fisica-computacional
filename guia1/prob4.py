import numpy as np
import matplotlib.pyplot as plt
import scipy as sp


class BisectionAlgorithm:
    def __init__(self, func, x):
        self.func = func
        self.x = x

    def initialPlot(self):
        ax, fig = plt.subplots()
        plt.plot(self.x, self.func(self.x))
        plt.axhline(y=0, color='grey', linestyle='--', linewidth=1)
        plt.grid(True, axis='x')
        plt.show()

    def algorithm(self, a, b, tol=1e-10, max_iter=1000):

        if self.func(a) * self.func(b) >= 0:
            raise ValueError("f(a) y f(b) deben tener signos opuestos")

        count = 0

        while abs(self.func(a) - self.func(b)) > tol and count < max_iter:
            count += 1
            c = 0.5 * (a + b)
            if self.func(c) == 0.0:
                return c

            if self.func(a) * self.func(c) < 0:
                b = c

            else:
                a = c

        return (a + b) / 2.0

    def comparision(self, a, b, tol=1e-10):
        sp_root = sp.optimize.bisect(self.func, a, b)
        error = abs(sp_root - self.algorithm(a, b, tol, max_iter=1000))
        return f'El valor de la raiz utilizando Scipy es, {sp_root} y su error absoluto, {error}'


def even_wavefunction(Eb):
    return np.sqrt(10 - Eb) * np.tan(np.sqrt(10 - Eb)) - np.sqrt(Eb)


def even_wavefunction2(E):
    def cot(x):
        return 1 / np.tan(x)

    return np.sqrt(E) * cot(np.sqrt(10 - E)) - np.sqrt(10 - E)


Eb = E = np.linspace(8, 10, 100)

bisection = BisectionAlgorithm(even_wavefunction, Eb)
bisection.initialPlot()
root = BisectionAlgorithm(even_wavefunction, Eb).algorithm(8.5, 8.75)
print("la raiz es",root)
print(bisection.comparision(8.5, 8.75))

bisection2 = BisectionAlgorithm(even_wavefunction2, E)
bisection2.initialPlot()
root2 = BisectionAlgorithm(even_wavefunction, Eb).algorithm(8.5, 8.75)
print(root)
print(bisection2.comparision(8.5, 8.75))

