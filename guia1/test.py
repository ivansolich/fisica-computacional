import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def f(x):
    return np.exp(x) * np.log(x) - x ** 2


def derivada(x):
    return np.exp(x) * np.log(x) + np.exp(x) + 1 / x - 2 * x


def Newton_Raphson(f, dx, x0, N, tol):
    x_values = [x0]
    x = x0
    for i in range(N):
        x_new = x - f(x) / dx(x)

        x_values.append(x_new)

        if abs(x_new - x) < tol:
            break

        x = x_new
    return x_values


N = 100000
x_values = Newton_Raphson(f, derivada, 2, N, 1e-6)
raiz = x_values[-1]

print(raiz)


x = np.linspace(0.5, 2.5, 1000)


fig, ax = plt.subplots()
ax.plot(x, f(x), label='f(x)')
ax.axhline(y=0, color='grey', linestyle='--', linewidth=1)


# Inicializar el punto rojo que muestra la raíz
point, = ax.plot([], [], 'ro', label='Raíz estimada')

# Función para inicializar la animación
def init():
    point.set_data([], [])
    return point,

# Función para actualizar la animación
def update(frame):
    x_val = x_values[frame]
    y_val = f(x_val)
    point.set_data([x_val], [y_val])  # Usamos listas para las coordenadas x e y
    return point,


ani = FuncAnimation(fig, update, frames=len(x_values), init_func=init, blit=True, repeat=False)


plt.legend()
plt.show()
