import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

a = 0.8
b = 1
gamma = 0.5
Ia = 0  # I_a = 0 inicialmente


def derivadas(y, t, a, b, gamma, Ia):
    v, w = y
    dv_dt = v * (a - v) * (v - 1) - w + Ia
    dw_dt = b * v - gamma * w
    return np.array([dv_dt, dw_dt])


def RK4(derivadas, y0, t, a, b, gamma, Ia):
    h = t[1] - t[0]  # paso temporal
    sol = np.zeros((len(t), len(y0)))
    sol[0] = y0

    for i in range(1, len(t)):
        k1 = derivadas(sol[i - 1], t[i - 1], a, b, gamma, Ia)
        k2 = derivadas(sol[i - 1] + 0.5 * h * k1, t[i - 1] + 0.5 * h, a, b, gamma, Ia)
        k3 = derivadas(sol[i - 1] + 0.5 * h * k2, t[i - 1] + 0.5 * h, a, b, gamma, Ia)
        k4 = derivadas(sol[i - 1] + h * k3, t[i - 1] + h, a, b, gamma, Ia)
        sol[i] = sol[i - 1] + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    return sol


# Condiciones iniciales
y0 = [0.5, 0.0]  # [v0, w0]
t = np.linspace(0, 20, 1000)  # tiempo

# Solución numérica
sol = RK4(derivadas, y0, t, a, b, gamma, Ia)
v_sol, w_sol = sol[:, 0], sol[:, 1]

# Graficamos los resultados
plt.figure(figsize=(10, 6))
plt.plot(t, v_sol, label='v(t)', color='b')
plt.plot(t, w_sol, label='w(t)', color='r')
plt.xlabel("Tiempo")
plt.ylabel("v(t), w(t)")
plt.legend()
plt.show()

# Exploración de soluciones para diferentes parámetros
parametros = [
    {'a': 0.5, 'b': 0.2, 'gamma': 0.5},
    {'a': 1, 'b': 1.0, 'gamma': 0.6},
    {'a': 1.5, 'b': 1.7, 'gamma': 1}
]

plt.figure(figsize=(12, 8))

for idx, params in enumerate(parametros):
    sol = RK4(derivadas, y0, t, params['a'], params['b'], params['gamma'], Ia)
    v_sol, w_sol = sol[:, 0], sol[:, 1]

    plt.subplot(3, 1, idx + 1)
    plt.plot(t, v_sol, label=f"v(t) con a={params['a']}, b={params['b']}, gamma={params['gamma']}")
    plt.plot(t, w_sol, label=f"w(t) con a={params['a']}, b={params['b']}, gamma={params['gamma']}")
    plt.xlabel("Tiempo")
    plt.ylabel("v(t), w(t)")
    plt.legend()

plt.tight_layout()
plt.show()

Ia = 0.5  # Añadimos una corriente constante

sol_con_corriente = RK4(derivadas, y0, t, a, b, gamma, Ia)
v_sol_corriente, w_sol_corriente = sol_con_corriente[:, 0], sol_con_corriente[:, 1]

plt.figure(figsize=(10, 6))
plt.plot(t, v_sol_corriente, label='v(t) con Ia = 0.5', color='b')
plt.plot(t, w_sol_corriente, label='w(t) con Ia = 0.5', color='r')
plt.xlabel("Tiempo")
plt.ylabel("v(t), w(t)")
plt.legend()
plt.show()
