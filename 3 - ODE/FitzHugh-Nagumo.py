import numpy as np
import matplotlib.pyplot as plt

# Definimos los parámetros del sistema
a = -0.1
b = 0.5
I = 0.5
epsilon = 0.01


# Definimos las ecuaciones del sistema
def f(v, w):
    dvdt = v * (v - a) * (1-v) - w + I
    dwdt =  epsilon * (v - b * w)
    return dvdt, dwdt


# Implementación del método RK4
def rk4_step(v, w, dt):
    # k1
    dv1, dw1 = f(v, w)
    k1_v = dv1 * dt
    k1_w = dw1 * dt

    # k2
    dv2, dw2 = f(v + 0.5 * k1_v, w + 0.5 * k1_w)
    k2_v = dv2 * dt
    k2_w = dw2 * dt

    # k3
    dv3, dw3 = f(v + 0.5 * k2_v, w + 0.5 * k2_w)
    k3_v = dv3 * dt
    k3_w = dw3 * dt

    # k4
    dv4, dw4 = f(v + k3_v, w + k3_w)
    k4_v = dv4 * dt
    k4_w = dw4 * dt

    # Actualizar los valores de v y w
    v_next = v + (k1_v + 2 * k2_v + 2 * k3_v + k4_v) / 6
    w_next = w + (k1_w + 2 * k2_w + 2 * k3_w + k4_w) / 6

    return v_next, w_next


# Simulación con RK4
def simulate_system(t_max, dt):
    t = np.arange(0, t_max, dt)
    v = np.zeros(len(t))
    w = np.zeros(len(t))

    # Condiciones iniciales
    v[0] = 0.1
    w[0] = 0

    # Bucle de integración
    for i in range(1, len(t)):
        v[i], w[i] = rk4_step(v[i - 1], w[i - 1], dt)

    return t, v, w


# Parámetros de simulación
t_max = 800
dt = 0.1

# Simulamos el sistema
t, v, w = simulate_system(t_max=t_max, dt=dt)

# Graficamos los resultados
plt.figure()
plt.plot(t, v, label='Potencial de membrana (v)')
plt.plot(t, w, label='Variable de recuperación (w)', ls="--")
plt.title('Modelo FitzHugh-Nagumo')
plt.xlabel('Tiempo')
plt.ylabel('v, w')
plt.legend()
plt.show()


plt.figure()
plt.plot(v,w)
plt.show()
