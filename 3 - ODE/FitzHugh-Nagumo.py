import numpy as np
import matplotlib.pyplot as plt

gamma = 0.5
epsilon = 0.01


def f(v, w, a, I):
    dvdt = v * (v - a) * (1 - v) - w + I
    dwdt = epsilon * (v - gamma * w)
    return dvdt, dwdt


def rk4_step(v, w, dt, a, I):
    dv1, dw1 = f(v, w, a, I)
    k1_v = dv1 * dt
    k1_w = dw1 * dt

    dv2, dw2 = f(v + 0.5 * k1_v, w + 0.5 * k1_w, a, I)
    k2_v = dv2 * dt
    k2_w = dw2 * dt

    dv3, dw3 = f(v + 0.5 * k2_v, w + 0.5 * k2_w, a, I)
    k3_v = dv3 * dt
    k3_w = dw3 * dt

    dv4, dw4 = f(v + k3_v, w + k3_w, a, I)
    k4_v = dv4 * dt
    k4_w = dw4 * dt

    v_next = v + (k1_v + 2 * k2_v + 2 * k3_v + k4_v) / 6
    w_next = w + (k1_w + 2 * k2_w + 2 * k3_w + k4_w) / 6

    return v_next, w_next


def simulate_system(t_max, dt, a, I):
    t = np.arange(0, t_max, dt)
    v = np.zeros(len(t))
    w = np.zeros(len(t))

    v[0] = 0.1
    w[0] = 0

    for i in range(1, len(t)):
        v[i], w[i] = rk4_step(v[i - 1], w[i - 1], dt, a, I)

    return t, v, w


def plot_simulation(t, v, w, a, I, sim_number):

    plt.style.use('seaborn-v0_8-dark')

    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['font.family'] = 'STIXGeneral'

    plt.figure()
    plt.plot(t, v, label='Potencial de membrana (v)')
    plt.plot(0,0)
    plt.plot(0, 0)
    plt.plot(t, w, label='Variable de recuperación (w)', ls="--")
    plt.title(f'Modelo FitzHugh-Nagumo - Simulación {sim_number}\n(a={a}, I={I})')
    plt.xlabel('Tiempo')
    plt.ylabel('v, w')
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(v, w)
    plt.title(f'Espacio de fases - Simulación {sim_number}\n(a={a}, I={I})')
    plt.xlabel('v')
    plt.ylabel('w')
    plt.show()


t_max = 500
dt = 0.1

# Simulación 1
a1 = -0.1
I1 = 0.0
t1, v1, w1 = simulate_system(t_max=t_max, dt=dt, a=a1, I=I1)

# Simulación 2
a2 = 0.1
I2 = 0.0
t2, v2, w2 = simulate_system(t_max=t_max, dt=dt, a=a2, I=I2)

# Simulación 3
a3 = -0.1
I3 = 0.5
t3, v3, w3 = simulate_system(t_max=t_max, dt=dt, a=a3, I=I3)

plot_simulation(t1, v1, w1, a1, I1, sim_number=1)
plot_simulation(t2, v2, w2, a2, I2, sim_number=2)
plot_simulation(t3, v3, w3, a3, I3, sim_number=3)
