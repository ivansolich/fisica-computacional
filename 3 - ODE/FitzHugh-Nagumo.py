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
    plt.style.use('seaborn-v0_8-muted')

    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['font.family'] = 'STIXGeneral'

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    ax1.plot(t, v, label='Potencial de membrana (v)')
    ax1.plot(t, w, label='Variable de recuperación (w)', ls="--")
    ax1.set_title(f'Soluciones del sistema')
    ax1.set_xlabel('Tiempo')
    ax1.set_ylabel('v, w')
    ax1.legend()

    v_min, v_max = np.min(v), np.max(v)
    w_min, w_max = np.min(w), np.max(w)

    margin = 0.1 * (v_max - v_min)
    ax1.set_ylim(min(v_min, w_min) - margin, max(v_max, w_max) + margin)

    ax2.plot(v, w, label='Trayectoria')

    v_vals = np.linspace(v_min - margin, v_max + margin, 500)

    # Nullcline dv/dt = 0
    w_nullcline_v = v_vals * (v_vals - a) * (1 - v_vals) + I
    ax2.plot(v_vals, w_nullcline_v, label=r'$\frac{dv}{dt} = 0$', linestyle='--')

    # Nullcline dw/dt = 0
    w_nullcline_w = v_vals / gamma
    ax2.plot(v_vals, w_nullcline_w, label=r'$\frac{dw}{dt} = 0$', linestyle='--')

    """V, W = np.meshgrid(np.linspace(min(v), max(v), 15),
                       np.linspace(min(w), max(w), 15))
    dv, dw = f(V, W, a, I)
    ax2.quiver(V, W, dv, dw, color='gray', alpha=0.5)"""

    #ax2.axhline(y=0, color='black', linestyle=(0, (3, 1, 1, 1, 1, 1)), alpha=0.5)
    #ax2.axvline(x=0, color='black', linestyle=(0, (3, 1, 1, 1, 1, 1)), alpha=0.5)

    ax2.set_title(f'Diagrama de fases')
    ax2.set_xlabel('v')
    ax2.set_ylabel('w')

    margin_w = 0.1 * (np.max(w) - np.min(w))
    ax2.set_ylim(np.min(w) - margin_w, np.max(w) + margin_w)

    fig.tight_layout()

    fig.savefig('simulation_' + str(sim_number) + '.png')

    fig.show()


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
