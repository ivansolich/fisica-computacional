import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

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

def animate_simulation(t, v, w, a, I, sim_number, skip=10):

    plt.style.use('seaborn-v0_8-muted')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))  # Reducir el tamaño de la figura

    ax1.set_title(f'Modelo FitzHugh-Nagumo - Simulación {sim_number}\n(a={a}, I={I})')
    ax1.set_xlabel('Tiempo')
    ax1.set_ylabel('v, w')
    ax1.set_xlim(0, t.max())
    ax1.set_ylim(min(min(v), min(w)) - 0.1, max(max(v), max(w)) + 0.1)

    ax2.set_title(f'Espacio de fases - Simulación {sim_number}\n(a={a}, I={I})')
    ax2.set_xlabel('v')
    ax2.set_ylabel('w')
    ax2.set_xlim(min(v) - 0.1, max(v) + 0.1)
    ax2.set_ylim(min(w) - 0.1, max(w) + 0.1)

    line_v, = ax1.plot([], [], label='Potencial de membrana (v)')
    line_w, = ax1.plot([], [], label='Variable de recuperación (w)', linestyle='--')
    ax1.legend()

    line_phase, = ax2.plot([], [], color='orange', label='Diagrama de fase (v vs w)')
    ax2.legend()

    plt.tight_layout()

    def init():
        line_v.set_data([], [])
        line_w.set_data([], [])
        line_phase.set_data([], [])
        return line_v, line_w, line_phase

    def update(frame):
        frame = frame * skip  # Ajustar para saltar pasos
        line_v.set_data(t[:frame], v[:frame])
        line_w.set_data(t[:frame], w[:frame])
        line_phase.set_data(v[:frame], w[:frame])
        return line_v, line_w, line_phase

    anim = FuncAnimation(fig, update, frames=len(t)//skip, init_func=init, blit=True, interval=10)

    anim.save('fhn_simulation.gif', writer='pillow', fps=15)  # Reducir FPS para mayor velocidad

    plt.tight_layout()
    plt.show()


a3 = -0.1
I3 = 0
t_max = 300  # Reducir el tiempo máximo
dt = 0.2    # Aumentar el tamaño del paso

# Simulación 3
t3, v3, w3 = simulate_system(t_max=t_max, dt=dt, a=a3, I=I3)


animate_simulation(t3, v3, w3, a=a3, I=I3, sim_number=3, skip=10)  # Saltar cada 10 pasos
