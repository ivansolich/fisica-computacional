import numpy as np
import matplotlib.pyplot as plt

def suma_ascendente(N):
    """Calcula la suma ascendente S^(up)"""
    return np.sum(1.0 / np.arange(1, N + 1))

def suma_descendente(N):
    """Calcula la suma descendente S^(down)"""
    return np.sum(1.0 / np.arange(N, 0, -1))

def calcular_diferencia_relativa(N):
    """Calcula la diferencia relativa entre las sumas ascendentes y descendentes"""
    S_up = suma_ascendente(N)
    S_down = suma_descendente(N)
    diferencia_relativa = np.abs(S_up - S_down) / (np.abs(S_up) + np.abs(S_down))
    return diferencia_relativa

# Rango de valores de N
N_values = np.arange(1, 100)  # Desde 1 hasta 10000

# Calcular la diferencia relativa para cada valor de N
diferencias_relativas = [calcular_diferencia_relativa(N) for N in N_values]

# Crear el gráfico log-log
plt.figure(figsize=(10, 6))
plt.loglog(N_values, diferencias_relativas, label='Diferencia Relativa')

# Añadir títulos y etiquetas
plt.title('Diferencia relativa entre Suma Ascendente y Descendente')
plt.xlabel('N')
plt.ylabel('Diferencia Relativa')
plt.grid(True, which="both", ls="--")
plt.legend()
plt.show()
