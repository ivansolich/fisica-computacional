import numpy as np
import scipy as sp
import matplotlib.pyplot as plt


class QuantumSquareWell:
    def __init__(self, V0, a=1, hbar=1, m=0.5):
        self.V0 = V0
        self.a = a
        self.hbar = hbar
        self.m = m
        self.unit_energy = self.V0 / 10  # Energía característica

    def even_wavefunction(self, E_B):
        return np.sqrt(self.unit_energy - E_B) * np.tan(np.sqrt(self.unit_energy - E_B)) - np.sqrt(E_B)

    def find_bound_state_energies(self, initial_guesses):
        """Encuentra las energías de los estados ligados resolviendo la ecuación trascendental."""
        E_B_solutions = []
        for guess in initial_guesses:
            root = sp.optimize.fsolve(self.even_wavefunction, guess)[0]
            if root > 0 and root not in E_B_solutions:
                E_B_solutions.append(root)
        return E_B_solutions

    def plot_wavefunction(self, E_B_values):
        """Grafica la función trascendental para visualizar las raíces."""
        f_values = self.even_wavefunction(E_B_values)
        plt.plot(E_B_values, f_values, label='f(E_B)')
        plt.axhline(0, color='red', linestyle='--')
        plt.xlabel('E_B')
        plt.ylabel('f(E_B)')
        plt.title('Raíces de la función trascendental')
        plt.legend()
        plt.show()


# Crear una instancia de la clase con V0 = 10
quantum_well = QuantumSquareWell(V0=20)

# Definir los valores iniciales para E_B
initial_guesses = np.linspace(0.1, 9.9, 5)

# Encontrar las energías de los estados ligados
bound_state_energies = quantum_well.find_bound_state_energies(initial_guesses)

# Imprimir las energías encontradas
print("Energías de los estados ligados (pares):", bound_state_energies)

# Graficar la función para visualizar las raíces
E_B_values = np.linspace(0.1, 9.9, 400)
quantum_well.plot_wavefunction(E_B_values)