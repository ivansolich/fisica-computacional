import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class heatEquation:
    def __init__(self, KAPPA, SPH, RHO, stepFactor):
        # Parámetros dados
        self.stepFactor = stepFactor
        self.L = 1.0  # Longitud de la barra en metros
        self.Nx = 101  # Número de divisiones espaciales (100 divisiones más un extremo)
        self.Dx = self.L / (self.Nx - 1)  # Tamaño del paso espacial
        self.KAPPA = KAPPA  # Conductividad térmica en W/(m K)
        self.SPH = SPH  # Calor específico en J/(kg K)
        self.RHO = RHO  # Densidad en kg/m^3
        self.Dt = 0.3 * (self.SPH * self.RHO * self.Dx * self.Dx) / self.KAPPA  # Paso temporal para estabilidad
        self.Tpl = np.zeros((self.Nx, 31))  # Matriz para guardar temperaturas en diferentes momentos
        self.T = np.zeros((self.Nx, 2))  # Temperatura en cada punto espacial para dos tiempos (presente y pasado)

    def calculate(self):
        # Constante para el cálculo de diferencias finitas
        const = (self.KAPPA / (self.SPH * self.RHO)) * (self.Dt / (self.Dx * self.Dx))
        print(f"Constante calculada: {const} (debería ser <= 0.5 para estabilidad)")

        # Inicialización de la matriz de temperatura
        for i in range(1, self.Nx - 1):
            self.T[i, 0] = 100  # Inicializar con 100 K en el interior de la barra
        self.T[0, 0] = self.T[0, 1] = 0  # Extremo izquierdo a 0 K
        self.T[self.Nx - 1, 0] = self.T[self.Nx - 1, 1] = 0  # Extremo derecho a 0 K

        # Proceso iterativo para resolver la ecuación de calor
        m = 0  # Contador para filas, una cada 450 pasos de tiempo
        for t in range(1, int(self.stepFactor * 1000 * 0.9 / self.Dt)):
            for i in range(1, self.Nx - 1):
                self.T[i, 1] = self.T[i, 0] + const * (self.T[i + 1, 0] + self.T[i - 1, 0] - 2 * self.T[i, 0])

            # Guardar los resultados cada 450 pasos de tiempo o en el primer paso
            if t % (self.stepFactor * 100) == 0 or t == 1:
                for i in range(1, self.Nx - 1, 2):  # Guardar solo cada segundo punto para reducir la cantidad de datos
                    self.Tpl[i, m] = self.T[i, 1]
                m += 1

            # Actualizar los valores de temperatura para el siguiente paso de tiempo
            for i in range(1, self.Nx - 1):
                self.T[i, 0] = self.T[i, 1]

    def plotting(self):
        # Crear una malla para graficar
        x = np.linspace(0, self.L, self.Nx)[1::2]  # Posiciones espaciales, solo cada segundo punto
        y = np.linspace(0, int(self.stepFactor * 1000 * 0.9 / self.Dt) * self.Dt, 31)  # Tiempos seleccionados
        X, Y = np.meshgrid(x, y)  # Crear malla de posición y tiempo
        Z = self.Tpl[1::2, :].T  # Matriz de temperatura transpuesta para la gráfica

        # Graficar superficie de temperatura en función de posición y tiempo
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_wireframe(X, Y, Z, color='r')
        ax.set_xlabel('Posición (m)')
        ax.set_ylabel('Tiempo (s)')
        ax.set_zlabel('Temperatura (K)')
        fig.tight_layout()
        ax.view_init(elev=22, azim=51)
        plt.savefig('sistemaA\Mapa3D.png', dpi=300)
        plt.show()

        # Graficar mapa de calor 2D de posición vs tiempo
        plt.figure()
        plt.contourf(X, Y, Z, 20, cmap='hot')
        plt.colorbar(label='Temperatura (K)')
        plt.xlabel('Posición (m)')
        plt.ylabel('Tiempo (s)')
        #plt.title('Mapa de calor de temperatura')
        fig.tight_layout()
        plt.savefig('sistemaA\MapaCalor2D.png', dpi=300)
        plt.show()

        # Graficar superficie de temperatura en función de posición y tiempo con mapa de calor 3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        heatmap = ax.plot_surface(X, Y, Z, cmap='hot', edgecolor='none')
        fig.colorbar(heatmap, ax=ax, shrink=0.5, aspect=5, label='Temperatura (K)')
        ax.set_xlabel('Posición (m)')
        ax.set_ylabel('Tiempo (s)')
        ax.set_zlabel('Temperatura (K)')
        #plt.title('Mapa de calor 3D de la temperatura')
        fig.tight_layout()
        ax.view_init(elev=22, azim=51)
        plt.savefig("sistemaA\MapaCalor3D.png", dpi=300)
        plt.show()

        # Gráfica de contornos (isotermas)
        plt.figure()
        cp = plt.contour(X, Y, Z, levels=10, cmap='coolwarm')
        plt.clabel(cp, inline=True, fontsize=8)
        plt.xlabel('Posición (m)')
        plt.ylabel('Tiempo (s)')
        #plt.title('Isotermas de temperatura')
        fig.tight_layout()
        plt.savefig("sistemaA\Isoterma2D.png", dpi=300)
        plt.show()

    def plot_theoretical_solution(self):
        # Crear malla para la solución teórica
        x = np.linspace(0, self.L, self.Nx)[1::2]  # Posiciones espaciales
        y = np.linspace(0, int(self.stepFactor * 1000 * 0.9 / self.Dt) * self.Dt, 31)  # Tiempos seleccionados
        X, Y = np.meshgrid(x, y)

        # Calcular la solución teórica en cada punto (x, t)
        Z_theoretical = np.sin((np.pi * X) / self.L) * np.exp((-np.pi ** 2 * self.KAPPA * Y) / (self.L ** 2 * self.RHO * self.SPH))

        # Graficar la solución teórica en una superficie 3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, Z_theoretical, cmap='hot', edgecolor='none')
        ax.set_xlabel('Posición (m)')
        ax.set_ylabel('Tiempo (s)')
        ax.set_zlabel('Temperatura (K)')
        #plt.title('Solución teórica de la temperatura')
        fig.tight_layout()
        ax.view_init(elev=22, azim=51)
        plt.savefig('sistemaA\Mapa3D_Teorico.png', dpi=300)
        plt.show()

        # Graficar mapa de calor 2D de la solución teórica
        plt.figure()
        plt.contourf(X, Y, Z_theoretical, 20, cmap='hot')
        plt.colorbar(label='Temperatura (K)')
        plt.xlabel('Posición (m)')
        plt.ylabel('Tiempo (s)')
        #plt.title('Mapa de calor teórico de la temperatura')
        fig.tight_layout()
        plt.savefig('sistemaA\Mapa2D_Teorico.png', dpi=300)
        plt.show()

    def plot_relative_error(self):
        # Calcular el valor teórico en t = 1000 segundos
        x = np.linspace(0, self.L, self.Nx)[1::2]
        T_theoretical = np.sin((np.pi * x) / self.L) * np.exp((-np.pi ** 2 * self.KAPPA * 1000) / (self.L ** 2 * self.RHO * self.SPH))

        # Obtener el valor numérico en t = 1000 segundos de la matriz Tpl
        T_numerical = self.Tpl[1::2, -1]

        # Calcular el error relativo
        relative_error = np.abs((T_numerical - T_theoretical) / T_theoretical) * 100

        # Graficar el error relativo
        plt.figure()
        plt.plot(x, relative_error, 'b-', label="Error relativo (%)")
        plt.xlabel("Posición (m)")
        plt.ylabel("Error relativo (%)")
        plt.title("Error relativo en t = 1000 segundos")
        plt.legend()
        plt.grid(True)
        plt.savefig("ErrorRelativo.png", dpi=300)
        plt.show()

# Crear instancia y calcular la ecuación de calor para el aluminio
barraAl = heatEquation(237, 900, 2700, 4.5)
barraAl.calculate()
barraAl.plotting()
barraAl.plot_theoretical_solution()
