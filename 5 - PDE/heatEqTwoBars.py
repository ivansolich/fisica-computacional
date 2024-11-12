import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class HeatEquation:
    def __init__(self, KAPPA, SPH, RHO, stepFactor, h=0, Te=0):
        # Parámetros dados
        self.stepFactor = stepFactor
        self.L = 0.25  # Longitud de cada barra en metros
        self.Nx = 101  # Número de divisiones espaciales por barra (100 divisiones más un extremo)
        self.Dx = self.L / (self.Nx - 1)  # Tamaño del paso espacial
        self.KAPPA = KAPPA  # Conductividad térmica en W/(m K)
        self.SPH = SPH  # Calor específico en J/(kg K)
        self.RHO = RHO  # Densidad en kg/m^3
        self.h = h  # Coeficiente de transferencia de calor para enfriamiento de Newton
        self.Te = Te  # Temperatura ambiente
        self.Dt = 0.3 * (self.SPH * self.RHO * self.Dx * self.Dx) / self.KAPPA  # Paso temporal para estabilidad
        self.Tpl = np.zeros((2 * self.Nx - 1, 31))  # Matriz para guardar temperaturas en diferentes momentos

    def calculate(self):
        const = (self.KAPPA / (self.SPH * self.RHO)) * (self.Dt / (self.Dx * self.Dx))
        print(f"Constante calculada: {const} (debería ser <= 0.5 para estabilidad)")

        T = np.zeros((2 * self.Nx - 1, 2))  # Temperatura en cada punto espacial para dos tiempos (presente y pasado)

        # Condiciones iniciales: barra izquierda a 100 K, barra derecha a 50 K
        for i in range(1, self.Nx - 1):
            T[i, 0] = 100  # Barra izquierda
        for i in range(self.Nx, 2 * self.Nx - 2):
            T[i, 0] = 50
        T[0, 0] = T[0, 1] = 0
        T[-1, 0] = T[-1, 1] = 0

        # Proceso iterativo para resolver la ecuación de calor con contacto y enfriamiento
        print("Working, wait for figure after count to 10")
        m = 0  # Contador para filas, una cada 450 pasos de tiempo
        for t in range(1, int(self.stepFactor * 1000 * 0.9 / self.Dt)):
            for i in range(1, 2 * self.Nx - 2):
                if i == self.Nx - 1:  # Punto de contacto entre las dos barras
                    T[i, 1] = 0.5 * (T[i + 1, 0] + T[i - 1, 0])  # Promedio entre los dos lados en contacto
                else:
                    # Ecuación de calor con enfriamiento de Newton en ambos extremos
                    T[i, 1] = T[i, 0] + const * (T[i + 1, 0] + T[i - 1, 0] - 2 * T[i, 0])
                    if self.h > 0:
                        T[i, 1] -= self.h * (T[i, 0] - self.Te) * self.Dt / (self.SPH * self.RHO)

            # Guardar los resultados cada 450 pasos de tiempo o en el primer paso
            if t % (self.stepFactor * 100) == 0 or t == 1:
                if m < 31:  # Verificar que m no exceda el tamaño de la matriz Tpl
                    for i in range(1, 2 * self.Nx - 2, 2):  # Guardar solo cada segundo punto para reducir la cantidad de datos
                        self.Tpl[i, m] = T[i, 1]
                    print(m)
                    m += 1

            # Actualizar los valores de temperatura para el siguiente paso de tiempo
            for i in range(1, 2 * self.Nx - 2):
                T[i, 0] = T[i, 1]

    def plotting(self):
        x = np.linspace(0, 2 * self.L, 2 * self.Nx - 1)[1::2]  # Posiciones espaciales, solo cada segundo punto
        y = np.linspace(0, int(self.stepFactor * 1000 * 0.9 / self.Dt) * self.Dt, 31)  # Tiempos seleccionados
        X, Y = np.meshgrid(x, y)
        Z = self.Tpl[1::2, :].T  # Matriz de temperatura transpuesta para la gráfica

        # Graficar superficie de temperatura en función de posición y tiempo
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_wireframe(X, Y, Z, color='r')
        ax.set_xlabel('Posición (m)')
        ax.set_ylabel('Tiempo (s)')
        ax.set_zlabel('Temperatura (K)')
        plt.show()

        # Graficar mapa de calor 2D de posición vs tiempo
        plt.figure()
        plt.contourf(X, Y, Z, 20, cmap='hot')
        plt.colorbar(label='Temperatura (K)')
        plt.xlabel('Posición (m)')
        plt.ylabel('Tiempo (s)')
        plt.title('Mapa de calor de temperatura')
        plt.show()

        # Graficar superficie de temperatura en función de posición y tiempo con mapa de calor 3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        heatmap = ax.plot_surface(X, Y, Z, cmap='afmhot', edgecolor='none')
        fig.colorbar(heatmap, ax=ax, shrink=0.5, aspect=5, label='Temperatura (K)')
        ax.set_xlabel('Posición (m)')
        ax.set_ylabel('Tiempo (s)')
        ax.set_zlabel('Temperatura (K)')
        plt.title('Mapa de calor 3D de la temperatura')
        plt.show()

        # Gráfica de contornos (isotermas)
        plt.figure()
        cp = plt.contour(X, Y, Z, levels=10, cmap='coolwarm')
        plt.clabel(cp, inline=True, fontsize=8)
        plt.xlabel('Posición (m)')
        plt.ylabel('Tiempo (s)')
        plt.title('Isotermas de temperatura')
        plt.show()

barraFe = HeatEquation(80.4, 449, 7874, 6, h=0.1, Te=20)
barraFe.calculate()
barraFe.plotting()
