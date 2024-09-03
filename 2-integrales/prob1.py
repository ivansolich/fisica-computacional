import numpy as np
import matplotlib.pyplot as plt


class Integration():
    def __init__(self, f, a, b, exact_value, N_max):
        self.f = f
        self.a = a
        self.b = b
        self.exact_value = exact_value
        self.N_max = N_max

    def trapezoid_rule(self, N):
        h = (self.b - self.a) / N
        integral = 0.5 * (self.f(self.a) + self.f(self.b))
        for i in range(1, N):
            integral += self.f(self.a + i * h)
        integral *= h
        return integral

    # Regla de Simpson
    def simpson_rule(self, N):
        #if N % 2 == 1:
        #    N += 1
        h = (self.b - self.a) / N
        integral = self.f(self.a) + self.f(self.b)
        for i in range(1, N, 2):
            integral += 4 * self.f(self.a + i * h)
        for i in range(2, N - 1, 2):
            integral += 2 * self.f(self.a + i * h)
        integral *= h / 3
        return integral

    def relative_error(self):
        error_t = []
        error_s = []
        N = np.array([2])

        i = 10

        while i <= self.N_max:
            N = np.append(N, i)
            i = i * 2

        for i in range(len(N)):
            T = self.trapezoid_rule(N[i])
            S = self.simpson_rule(N[i])
            error_t.append(abs((T - exact_value) / exact_value))
            error_s.append(abs((S - exact_value) / exact_value))

        error_t = np.array(error_t)
        error_s = np.array(error_s)

        return error_t, error_s, N

    def plot_error(self):
        error_t, error_s, N = self.relative_error()

        plt.rcParams['mathtext.fontset'] = 'stix'
        plt.rcParams['font.family'] = 'STIXGeneral'
        plt.style.use('seaborn-v0_8-deep')

        plt.loglog(N, error_t,label='Regla del Trapecio')
        plt.loglog(N, error_s,label='Regla de Simpson')

        plt.xlabel('N')
        plt.ylabel('| error |')
        plt.legend(loc='best')
        plt.grid(True)

        plt.tight_layout()

        plt.show()


def f(t):
    return np.exp(-t)


exact_value = 1 - np.exp(-1)

int = Integration(f, 0, 1, exact_value, 1000000)
int.plot_error()