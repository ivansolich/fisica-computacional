import matplotlib.pyplot as plt
import numpy as np


class DifferentiationApproximation:
    def __init__(self, func, true_derivative, t, h0, num_iterations=100):
        self.func = func
        self.true_derivative = true_derivative
        self.t = t
        self.h0 = h0
        self.num_iterations = num_iterations

    def machineEpsilon(self, f=float):
        machine_epsilon = f(1)
        while f(1) + machine_epsilon != f(1):
            machine_epsilon_last = machine_epsilon
            machine_epsilon = f(machine_epsilon) / f(2)
        return machine_epsilon_last

    def forward_difference(self, h):
        return (self.func(self.t + h) - self.func(self.t)) / h

    def central_difference(self, h):
        return (self.func(self.t + 0.5 * h) - self.func(self.t - 0.5 * h)) / h

    def relative_error(self, h):
        return abs((self.true_derivative(self.t) - self.forward_difference(h)) / self.true_derivative(self.t))

    """def calculate(self):
        h_values = []
        approx = []
        h = self.h0
        epsilon = self.machineEpsilon()
        while h < epsilon:
            approx.append(self.forward_difference(h))
            h_values.append(h)
            h = h / 2

        return approx, h_values"""

    def calculate(self):
        h_values = []
        approx = []
        error = []
        h = self.h0

        for i in range(self.num_iterations):
            if h > self.machineEpsilon():
                approx.append(self.forward_difference(h))
                h_values.append(h)
                error.append(self.relative_error(h))

                h = h / 2
            else:
                break

        approx = np.array(approx)
        error = np.array(error)
        h_values = np.array(h_values)
        return approx, h_values, error




def func_cos(t):
    return np.cos(t)


def dx(t):
    return -np.sin(t)


t = np.array([0.1, 1, 10])

diff_cos = DifferentiationApproximation(func_cos, dx, t[0], 0.01)
print(dx(0.1))
print(diff_cos.forward_difference(0.01))
print(diff_cos.relative_error(0.01))

approx, h,error = diff_cos.calculate()

print(h)
print(error)


ax, fig = plt.subplots()

plt.loglog(h, error, color="tab:red")
plt.set_cmap('plasma_r')
plt.xlabel(r'$h$')
plt.ylabel(r'Error relativo $|\epsilon|$')

plt.axhline(diff_cos.machineEpsilon(), color='red', linestyle='--')


plt.show()

