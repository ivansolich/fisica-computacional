import numpy as np
import matplotlib.pyplot as plt

def cuadratic_formula(a,b,c):
    x1 = (-b + np.sqrt(b**2 - 4*a*c))/(2*a)
    x2 = (-b - np.sqrt(b**2 - 4*a*c))/(2*a)
    return x1, x2

def cuadratic_formula2(a,b,c):
    x1 = (-2 * c) / (b + np.sqrt(b**2 - 4*a*c))
    x2 = (-2 * c) / (b - np.sqrt(b**2 - 4*a*c))
    return x1, x2