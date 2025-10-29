#%%
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

X1, X2, X3, x1, x2, x3, t = sp.symbols('X1 X2 X3 x1 x2 x3 t')

x = sp.Matrix([X1 + X2 /4 * sp.sin(sp.pi * t) + X3 *t, X2 * (1 + t/8), X3 * (1 - sp.sin(2 * sp.pi * t)/6)])

X = sp.Matrix([X1, X2, X3])

F = x.jacobian(X)
F