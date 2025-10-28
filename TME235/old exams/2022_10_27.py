#%%
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

#%%
# 1

v01, v02, t, H, mu, X2 = sp.symbols('v01 v02 t H mu X2', real=True)

F = sp.Matrix([
    [1, v01 * t / (5 * H), 0],
    [0, 1 + v02 * t / (5 * H), 0],
    [0, 0, 1]
])

E = 1/2 *(sp.transpose(F) * F - sp.eye(3))


dv_dx = F.diff(t) * F.inv()

D = 1/2 * (sp.transpose(dv_dx) + dv_dx)
D_dev = D - sp.eye(3) * sp.trace(D) / 3

sigma_dev = 2 * mu * D_dev

sigma_dev_f = sp.lambdify((v01, v02, t, H, mu, X2), sigma_dev, 'numpy')

t_num = 1
v01_num = 1
v02_num = 2
H_num = 3
mu_num = 1.8

eigvals, _ = np.linalg.eig(sigma_dev_f(v01_num, v02_num, t_num, H_num, mu_num, 1))


print(np.max(eigvals))
