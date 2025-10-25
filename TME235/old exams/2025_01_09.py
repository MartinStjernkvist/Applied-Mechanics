#%%
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

#%%
# 1

a, b = sp.symbols('a b')

l1 = sp.Matrix([
    [sp.cos(b), 0, -sp.sin(b)],
    [0, 1 , 0],
    [sp.sin(b), 0, sp.cos(b)]
])

l2 = sp.Matrix([
    [sp.cos(a), sp.sin(a), 0],
    [-sp.sin(a), sp.cos(a), 0],
    [0,0,1]
])

l = l2 * l1
l

acc = sp.Matrix([
    [1],
    [2],
    [-2]
])

acc_new = l * acc
acc_new

l_f = sp.lambdify((a, b), l, 'numpy')
l_num = l_f(60 * (np.pi / 180), 20 *(np.pi / 180))

acc_new_values = sp.lambdify((a, b), acc_new, 'numpy')
acc_new_values(60 * (np.pi / 180), 20 *(np.pi / 180))

#%%
# 2

sigma_new = np.array([
    [150, 70, 0],
    [70, 50, 15],
    [0, 15, 100]
])
sigma = np.linalg.inv(l_num) @ sigma_new @ np.linalg.inv(l_num.T)
print('sigma:')
print(sigma)

eigvals, _ = np.linalg.eig(sigma)
print('\neigvals:')
print(eigvals)


#%%
# 4

x, E, I, G, Ks, A, M_hat, P_hat, L, C1, C2, C3, C4 = sp.symbols('x E I G Ks A M_hat P_hat L C1 C2 C3 C4', real=True)

phi = 1/(E * I) * (C1 * x**2 /2 + C2 * x + C3)

w = sp.integrate(phi + C4 / (G * Ks * A), x)
print(w)

M = - E * I * phi.diff(x)
# V = G * Ks * A * (-phi + w.diff(x))
V = M.diff(x)

boundary_conditions = [
    w.subs(x, 0),
    w.diff(x).subs(x, 0),
    M.subs(x, L) - M_hat,
    V.subs(x, L) - P_hat
]

unknowns = (C1, C2, C3, C4)

integration_constants = sp.solve(boundary_conditions, unknowns, real=True)
print(integration_constants)

w_sol = w.subs(integration_constants)
print("w after substitution:")
print(w_sol.simplify())

w_f = sp.lambdify((x, E, I, G, Ks, A, M_hat, P_hat, L), w_sol, 'numpy')

E_num =80e9
b_num = 50e-3
h_num = 30e-3
nu_num = 0.2
I_num = b_num * h_num **3 / 12
G_num = E_num / (2 * (1 + nu_num))
A_num = b_num * h_num
num = w_f(x=1, E=E_num, I=I_num, G=G_num, Ks=5/6, A=A_num, M_hat=500, P_hat=2e3, L=1)
print(num)
