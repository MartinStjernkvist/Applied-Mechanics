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

#%%
# 3

x, Ks, G, E, I, A, q0, L = sp.symbols('x Ks G E I A q0 L', real = True)

q = -q0 * x / L

phi = sp.Function('phi')(x)

diffeq_phi = sp.Eq(E * I * phi.diff(x, 3), q)
phi = sp.dsolve(diffeq_phi, phi).rhs

w = sp.Function('w')(x)

diffeq_w = sp.Eq(w.diff(x), - E * I / (G * Ks * A) * phi.diff(x,2) + phi)
w = sp.dsolve(diffeq_w, w).rhs
print('\nw:')
print(w)

# phi = - 1 / (E * I) * q * x**3 / 6 + C1 * x **2 / 2 + C2 * x + C3

M = - E * I * phi.diff(x)

boundary_conditions = [
    w.subs(x, 0),
    w.subs(x, L),
    M.subs(x, 0),
    w.diff(x).subs(x, L)
]

unknowns = ('C1, C2, C3, C4')
integration_constants = sp.solve(boundary_conditions, unknowns)
print('\nintegration constants:')
print(integration_constants)
w_sol = w.subs(integration_constants)

w_f = sp.lambdify((x, Ks, G, E, I, A, q0, L), w_sol, 'numpy')

L_num = 1
E_num = 210e9
G_num = 81e9
I_num = 833e-12
Ks_num = 5/6
A_num = 100e-6
q0_num = 1

x_vals = np.linspace(0, L_num, 500)
w_vals = w_f(x=x_vals, Ks=Ks_num, G=G_num, E=E_num, I=I_num, A=A_num, q0=q0_num, L=L_num)

plt.figure()
plt.plot(x_vals, w_vals)
plt.show()

w_max = np.max(w_vals)
print('\nmax deflection w [mm]:')
print(w_max*10e3)