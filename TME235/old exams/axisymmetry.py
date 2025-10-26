#%%
# Computer assignment 3 träning
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

r, nu, E, h, q, a, b = sp.symbols('r nu E h q a b', real=True)

D = E * h**3 / (12 * (1 - nu**2))

w = sp.Function('w')(r)

diffeq = sp.Eq(1/r * sp.diff(r * sp.diff(1 / r * sp.diff(r * sp.diff(w, r), r), r), r), q / D)

w = sp.dsolve(diffeq, w)
w = w.rhs
print('\nw:')
print(w)

# Formula sheet:
M_r = D * (- w.diff(r, 2) - nu * w.diff(r) / r)
M_phi = D * (- nu * w.diff(r, 2) - w.diff(r) / r)

V = 1/r * (sp.diff(r * M_r, r) - M_phi)

boundary_conditions = [
    w.subs(r, b),
    M_r.subs(r, b),
    M_r.subs(r, a),
    V.subs(r, a)
]
unknowns = ('C1, C2, C3, C4')
integration_constants = sp.solve(boundary_conditions, unknowns, real=True)
print('\nintegration constants:')
print(integration_constants)

w_sol = w.subs(integration_constants)
print('w_sol:')
print(w_sol)
w_f = sp.lambdify((r, nu, E, h, q, a, b), w_sol, 'numpy')

r_vals = np.linspace(0.1, 0.3, 500)

nu_num = 0.3
E_num = 210e9
q_num = 15e6
a_num = 0.1
b_num = 0.3
h_num = a_num/4

w_vals = w_f(r=r_vals, nu=nu_num, E=E_num, h=h_num, q=q_num, a=a_num, b=b_num)

plt.figure()
plt.plot(r_vals, w_vals)
plt.show()