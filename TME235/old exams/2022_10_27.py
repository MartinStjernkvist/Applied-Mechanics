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
# 3a

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


#%%
# 3b

x, Ks, G, E, I, A, q0, L = sp.symbols('x Ks G E I A q0 L', real=True, positive=True)
C1, C2, C3, C4, C5, C6, C7, C8 = sp.symbols('C1 C2 C3 C4 C5 C6 C7 C8', real=True)

# Region 1: 0 <= x <= L/2 with distributed load q = -q0
phi1 = -q0*x**3/(6*E*I) + C1*x**2/2 + C2*x + C3
w1 = sp.integrate(-E*I/(G*Ks*A) * phi1.diff(x, 2) + phi1, x) + C4
print('\nw1:')
print(w1)

# Region 2: L/2 <= x <= L with no load q = 0
phi2 = C5*x**2/2 + C6*x + C7
w2 = sp.integrate(-E*I/(G*Ks*A) * phi2.diff(x, 2) + phi2, x) + C8
print('\nw2:')
print(w2)

M1 = -E * I * phi1.diff(x)
M2 = -E * I * phi2.diff(x)

boundary_conditions = [
    w1.subs(x, 0),                          
    M1.subs(x, 0),                          
    w1.subs(x, L/2) - w2.subs(x, L/2),     
    phi1.subs(x, L/2) - phi2.subs(x, L/2), 
    w2.subs(x, L),                          
    M2.subs(x, L),                          
    w1.diff(x).subs(x, L/2) - w2.diff(x).subs(x, L/2),  
    phi1.diff(x).subs(x, L/2) - phi2.diff(x).subs(x, L/2)  
]

all_constants = [C1, C2, C3, C4, C5, C6, C7, C8]

integration_constants = sp.solve(boundary_conditions, all_constants)

w1_sol = w1.subs(integration_constants)
w2_sol = w2.subs(integration_constants)

num_vals = {q0: 1, L: 1, E: 210e9, I: 833e-12, G: 81e9, Ks: 5/6, A: 100e-6}
w1_num = w1_sol.subs(num_vals)
w2_num = w2_sol.subs(num_vals)

w1_f = sp.lambdify(x, w1_num, 'numpy')

x_vals1 = np.linspace(0, 0.5, 500)
x_vals2 = np.linspace(0.5, 1, 500)
w1_vals = w1_f(x_vals1)

plt.figure(figsize=(10, 5))
plt.plot(x_vals1, w1_vals * 1e3, label='Region 1 (loaded)', linewidth=2)
plt.xlabel('Position (m)')
plt.ylabel('Deflection (mm)')
plt.show()

print(f'\nDeflection at x=L/2 [mm]: {w1_f(0.5)*1e3:.4f}')
print(f'Max deflection [mm]: {np.max(np.abs(w1_vals))*1e3:.4f}')