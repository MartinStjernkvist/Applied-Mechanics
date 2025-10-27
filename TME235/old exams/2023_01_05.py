#%%
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

#%%
# 1

alpha, beta = sp. symbols('alpha beta', real=True)

l1 = sp.Matrix([
    [sp.cos(alpha), -sp.sin(alpha), 0],
    [sp.sin(alpha), sp.cos(alpha), 0],
    [0,0,1]
])

l2 = sp.Matrix([
    [sp.cos(beta), 0, -sp.sin(beta)],
    [0, 1, 0],
    [sp.sin(beta), 0, sp.cos(beta)]
])

l_tot = l2 * l1
print('l_tot:')
print(l_tot)
# l_tot

l_tot_f = sp.lambdify((alpha, beta), l_tot, 'numpy')

alpha_num = 30 * np.pi / 180
beta_num = -60 * np.pi / 180
l_tot_num = l_tot_f(alpha_num, beta_num)
print('l_tot_num:')
print(l_tot_num)


v = np.array([2, -3, 1])

v_bis = np.einsum('ij,i->i', l_tot_num, v)
print('\nv_bis:')
print(v_bis)

sigma_bis = np.array([
    [100, 50, 0],
    [50, -20, 75],
    [0, 75, 80]
])

sigma = np.linalg.inv(l_tot_num) @ sigma_bis @ np.linalg.inv(l_tot_num.T)
print('\nsigma: ')
print(sigma)

eigvals, eigvecs = np.linalg.eig(sigma)
print('\neigenvalues: ', eigvals)

#%%
#4

r, nu, E, h, q0, a, b, C1, C2, C3, C4 = sp.symbols('r nu E h q0 a b C1 C2 C3 C4', real=True)

q = - q0 * (r - a) / (b - a)

D = E * h **3 / (12 * (1 - nu**2)) 

# w = sp.Function('w')(r)

# diffeq = sp.Eq(1 / r * sp.diff(r * sp.diff(1 / r * sp.diff( r * w.diff(r), r), r), r) , q / D)

# w = sp.dsolve(diffeq, w)
# w = w.rhs

# First integration
int1 = sp.integrate(q * r / D, r)
expr1 = int1 + C1

# Second integration
int2 = sp.integrate(expr1 / r, r)
expr2 = int2 + C2

# Third integration  
int3 = sp.integrate(expr2 * r, r)
expr3 = int3 + C3

# Fourth integration
w = sp.integrate(expr3 / r, r) + C4

M_r = D * (- w.diff(r, 2) - nu * w.diff(r) / r)
M_phi = D * (- nu * w.diff(r, 2) - w.diff(r) / r)

V = 1/r * (sp.diff(r * M_r, r) - M_phi)

boundary_conditions = [
    w.subs(r, b),
    w.diff(r).subs(r, b),
    M_r.subs(r, a),
    V.subs(r, a)
]

unknowns = (C1, C2, C3, C4)

integration_constants = sp.solve(boundary_conditions, unknowns)
w_sol = w.subs(integration_constants)
w_f = sp.lambdify((r, nu, E, h, q0, a, b), w_sol, 'numpy')

a_num = 20e-3
b_num = 100e-3
h_num = 5e-3
nu_num = 0.3
E_num = 210e9

q0_vals = np.linspace(5e6, 1e7, 1000)
# q0_num = 100e6

r_vals = np.linspace(a_num, b_num,500)
w_vals = w_f(a_num, nu_num, E_num, h_num, q0_vals, a_num, b_num)

plt.figure()
plt.plot(q0_vals, w_vals)
plt.axhline(-2e-3)
plt.show()

difference = [-np.abs(w_vals[i] + 2e-3) for i in range(len(w_vals))]
index = np.argmax(difference)
print(index)

print(q0_vals[index])


#%%
#5

v0 = 100e-3
h0 = 100e-3
t = 0.1
Kb = 100e6
G = 10e6


F = np.array([
    [1, v0 * t / h0],
    [0, 1]
])

J = np.linalg.det(F)
print(J)

C = np.einsum('ik,kj->ij', F.T, F)
print(C)

S = G * (np.eye(2) - np.linalg.inv(C)) + (Kb - 2 * G / 3) * np.log(J) * np.linalg.inv(C)

sigma = J * np.einsum('ik,kj->ij', np.einsum('ik,kj->ij',F, S), F.T)

print('sigma:')
print(sigma)