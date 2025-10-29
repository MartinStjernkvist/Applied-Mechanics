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





###############################

#%%
# 1

alpha, beta = sp.symbols('alpha beta', real=True)

l1 = sp.Matrix([
    [sp.cos(beta), 0, -sp.sin(beta)],
    [0, 1, 0],
    [sp.sin(beta), 0, sp.cos(beta)]
])

l2 = sp.Matrix([
    [sp.cos(alpha), sp.sin(alpha), 0],
    [-sp.sin(alpha), sp.cos(alpha), 0],
    [0,0,1]
])

l_tot = l2 * l1

l_f = sp.lambdify((alpha, beta), l_tot, 'numpy')

acc = sp.Matrix([1, 2, -2])

l_num =  l_f(alpha= 60*np.pi/180, beta= 20*np.pi/180)
acc_new = l_num * acc

print('acceleration:')
print(acc_new)

sigma_new = np.array([
    [150, 70, 0],
    [70, 50, 15],
    [0, 15, 100]
])

sigma = np.linalg.inv(l_num) * sigma_new * np.linalg.inv(l_num.T)

print('\nsigma:')
print(sigma)

eigvals, _ = np.linalg.eig(sigma)
print('\neigenvalues:')
print(eigvals)


#%%
# 4

x, L, Ks, b, h, P_hat, M_hat, nu, E, q = sp.symbols('x L Ks b h P_hat M_hat nu E q', real=True)
# C1, C2, C3, C4 = sp.symbols('C1 C2 C3 C4', real=True)

I = b * h**3 / 12
A = b * h
G = E / (2 *(1 + nu))

phi = sp.Function('phi')(x)
w = sp.Function('w')(x)

diffeq_phi = sp.Eq(E * I * phi.diff(x, 3), q)
phi = sp.dsolve(diffeq_phi, phi).rhs
print('\nphi:')
print(phi)

diffeq_w = sp.Eq(w.diff(x), - E * I / ( G * Ks * A) * phi.diff(x,2) + phi)
w = sp.dsolve(diffeq_w, w).rhs
print('\nw:')
print(w)

M = -E * I * phi.diff(x)
V = G * Ks * A * (-phi + w.diff(x))

bc = [
    w.subs(x, 0),
    w.diff(x).subs(x, 0),
    M.subs(x, L) - M_hat,
    V.subs(x, L) - P_hat
]

unknowns = ('C1, C2, C3, C4')
ic = sp.solve(bc, unknowns)
print('\nintegration constants:')
print(ic)

w_sol = w.subs(ic)
print('\nw solution:')
print(w_sol)

w_f = sp.lambdify((x, L, Ks, b, h, P_hat, M_hat, nu, E, q), w_sol, 'numpy')

E_num = 80e9
nu_num = 0.2
P_num = 2e3
M_num = 500
L_num = 1
h_num = 30e-3
b_num = 50
Ks_num = 5/6
q_num = 0

x_vals = np.linspace(0, L_num, 500)
w_vals = w_f(x_vals, L_num, Ks_num, b_num, h_num, P_num, M_num, nu_num, E_num, q_num)

plt.figure()
plt.plot(x_vals, w_vals)
plt.show()

print('\ndeflection at the end:')
print(w_vals[-1])


#%%
# 5 

s, theta, alpha, phi, rho, g, h, H, r, l, C1 = sp.symbols('s theta alpha phi rho g h H r l C1', real=True)

f = rho * g * h
f_theta = f * sp.cos(alpha)
f_r = - f * sp.sin(alpha)

N_phi = f_r * s * sp.tan(alpha)

N_theta = 1/s * sp.integrate(N_phi - f_theta * s, s) + C1

eq = sp.Eq(N_theta.subs(s, H/sp.cos(alpha)), - rho * l * g * h * sp.cos(alpha) / 2)
C1 = sp.solve(eq, C1)
print(C1)


N_theta.simplify()