#%%
# !pip install ipympl
%matplotlib widget
# %matplotlib inline
from scipy.optimize import fsolve
# from scipy.differentiate import hessian
import numpy as np
from numpy import einsum
import matplotlib.pyplot as plt
import sympy as sp
from IPython.display import display, Math
from mpl_toolkits.mplot3d import axes3d
from numpy.random import rand
from IPython.display import HTML
from matplotlib import animation

sp.init_printing(use_latex='mathjax')

def disp(expr):
    expr = expr.simplify()
    latex_expr = sp.latex(expr)
    display(Math(latex_expr))

def doit(expr):
    return disp(expr.doit)

def new_prob(num):
    print_string = '\n-----------\n' + 'A' + str(num) + '\n-----------\n'
    return print(print_string)
    
##################################################
# A2
##################################################
new_prob(2)
g = 9.81

f = lambda z: 6/np.sqrt(11) * 1 / (9 + 12/13 * z) * (12 + 4 * np.sqrt(z**2 - 6*z +34)) + 48 * z

solution = fsolve(f, 0)
# print(solution)

F_AB = lambda z: g * 6/np.sqrt(11) * 1 / (9 + 12/13 * z) * np.sqrt(z**2 - 6*z +34)
F_AC = lambda z: F_AB(z) * 4 * np.sqrt(11) /13 * z / np.sqrt(z**2 - 6*z +34) - 2 * g

z_range = np.linspace(-20, 10, 100)

title = 'A2'

fig = plt.figure()
plt.plot(z_range, f(z_range), label = 'f(z)')
plt.plot(z_range,F_AB(z_range), 'red', label = 'F_AB(z)')
plt.plot(z_range,F_AC(z_range), 'green', label = 'F_AC(z)')
plt.axvline(x=solution, color='black', linestyle='--')
plt.xlim (-12, 2)
plt.ylim(-400, 400)
plt.grid()
plt.legend()
plt.show()
fig.savefig(title, bbox_inches = 'tight')





##################################################
# A3
##################################################
new_prob(3)

# a = np.array([1,2,3])
# rot_matrix = lambda phi: np.array([[np.cos(phi), np.sin(phi),0], 
#                                    [-np.sin(phi), np.cos(phi),0], 
#                                    [0, 0, 1]])
# e = np.array([[1,0,0], 
#               [0,1,0], 
#               [0,0,1]])

# e_prim = lambda phi: rot_matrix(phi) @ e
# a_prim = lambda phi: np.dot(e_prim(phi), e) * a

# phi = np.linspace(0,2 * np.pi)
# a_prim_values = np.array([a_prim(p) for p in phi])

# title = 'A3_1'

# fig2 = plt.figure()

# plt.plot(phi, a_prim_values[:,0], 'blue', label = 'a1_prim')
# plt.plot(phi, a_prim_values[:,1], 'red', label = 'a2_prim')
# plt.plot(phi, a_prim_values[:,2], 'green', label = 'a3_prim')
# plt.legend()
# plt.show()
# fig2.savefig(title, bbox_inches = 'tight')

# a_1 = 1
# a_2 = 2

a = np.array([1, 2, 3])
e = np.array([[1, 0, 0], 
              [0, 1, 0], 
              [0, 0, 1]])
rot_matrix = lambda phi: np.array([[np.cos(phi), np.sin(phi), 0],
                                   [-np.sin(phi), np.cos(phi), 0], 
                                   [0, 0, 1]])

e_prim = lambda phi: rot_matrix(phi) @ e
a_prim = lambda phi: np.dot(e_prim(phi), e) @ a

a_1_prim = lambda phi: a_prim(phi)[0]
a_2_prim = lambda phi: a_prim(phi)[1]

phi = np.linspace(0, 2 * np.pi, 1000)

a_1_prim_values = [a_1_prim(p) for p in phi]
a_2_prim_values = [a_2_prim(p) for p in phi]
a_prim_sum_values = [a_1_prim(p) + a_2_prim(p) for p in phi]

title = 'A3_2'

fig = plt.figure()
plt.plot(phi, a_1_prim_values, label = 'a1_prim')
plt.plot(phi, a_2_prim_values, label = 'a2_prim')
plt.plot(phi, a_prim_sum_values, label = 'a1_prim + a2_prim')
plt.legend()
plt.axvline(x=np.arctan(2), color='black', linestyle='--')
plt.axvline(x=np.arctan(-1/2) + 2* np.pi, color='black', linestyle='--')
plt.show()
fig.savefig(title, bbox_inches='tight')

print('maximum values of a1_prim and a2_prim:')
print('a1_prim:', 1*np.cos(np.arctan(2)) + 2* np.sin(np.arctan(2)))
print('a2_prim:', -1*np.sin(np.arctan(-1/2)) + 2* np.cos(np.arctan(-1/2)))


max_y = max(a_prim_sum_values)
max_x = phi[a_prim_sum_values.index(max_y)]


##################################################
# A4
##################################################
new_prob(4)

l_ij = np.array([[1 / 3, 2/3, 2/3], 
                 [0, 1 /np.sqrt(2), -1 /np.sqrt(2)],
                 [-4/(3 * np.sqrt(2)), 1/(3 * np.sqrt(2)), 1/(3 * np.sqrt(2))]])

sigma_ij = np.array([[200, 100, -50], 
                      [100, 300, 70], 
                      [-50, 70, 100]])

# sigma_prim_ij = einsum('ji, ij -> ij',l_ij.evalf(),sigma_ij.evalf())
sigma_prim_ij = l_ij.T @ sigma_ij @ l_ij

print('sigma_prim_ij (MPa):')
print(sigma_prim_ij)



##################################################
# A5
##################################################
new_prob(5)

T = np.array([[6, 4, 0], 
              [4, 3, 0], 
              [0, 0, 2]])


eval, evect = np.linalg.eig(T)
print('eigenvalues:')
print(eval)
print('eigenvectors:')
print(evect)


##################################################
# A6
##################################################
new_prob(6)

# Define function
x1, x2 = sp.symbols('x1 x2')
Phi = (10 - x1)**2 + 3*(x2 - x1**2)**2 - 1
Phi_f = sp.lambdify([x1, x2], Phi)

N = 100
lim_max = 1000
lim_min = -1000
linspace = np.linspace(lim_min, lim_max, N)
x1_range, x2_range = np.meshgrid(linspace, linspace)

figsize = (5,5)
fig = plt.figure(figsize=figsize)
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(x1_range, x2_range, Phi_f(x1_range, x2_range), cmap='inferno', label = 'scalar field')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
fig.colorbar(surf, ax=ax)
plt.tight_layout()
plt.show()

grad_Phi = [sp.diff(Phi, var) for var in (x1, x2)]
print('gradient:')
display(grad_Phi)

solution = sp.solve(grad_Phi, [x1,x2])
print('solution to grad(Phi) = [0,0]')
print(solution)

grad_Phi_f_x1 = sp.lambdify([x1, x2], grad_Phi[0])
grad_Phi_f_x2 = sp.lambdify([x1, x2], grad_Phi[1])
print('confirm solution:')
print(grad_Phi_f_x1(10,100), grad_Phi_f_x2(10,100))

# fig, axes = plt.subplots(1,2, figsize = figsize, squeeze=False)
# ax1 = axes[0,0]
# ax2 = axes[0,1]
# ax1.contourf(x1_range, x2_range, grad_Phi_f_x1(x1_range, x2_range), levels = 50, cmap = 'inferno')
# ax2.contourf(x1_range, x2_range, grad_Phi_f_x2(x1_range, x2_range), levels = 50, cmap = 'inferno')
# plt.tight_layout()
# plt.show()

hess_Phi = sp.hessian(Phi, (x1, x2))
print('hessian:')
display(sp.simplify(hess_Phi))

hess_Phi_f = sp.lambdify([x1, x2], hess_Phi)
print('value of hessian at solution:')
print(hess_Phi_f(10,100))

eval, evect = np.linalg.eig(hess_Phi_f(10,100))
print('eigenvalues of hessian:')
print(eval)

fig, axes = plt.subplots(1,1, figsize = figsize, squeeze=False)
ax1 = axes[0,0]
cp = ax1.contourf(x1_range, x2_range, Phi_f(x1_range, x2_range), levels = 50, cmap = 'inferno')
ax1.set_xlabel('x1')
ax1.set_ylabel('x2')
fig.colorbar(cp, ax=ax1)
plt.tight_layout()
plt.show()

##################################################
# A8
##################################################
new_prob(8)

sigma_ij = np.array([[30, 0, 10],
                     [0, 30, 10],
                     [10, 10, 30]])

sigma_m = np.trace(sigma_ij) / 3
deviatoric_sigma_ij = sigma_ij - sigma_m * np.identity(3)
print('sigma_ij (dev):')
print(deviatoric_sigma_ij)

eval, evect = np.linalg.eig(deviatoric_sigma_ij)
print('principal stresses (eigenvalues):')
print(eval)
print('principal directions (eigenvectors):')
print(evect)

n = np.array([[-3 / np.sqrt(45)],
              [-6 / np.sqrt(45)],
              [0]])

t = deviatoric_sigma_ij @ n
print('stress vector on plane:')
print(t)

##################################################
# A9
##################################################
new_prob(9)

h0, v0, t = sp.symbols('h0 v0 t')
x1, x2, x3 = sp.symbols('x1 x2 x3')
X1, X2, X3 = sp.symbols('X1 X2 X3')


x = sp.Matrix([[X1 + X2 * v0 * t / h0],
               [X2 + X1 * v0 * t / h0],
               [X3]])
display(x)

x_dt = sp.diff(x, t)
display(x_dt)

system_of_eqn = [X1 + X2 * v0 * t / h0 - x1, X2 + X1 * v0 * t / h0 - x2]
variables = (X1, X2)
sol = sp.solve(system_of_eqn, variables, dict = True)
display(sol)
display(sol[0][X1])

X = sp.Matrix([[sol[0][X1]],
               [sol[0][X2]],
               [0]])

v = sp.diff(X, t)
display(v)

# Define symbols
x1_sym, x2_sym, x3_sym = sp.symbols('x1 x2 x3')

# Define the expressions for x1, x2, x3 in terms of X1, X2, X3, t, v0, h0
x1_expr = X1 + X2 * v0 * t / h0
x2_expr = X2 + X1 * v0 * t / h0
x3_expr = X3

# Substitute x1_sym -> x1_expr, x2_sym -> x2_expr in v
v_x = v.subs({x1_sym: x1_expr, x2_sym: x2_expr})
display(v_x)

# F = sp.diff(x, [X1, X2, X3])
# display(F)

print(np.sqrt(9+36))



# %%