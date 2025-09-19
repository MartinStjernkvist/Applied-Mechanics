#%%
# The following code has been written by Martin Stjernkvist
# Ideally, the code should be run in the application VSCODE,
# within the interactive python environment

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
    print_string = '\n----------------------\n' + 'Assignment ' + str(num) + '\n----------------------\n'
    return print(print_string)
    
##################################################
# A2
##################################################
new_prob(2)
g = 9.81

f = lambda z: 6/np.sqrt(11) * 1 / (9 + (12/13) * z) * (12 + 4 * np.sqrt(z**2 - 6*z +34)) - 48 * z

solution = fsolve(f, 0)
print('solution to f(z) = 0:')
print(solution)

F_AB = lambda z: m*g * 6/np.sqrt(11) * 1 / (9 + (12/13) * z) * np.sqrt(z**2 - 6*z +34)
F_AC = lambda z: F_AB(z) * 4 * np.sqrt(11) /13 * z / np.sqrt(z**2 - 6*z +34) - 2 *m * g

a = 10 #m
m = 4_000 #kg
print('resulting cable forces at z_solution:')
print('F_AB: ', F_AB(solution)/1000, 'kN')
print('F_AC: ', F_AC(solution)/1000, 'kN')

z_range = np.linspace(0, 100, 1000)

title = 'A2'

fig = plt.figure()
plt.plot(z_range, f(z_range), label = 'f(z)')
plt.axvline(x=solution, color='black', linestyle='--', label='f(z) = 0, with smallest F_AB(z) & F_AC(z)')
plt.xlim (0, 2)
plt.ylim(-1000,1000)
plt.xlabel('z')
plt.ylabel('f(z)')
plt.grid()
plt.legend()
plt.show()

fig = plt.figure()
plt.plot(z_range, f(z_range), label = 'f(z)')
plt.plot(z_range,F_AB(z_range), 'red', label = 'F_AB(z)')
plt.plot(z_range,F_AC(z_range), 'green', label = 'F_AC(z)')
plt.axvline(x=solution, color='black', linestyle='--', label='f(z) = 0, with smallest F_AB(z) & F_AC(z)')
plt.xlim (0, 2)
plt.ylim(-100_000, 100_000)
plt.xlabel('z')
plt.grid()
plt.legend()
plt.show()


##################################################
# A3
##################################################
new_prob(3)

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
plt.axvline(x=np.arctan(2), color='black', linestyle='--', label='maximum a1_prim')
plt.axvline(x=np.arctan(-1/2) + 2* np.pi, color='grey', linestyle='--', label='maximum a2_prim')
plt.xlabel(r'angle $\phi$')
plt.legend()
plt.show()

print('angles with maximum values (rad):')
print('a1_prim: ', np.arctan(2))
print('a2_prim: ', np.arctan(-1/2))

print('\nmaximum values of a1_prim and a2_prim:')
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

sigma_prim_ij = np.array([[200, 100, -50], 
                      [100, 300, 70], 
                      [-50, 70, 100]])

# sigma_prim_ij = einsum('ji, ij -> ij',l_ij.evalf(),sigma_ij.evalf())
sigma_ij = l_ij.T @ sigma_prim_ij @ l_ij

print('sigma_ij (MPa):')
print(sigma_ij)



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
print('eigenvectors (columns in the matrix):')
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

print('2D plot:')
fig, axes = plt.subplots(1,1, squeeze=False)
ax1 = axes[0,0]
cp = ax1.contourf(x1_range, x2_range, Phi_f(x1_range, x2_range), levels = 50, cmap = 'inferno')
ax1.set_xlabel('x1')
ax1.set_ylabel('x2')
fig.colorbar(cp, ax=ax1, label=r'$\Phi$(x1, x2)')
plt.tight_layout()
plt.show()

print('3D plot:')
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(x1_range, x2_range, Phi_f(x1_range, x2_range), cmap='inferno', label = 'scalar field')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
fig.colorbar(surf, ax=ax, label=r'$\Phi$(x1, x2)')
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
print('\nconfirm solution:')
print(grad_Phi_f_x1(10,100), grad_Phi_f_x2(10,100))

hess_Phi = sp.hessian(Phi, (x1, x2))
print('\nhessian:')
display(sp.simplify(hess_Phi))

hess_Phi_f = sp.lambdify([x1, x2], hess_Phi)
print('value of hessian at solution:')
print(hess_Phi_f(10,100))

eval, evect = np.linalg.eig(hess_Phi_f(10,100))
print('\neigenvalues of hessian:')
print(eval)

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
print('\nprincipal stresses (eigenvalues of deviatoric stress tensor), (MPa):')
print(eval)

print('\nprincipal directions (eigenvectors, columns to matrix):')
print(evect)

n = np.array([[-3 / np.sqrt(45)],
              [-6 / np.sqrt(45)],
              [0]])

t = deviatoric_sigma_ij @ n
print('\nstress vector on plane (MPa):')
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
print('x:')
display(x)

x_dt = sp.diff(x, t)
print('v, Lagrangian:')
display(x_dt)

system_of_eqn = [X1 + X2 * v0 * t / h0 - x1, X2 + X1 * v0 * t / h0 - x2]
variables = (X1, X2)
sol = sp.solve(system_of_eqn, variables, dict = True)
# display(sol)
# display(sol[0][X1])

X = sp.Matrix([[sol[0][X1]],
               [sol[0][X2]],
               [0]])

v = sp.diff(X, t)
print('v, Eulerian:')
display(v)


# %%