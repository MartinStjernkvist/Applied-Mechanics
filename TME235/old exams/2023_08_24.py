#%%
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp


#%%
# 2

phi_vals = np.linspace(0, np.pi/2, 1000)

sigma = np.array([
    [100, 120, 0],
    [120, 50, 0],
    [0,0,0]
])

norm_shear = np.zeros(np.shape(phi_vals))
# print(norm_shear)

for i in range(len(phi_vals)):
    
    e1_prim = np.array([
        [np.cos(phi_vals[i])],
        [np.sin(phi_vals[i])],
        [0]
    ])
    
    traction = sigma.T @ e1_prim
    print('\ntraction:')
    print(traction)
    
    normal_magnitude = (traction.T @ e1_prim)[0,0]
    normal_comp = normal_magnitude * e1_prim
    print('\nnormal comp:')
    print(normal_comp)
    
    traction_shear = traction - normal_comp
    print('\ntraction shear:')
    print(traction_shear)
    
    norm_shear[i] = np.linalg.norm(traction_shear)
    
plt.figure()
plt.plot(phi_vals, norm_shear)
plt.show()


#%%
# 2 - einstein notation

phi_vals = np.linspace(0, np.pi/2, 10000)

sigma_ij = np.array([
    [100, 120, 0],
    [120, 50, 0],
    [0,0,0]
])

norm_shear = np.zeros(np.shape(phi_vals))
# print(norm_shear)

for i in range(len(phi_vals)):
    
    e1_prim_i = np.array([np.cos(phi_vals[i]), np.sin(phi_vals[i]), 0])
    
    print(np.shape(e1_prim_i))
    traction_i = np.einsum('ij,j->i', sigma_ij.T, e1_prim_i)
    print('\ntraction:')
    print(traction_i)
    
    normal_comp_i = np.einsum('i,i->', traction_i, e1_prim_i) * e1_prim_i
    print('\nnormal comp:')
    print(normal_comp_i)
    
    traction_shear_i = traction_i - normal_comp_i
    print('\ntraction shear:')
    print(traction_shear_i)
    
    norm_shear[i] = np.linalg.norm(traction_shear_i)
    
max_i = np.argmax(norm_shear)
print(max_i)
print('\nmax value:')
print(norm_shear[max_i], phi_vals[max_i] * 180/np.pi)

min_i = np.argmax(-norm_shear)
print(min_i)
print('\nmin value:')
print(norm_shear[min_i], phi_vals[min_i] * 180/np.pi)
    
plt.figure()
plt.plot(phi_vals, norm_shear, color='red')
plt.show()

E = 70e3
nu = 0.3
eps = (1 + nu) / E * sigma_ij - nu / E * np.einsum('ii->', sigma_ij) * np.eye(3)
print('strain:')
print(eps)

eigvals, eigvecs = np.linalg.eig(eps)
print('\neigenvalues:')
print(eigvals)

print('\neigenvectors:')
print(eigvecs)


#%%
# 3

x, t, rho, gamma, omega, A, E, I, L, C1, C2, C3, C4 = sp.symbols('x t rho gamma omega A E I L C1 C2 C3 C4', real=True)

W = C1 * sp.sin(gamma * x / L) + C2 * sp.cos(gamma * x / L) + C3 * sp.sinh(gamma * x / L) + C4 * sp.cosh(gamma * x / L)

W_4 = W.diff(x, 4)

w = W * sp.sin(omega * t)

M = - E * I * w.diff(x, 2)

w_sub_0 = w.subs(x, 0)
print('\n0 deflection at x=0:')
print(w_sub_0)

M_sub_0 = M.subs(x, 0)
print('\n0 moment at x=0:')
print(M_sub_0)

w_sub_L = w.subs([(x, L), (C2, 0), (C4, 0)])
print('\n0 deflection at x=L:')
print(w_sub_L)

M_sub_L = M.subs([(x, L), (C2, 0), (C4, 0)])
print('\n0 moment at x=L:')
print(M_sub_L)



#%%
# 3

g = 9.81
rho = 7800
r = 1 
h = 2e-3

def sigma_theta(theta):
    return - rho * g * np.cos(theta) / (np.sin(theta))**2

def sigma_phi(theta):
    return - rho * g * r * np.cos(theta) / (np.sin(theta))**2 * (1 + (np.sin(theta))**2)

theta_vals = np.linspace(np.pi/4, np.pi/2, 100)

sigma_theta_vals = sigma_theta(theta_vals)
sigma_phi_vals = sigma_phi(theta_vals)


plt.figure()
plt.plot(theta_vals, sigma_theta_vals, label='sigma_theta')
plt.plot(theta_vals, sigma_phi_vals,'--',label='sigma_phi')
plt.legend()
plt.grid()
plt.show()

def sigma_vm(theta):
    sigma_theta = - rho * g * r * np.cos(theta) / np.sin(theta)**2
    sigma_phi = - rho * g * r * np.cos(theta) / np.sin(theta)**2 * (1 + np.sin(theta)**2)
    return 1/np.sqrt(2) * np.sqrt((sigma_theta - sigma_phi)**2 + sigma_theta**2 + sigma_phi**2)

sigma_vm_vals = sigma_vm(theta_vals)

plt.figure()
plt.plot(theta_vals, sigma_vm_vals, label='sigma_vm')
plt.legend()
plt.grid()
plt.show()