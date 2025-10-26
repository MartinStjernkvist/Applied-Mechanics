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