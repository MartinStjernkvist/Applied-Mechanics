import scipy.optimize
import numpy as np
import matplotlib.pyplot as plt
import sympy as smp
"""
A2
"""
# g = 9.81

# f = lambda z: 6/np.sqrt(11) * 1 / (9 + 12/13 * z) * (12 + 4 * np.sqrt(z**2 - 6*z +34)) + 48 * z

# solution = fsolve(f, 0)
# print(solution)

# F_AB = lambda z: g * 6/np.sqrt(11) * 1 / (9 + 12/13 * z) * np.sqrt(z**2 - 6*z +34)
# F_AC = lambda z: F_AB(z) * 4 * np.sqrt(11) /13 * z / np.sqrt(z**2 - 6*z +34) - 2 * g

# z_range = np.linspace(-20, 10, 100)

# title = 'upg1.png'
# fig = plt.figure(figsize=(10,10))
# plt.plot(z_range, f(z_range), label = 'f(z)')
# plt.plot(z_range,F_AB(z_range), 'red', label = 'F_AB(z)')
# plt.plot(z_range,F_AC(z_range), 'green', label = 'F_AC(z)')
# plt.axvline(x=solution, color='black', linestyle='--')
# plt.xlim (-12, 2)
# plt.ylim(-400, 400)
# plt.grid()
# plt.legend()
# fig.savefig(title, bbox_inches = 'tight')
"""
A3
"""
'''
a = np.array([1,2,3])
rot_matrix = lambda phi: np.array([[np.cos(phi), np.sin(phi),0], [-np.sin(phi), np.cos(phi),0], [0, 0, 1]])
e = np.array([[1,0,0], [0,1,0], [0,0,1]])
e_prim = lambda phi: rot_matrix(phi) @ e

a_prim = lambda phi: np.dot(e_prim(phi), e) * a

phi = np.linspace(0,2 * np.pi)
print(a_prim)

title = 'upg2.png'
fig2 = plt.figure(figsize=(10,10))
plt.plot(phi, a_prim(phi)[0], 'blue')
plt.plot(phi, a_prim(phi)[1], 'red')
plt.plot(phi, a_prim(phi)[2], 'green')
plt.show()
fig2.savefig(title, bbox_inches = 'tight')
'''
a_1 = 1
a_2 = 2

a = np.array([1, 2, 3])
e = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
rot_matrix = lambda phi: np.array([[np.cos(phi), np.sin(phi), 0],
                                   [-np.sin(phi), np.cos(phi), 0], [0, 0, 1]])
e_prim = lambda phi: rot_matrix(phi) @ e

a_prim = lambda phi: np.dot(e_prim(phi), e) @ a

a_1_prim = lambda phi: a_prim(phi)[0]
a_2_prim = lambda phi: a_prim(phi)[1]

phi = np.linspace(0, 2 * np.pi, 1000)
a_1_prim_values = [a_1_prim(p) for p in phi]
a_2_prim_values = [a_2_prim(p) for p in phi]
a_prim_sum_values = [a_1_prim(p) + a_2_prim(p) for p in phi]

title = 'upg2.png'
fig = plt.figure()
plt.plot(phi, a_1_prim_values)
plt.plot(phi, a_2_prim_values)
plt.plot(phi, a_prim_sum_values)
fig.savefig(title, bbox_inches='tight')

max_y = max(a_prim_sum_values)
max_x = phi[a_prim_sum_values.index(max_y)]
print(max_x / np.pi * 180)
'''
# a_1_prim = lambda phi: np.cos(phi) * a_1 + np.sin(phi) * a_2
# a_2_prim = lambda phi: -np.sin(phi) * a_1 + np.cos(phi) * a_2

# f = np.polyfit(phi, a_prim_sum_values, 100)
# func = np.poly1d(f)
# max = scipy.optimize.fmin(func, 0)

# max = scipy.optimize.fmin(lambda p: -np.sum(a_prim(p)), 0)

# print(a_1_prim(max), a_2_prim(max))
'''
"""
A4
"""

l_ij = np.array([[1 / 2, 0, np.sqrt(2) / 3 * 2], 
                 [1, 1, np.sqrt(2) / 3 * 1 / 2],
                 [1, -1, np.sqrt(2) / 3 * 1 / 2]])

sigma_ij = np.array([[200, 100, -50], [100, 300, 70], [-50, 70, 100]])

sigma_prim_ij = l_ij.T @ sigma_ij
print(sigma_prim_ij)

print(np.sqrt(4.5))

