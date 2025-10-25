#%%
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

#%%

###########################################################################
# Problem 2
###########################################################################

q0, r, a, b_minus_a, M, E, nu, h = sp.symbols('q0 r a b_minus_a M E nu h', real='True')

D = E * h**3 / (12 * (1 - nu**2))

w = sp.Function('w')(r)

diffeq = sp.Eq(1 / r * sp.diff(r * sp.diff(1 / r * sp.diff(r * sp.diff(w, r), r), r), r), q0 * (r - a)/ (b_minus_a) / D)

w_ = sp.dsolve(diffeq, w).rhs

Mr = D * (-sp.diff(w_, r, 2) - nu * sp.diff(w_, r) / r)
Mphi = D * (-sp.diff(w_, r) / r - nu * sp.diff(w_, r, 2))

V = 1 / r * (sp.diff(r * Mr, r) - Mphi)

bc = [
    w_.subs(r, b_minus_a + a),
    sp.diff(w_, r).subs(r, b_minus_a + a),
    V.subs(r, a),
    Mr.subs(r, a) + M
]

ic = sp.solve(bc, 'C1, C2, C3, C4' )
print('ic: ', ic)
w_sol = w_.subs(ic).simplify()

print(w_sol)

w_func = sp.lambdify((q0, r, a, b_minus_a, M, E, nu, h), w_sol, 'numpy')

r_vals = np.linspace(0.09, 0.3, 300)
vals = w_func(q0=1e6, r=r_vals, a=0.1, b_minus_a=0.2, M=100, E=80e9, nu=0.3, h=0.01)
# value = w_func(1,1,1,1,1,1,1,1)
print(np.argmax(vals))
print(np.argmin(vals))

plt.figure()
plt.plot(r_vals, vals)
plt.show()

#%%

###########################################################################
# Problem 3
###########################################################################



#%%

###########################################################################
# Problem 5
###########################################################################


# CAUSES PROBLEMS: 
# t = sp.symbols('t', real=True)

# C = sp.Matrix([
#     [1, 0, sp.sin(sp.pi * t)],
#     [0, (1 + t / 4)**2, (t/4) * (1 - sp.sin(2 * sp.pi * t))/6],
#     [sp.sin(sp.pi * t), (t/4) * (1 - sp.sin(2 * sp.pi * t))/6, ((1 - sp.sin(2 * sp.pi * t))/6)**2]
# ])

# print(C.eigenvals())

# # Extract eigenvalues from the dictionary
# eigenvals_dict = C.eigenvals()
# eigenvals_list = list(eigenvals_dict.keys())

# eigval1 = eigenvals_list[0]
# eigval2 = eigenvals_list[1]
# eigval3 = eigenvals_list[2]

# print(eigval1)

# eigval1_f = sp.lambdify(t, eigval1, 'numpy')
# eigval2_f = sp.lambdify(t, eigval2, 'numpy')
# eigval3_f = sp.lambdify(t, eigval3, 'numpy')

# t_vals = np.linspace(0, 100, 1000)

# eigval1_vals = eigval1_f(t_vals)
# eigval2_vals = eigval2_f(t_vals)
# eigval3_vals = eigval3_f(t_vals)

# plt.figure()
# plt.plot(t_vals, np.real(eigval1_vals), label='λ1')  # Take real part in case of numerical noise
# plt.plot(t_vals, np.real(eigval2_vals), label='λ2')
# plt.plot(t_vals, np.real(eigval3_vals), label='λ3')
# plt.legend()
# plt.xlabel('t')
# plt.ylabel('Eigenvalue')
# plt.show()


t = np.linspace(0, 1, 100)

stretch1_vec = 0 * t
stretch2_vec = 0 * t
stretch3_vec = 0 * t

for i in range(len(t)):
    
    F = np.array([
        [1,     0,                 np.sin(np.pi * t[i])            ],
        [0,     1 + t[i]/4,        0                               ],
        [0,      t[i]/4,             1 - np.sin(2 * np.pi * t[i])/6 ]
    ])
    
    C_v2 = F.T @ F
    
    # C = np.array([
    #     [1, 0, np.sin(np.pi * t[i])],
    #     [0, (1 + t[i] / 4)**2 + (t[i] / 4)**2, (t[i] / 4) * (1 - np.sin(2 * np.pi * t[i])/6)],
    #     [np.sin(np.pi * t[i]), (t[i] / 4) * (1 - np.sin(2 * np.pi * t[i])/6), (1 - np.sin(2 * np.pi * t[i])/6)**2]
    # ])

    eigenvals_list, _ = np.linalg.eig(C_v2)

    eigval1 = eigenvals_list[0]
    eigval2 = eigenvals_list[1]
    eigval3 = eigenvals_list[2]
    
    stretch1_vec[i] = eigval1 
    stretch2_vec[i] = eigval2
    stretch3_vec[i] = eigval3
    
plt.figure()
plt.plot(t, stretch1_vec, label='1') 
plt.plot(t, stretch2_vec, label='2')
plt.plot(t, stretch3_vec, label='3')
plt.legend()
plt.xlabel('t')
plt.ylabel('Eigenvalue')
plt.show()



G = 7e6
lam = 60e6
N = np.array([
    [1 + np.sqrt(2)],
    [0],
    [1 + np.sqrt(2)]
])
t0_list = 0 * t

for i in range(len(t)):
    
    F = np.array([
        [1,     0,                 np.sin(np.pi * t[i])            ],
        [0,     1 + t[i]/4,        0                               ],
        [0,      t[i]/4,             1 - np.sin(2 * np.pi * t[i])/6 ]
    ])
    
    C = F.T @ F
    
    J = np.linalg.det(F)
    S = G * (np.eye(3) - np.linalg.inv(C)) + lam * np.log(J) * np.linalg.inv(C)
    
    P = F @ S
    # print(P)
    
    t0 = P @ N
    
    t0_list[i] = np.linalg.norm(t0)
    
plt.figure()
plt.plot(t, t0_list)
plt.show()
  
print(P.shape)    
print(t0.shape)

#%%