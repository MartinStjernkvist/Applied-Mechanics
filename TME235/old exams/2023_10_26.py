#%%
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

#%%
# 1
c = 0.3
sigma = 100e6
alpha = [0.5]
alpha = np.linspace(0, 1, 1000)

sigma_eff_vals = []

for i in range(len(alpha)):
    sigma_ij = np.array([
        [(1-alpha[i])*sigma, alpha[i]*sigma, 0],
        [alpha[i]*sigma, 0, 0],
        [0,0,0]
        ])

    tr = np.linalg.trace(sigma_ij)

    eigvals, _ = np.linalg.eig(sigma_ij)
    lam1, lam2, lam3 = eigvals[0], eigvals[1], eigvals[2]
    sigma_eT = np.max([np.abs(lam1-lam2), np.abs(lam2-lam3), np.abs(lam1-lam3)])

    sigma_eff = sigma_eT + c * tr
    # print(sigma_eff)

    sigma_eff_vals.append(sigma_eff)

plt.figure()
plt.plot(alpha, sigma_eff_vals)
plt.show()

index = np.argmin(sigma_eff_vals)
print(index)
print(sigma_eff_vals[index])
print(alpha[index])


#%%
#4

r, omega, rho, E, nu, a, b = sp.symbols(' ', real=True)


f_r = rho * omega**2 * r

boundary_conditions = [
    u.subs(r, a),
    u.diff(r).subs(r, a),
    M.subs(r, b),
    V.subs(r, b)
]

unknowns = ()
integration_constants = sp.solve(boundary_conditions, unknowns)

u_sol = u.subs(integration_constants)

a_num = 100e-3
b_num = 400e-3
h_num = 10e-3
rho_num = 7800
E_num = 210e9
vu_num = 0.3
omega_num = 700 * np.pi / 180


# not able to solve due to lack of equations in the formula sheet


#%%
# 5

A0 = 100
L = 100
G = 100
alpha = np.linspace(-np.pi/4 , np.pi/4 , 500)

P_list = []
H_list = []

for i in range(len(alpha)):
    # initial horizontal distance
    LH = L * np.cos(np.pi/4)

    # deformed horizontal length
    l = LH / np.cos(alpha[i])

    # stretch
    lam = l / L

    # given formula
    stress = G * (lam** 2 - 1)

    # incompressibility, volume is preserved
    A1 = 2 * A0 * L / l
    A2 =  A0 * L / l

    # internal normal forces
    N1 = stress * A1
    N2 = stress * A2

    # horizontal equilibrium, upper node
    H = - np.cos(alpha[i]) * (N1 - N2)
    H_list.append(H)

    # vertical equilibrium, upper node
    P = - np.sin(alpha[i]) * (N1 + N2)
    P_list.append(P)
    

plt.figure()
plt.plot(alpha, P_list, label='P')
plt.plot(alpha, H_list, label='H')
plt.legend()
plt.show()

i_max_H = np.argmax(H_list)
i_max_P = np.argmax(P_list)

print(H_list[i_max_H])
print(P_list[i_max_P])

