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
    print(sigma_eff)
    
    sigma_eff_vals.append(sigma_eff)

plt.figure()
plt.plot(alpha, sigma_eff_vals)
plt.show()

index = np.argmin(sigma_eff)
print(sigma_eff_vals[index])

