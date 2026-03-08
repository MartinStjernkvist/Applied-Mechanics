#%%
import numpy as np

def p4_p1(p2_p1):
    return p2_p1 * (1 - 1/7 * (p2_p1 - 1) / np.sqrt(1 + 6/7 * (p2_p1 - 1)))**(-7)


p2_p1_list = np.linspace(2.12, 2.14, 10)

# for i in range(len(p2_p1_list)):
#     print(f'\np2_p1 = {p2_p1_list[i]:.2f}')
#     print(f'p4_p1 = {p4_p1(p2_p1_list[i]):.2f}')
    
Ms = 1.403
gamma = 1.4
T1 = 300
R = 287

M2prim = 0.739
T2_T1 = 1.25
T2 = T2_T1 * T1


fact = Ms / (Ms**2 -1) * np.sqrt(1 + 2 * (gamma -1) /(gamma +1)**2 * (Ms**2 - 1) * (gamma + 1 / Ms**2))

MR = + (1/fact)/2 + np.sqrt(((1/fact)/2)**2 + 1)
print(MR)

W = Ms * np.sqrt(gamma * R * T1)


# T2 = 2.13 * ((2.8 / 0.4) + 2.13)/(1 + (2.8 / 0.4)*2.13)

a2 = np.sqrt(gamma * R * T2) 
print('a2', a2)
u2prim = a2 * M2prim
print(u2prim)

up = W - u2prim
print(up)
# print(np.sqrt(gamma * 287 * T2) * 0.739)

# W = Ms * np.sqrt(gamma * 287 * 300)

# up = W - np.sqrt(gamma * 287 * T2) * 0.739

# print(up)

WR = a2 * MR - up
print(WR)

print((0.615 - 0.74)/0.25)
print((0.145 - 0.225)/0.25)
print((0.4 - 0.43)/0.25)
print((0.675 - 0.63)/0.25)

print(-0.32 * 0.25)

#%%
