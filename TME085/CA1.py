#%%
import numpy as np

def interpolation(x1, y1, x2, y2, x):
    return y1 + (x - x1) * (y2 - y1) / (x2 - x1)

def Cp_func(gamma, R):
    return gamma * R / (gamma - 1)

def Cv_func(gamma, R):
    return R / (gamma - 1)

def T0_func(T, M, gamma=1.4):
    return T * (1 + (gamma - 1)/2 * M**2)

def p2_func(p1, M1, M2, gamma=1.4):
    return p1 * (1 + gamma * M1**2) / (1 + gamma * M2**2)

def q_func(T01, T02, Cp):
    return Cp * (T02 - T01)

def rho2_func(rho1, M1, M2, gamma=1.4):
    return rho1 * ((1 + gamma * M2**2) / (1 + gamma * M1 **2))**2 * (M1 / M2)**2

def T2_func(T1, M1, M2, gamma=1.4):
    return T1 * ((1 + gamma * M1**2) / (1 + gamma * M2 **2))**2 * (M2 / M1)**2

def T02_func(T01, M1, M2, gamma=1.4):
    return T01 * ((1 + gamma * M1 **2)/(1 + gamma * M2 **2))**2 * (M2 / M1)**2 * (1 + (gamma - 1)/2 * M2**2) / (1 + (gamma - 1)/2 * M1**2)

def T02_from_q_func(q, T01, Cp):
    return q / Cp + T01

bar_to_Pa = 101_325

# Task1 1.a
M1_1a = 0.2
T1_1a = 293
P1_1a = 1 * bar_to_Pa
R_1a = 287.0
gamma_1a = 1.4
Cp_1a = Cp_func(gamma_1a, R_1a)

D = 0.3

# q_star_2_1a = 2.55954 * 0.9
# print(q_star_2_1a, 'W/m^2')

def task_1a(T1, M1, Cp):
    T01 = T0_func(T1, M1)
    T02_star = T02_func(T01, M1, M2=1)
    q_star = q_func(T01, T02_star, Cp)
    q = 0.9 * q_star
    print('q: ', q)
    T02 = T02_from_q_func(q, T01, Cp)
    T01_T0_star = 0.1736
    T02_T0_star = (T02 / T01) * T01_T0_star
    print('T02_T0_star: ', T02_T0_star)
    # 7.00000e-01 1.42349e+00 9.92895e-01 1.43367e+00 1.04310e+00 9.08499e-01
    # 7.20000e-01 1.39069e+00 1.00260e+00 1.38709e+00 1.03764e+00 9.22122e-01
    M2 = interpolation(9.08499e-01, 7.00000e-01, 9.22122e-01, 7.20000e-01, T02_T0_star)
    print('M2: ', M2)
    
task_1a(T1_1a, M1_1a, Cp_1a)

#%%