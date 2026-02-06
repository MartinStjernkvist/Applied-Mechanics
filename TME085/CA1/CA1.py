#%%
import numpy as np

def printt(string):
    print()
    print('=' * 40)
    print(string)
    print('=' * 40)
    print()

def interpolation(x1, y1, x2, y2, x):
    return y1 + (x - x1) * (y2 - y1) / (x2 - x1)

def rho_func(p, R, T):
    return p / (R * T)

def Cp_func(gamma, R):
    return gamma * R / (gamma - 1)

def Cv_func(gamma, R):
    return R / (gamma - 1)

# Added heat:
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

# Wall friction:
def T2_fric_func(T1, M1, M2, gamma=1.4):
    return T1 * (2 + (gamma - 1) * M1**2) / (2 + (gamma - 1) * M2**2)
    
def p2_fric_func(p1, M1, M2, gamma=1.4):
    return p1 * (M1 / M2) * ((2 + (gamma - 1) * M1**2) / (2 + (gamma - 1) * M2**2))**(1/2)

def rho2_fric_func(rho1, M1, M2, gamma=1.4):
        return rho1 * (M1 / M2) * ((2 + (gamma - 1) * M1**2) / (2 + (gamma - 1) * M2**2))**(-1/2)
    
def Lstar_fric_func(fbar, D, M, gamma=1.4):
    return D / (4 * fbar) * ((1 - M**2) / (gamma * M**2) + (gamma + 1) / (2 * gamma) * np.log(((gamma + 1) * M**2)/ (2 + (gamma - 1) * M**2)))

bar_to_Pa = 101_325
R_univ = 287.0
gamma_air = 1.4
#%%
# Task1 1.a
printt('Task 1.a:')
M1_1a = 0.2
T1_1a = 293
p1_1a = 1 * bar_to_Pa
rho1_1a = 1.2
R_1a = R_univ
gamma_1a = gamma_air
Cp_1a = Cp_func(gamma_1a, R_1a)

D = 0.3

# q_star_2_1a = 2.55954 * 0.9
# print(q_star_2_1a, 'W/m^2')

def task_1a(T1, M1, p1, rho1, Cp):
    T01 = T0_func(T1, M1)
    T02_star = T02_func(T01, M1, M2=1)
    print(f'T02_star: {T02_star:.2f}')
    q_star = q_func(T01, T02_star, Cp)
    print(f'q_star: {q_star:.2f}')
    q = 0.9 * q_star
    print(f'q: {q:.2f}')
    T02 = T02_from_q_func(q, T01, Cp)
    
    T01_T0_star = 0.1736
    T02_T0_star = (T02 / T01) * T01_T0_star
    print(f'T02_T0_star: {T02_T0_star:.4f}')
    # From Table A.3:
    # 7.00000e-01 1.42349e+00 9.92895e-01 1.43367e+00 1.04310e+00 9.08499e-01
    # 7.20000e-01 1.39069e+00 1.00260e+00 1.38709e+00 1.03764e+00 9.22122e-01
    M2 = interpolation(9.08499e-01, 7.00000e-01, 
                       9.22122e-01, 7.20000e-01, 
                       T02_T0_star)
    print(f'M2: {M2:.2f}')
    
    T2 = T2_func(T1, M1, M2)
    p2 = p2_func(p1, M1, M2)
    rho2 = rho2_func(rho1, M1, M2)
    print(f'T2: {T2:.2f}')
    print(f'p2: {p2:.2f}')
    print(f'rho2: {rho2:.2f}')

task_1a(T1_1a, M1_1a, p1_1a, rho1_1a, Cp_1a)
#%%
# Task 1.b
printt('Task 1.b:')
M1_1b = 2
T1_1b = 293
p1_1b = 1 * bar_to_Pa
R_1b = R_univ
gamma_1b = gamma_air
# Cp_1b = Cp_func(gamma_1a, R_1a)

# rho1_1b = rho_func(p1_1b, R_1b, T1_1b)
# print(f'rho1_1b: {rho1_1b:.2f}')

def task_1b(T1, M1, p1, gamma, R):
    
    Cp = Cp_func(gamma, R)
    rho1 = rho_func(p1, R, T1)
    print(f'rho1_1b: {rho1:.2f}')    
    
    T01 = T0_func(T1, M1)
    T02_star = T02_func(T01, M1, M2=1)
    print(f'T02_star: {T02_star:.2f}')
    q_star = q_func(T01, T02_star, Cp)
    print(f'q_star: {q_star:.2f}')
    q = 0.9 * q_star
    print(f'q: {q:.2f}')
    T02 = T02_from_q_func(q, T01, Cp)
    
    T01_T0_star = 7.93388e-01
    T02_T0_star = (T02 / T01) * T01_T0_star
    print(f'T02_T0_star: {T02_T0_star:.4f}')
    # From Table A.3:
    # 1.18000e+00 8.13736e-01 9.22000e-01 8.82577e-01 1.01573e+00 9.82299e-01
    # 1.20000e+00 7.95756e-01 9.11848e-01 8.72685e-01 1.01942e+00 9.78717e-01
    M2 = interpolation(9.82299e-01, 1.18000e+00, 
                       9.78717e-01, 1.20000e+00, 
                       T02_T0_star)
    print(f'M2: {M2:.2f}')
    
    T2 = T2_func(T1, M1, M2)
    p2 = p2_func(p1, M1, M2)
    rho2 = rho2_func(rho1, M1, M2)
    print(f'T2: {T2:.2f}')
    print(f'p2: {p2:.2f}')
    print(f'rho2: {rho2:.2f}')
    
task_1b(T1_1b, M1_1b, p1_1b, gamma_1b, R_1b)
#%%
# Task 2.a
printt('Task 2.a:')
D_2a = 0.3
fbar_2a = 0.005
M1_2a = 0.2 
T1_2a = 293
p1_2a = 1 * bar_to_Pa
gamma_2a = gamma_air
R_2a = R_univ

def task_2a(T1, M1, p1, D, fbar, gamma, R):
    
    rho1 = rho_func(p1, R, T1)
    print(f'rho1_1b: {rho1:.2f}')    
    
    frac_4fLstar1_D = 1.45333e+01 # from A.4, M1 = 0.2
    Lstar1 = frac_4fLstar1_D * D / (4 * fbar)
    print(f'Lstar1: {Lstar1:.2f}')
    
    L = 0.8 * Lstar1
    print(f'L: {L:.2f}')
    Lstar2 = Lstar1 - L
    print(f'Lstar2: {Lstar2:.2f}')
    
    frac_4fLstar2_D = 4 * fbar * Lstar2 / D
    print(f'fraction: {frac_4fLstar2_D:.4f}')
    # From Table A.4:
    # 3.60000e-01 1.16968e+00 3.00422e+00 2.56841e+00 1.73578e+00 3.18012e+00
    # 3.80000e-01 1.16632e+00 2.84200e+00 2.43673e+00 1.65870e+00 2.70545e+00
    M2 = interpolation(3.18012e+00, 3.60000e-01, 
                       2.70545e+00, 3.80000e-01, 
                       frac_4fLstar2_D)
    print(f'M2: {M2:.2f}')
    
    T2 = T2_fric_func(T1, M1, M2)
    p2 = p2_fric_func(p1, M1, M2)
    rho2 = rho2_fric_func(rho1, M1, M2)
    print(f'T2: {T2:.2f}')
    print(f'p2: {p2:.2f}')
    print(f'rho2: {rho2:.2f}')
    
task_2a(T1_2a, M1_2a, p1_2a, D_2a, fbar_2a, gamma_2a, R_2a)

#%%
# Task 2.b
printt('Task 2.b:')
D_2b = 0.3
fbar_2b = 0.005
M1_2b = 2.0
T1_2b = 293
p1_2b = 1 * bar_to_Pa
gamma_2b = gamma_air
R_2b = R_univ

def task_2b(T1, M1, p1, D, fbar, gamma, R):
    
    rho1 = rho_func(p1, R, T1)
    print(f'rho1_1b: {rho1:.2f}')    
    
    frac_4fLstar1_D = 3.04997e-01 # from A.4, M1 = 2.0
    Lstar1 = frac_4fLstar1_D * D / (4 * fbar)
    print(f'Lstar1: {Lstar1:.2f}')
    
    L = 0.8 * Lstar1
    print(f'L: {L:.2f}')
    Lstar2 = Lstar1 - L
    print(f'Lstar2: {Lstar2:.2f}')
    
    frac_4fLstar2_D = 4 * fbar * Lstar2 / D
    print(f'fraction: {frac_4fLstar2_D:.4f}')
    # From Table A.4:
    # 1.28000e+00 9.03832e-01 7.42735e-01 8.21762e-01 1.05810e+00 5.82014e-02
    # 1.30000e+00 8.96861e-01 7.28483e-01 8.12258e-01 1.06630e+00 6.48321e-02
    M2 = interpolation(5.82014e-02, 1.28000e+00, 
                       6.48321e-02, 1.30000e+00, 
                       frac_4fLstar2_D)
    print(f'M2: {M2:.2f}')
    
    T2 = T2_fric_func(T1, M1, M2)
    p2 = p2_fric_func(p1, M1, M2)
    rho2 = rho2_fric_func(rho1, M1, M2)
    print(f'T2: {T2:.2f}')
    print(f'p2: {p2:.2f}')
    print(f'rho2: {rho2:.2f}')
    
task_2b(T1_2b, M1_2b, p1_2b, D_2b, fbar_2b, gamma_2b, R_2b)

#%%
# Task 2.c
printt('Task 2.c:')
D_2c = 0.3
fbar_2c = 0.005
M1_2c = 2.0
T1_2c = 293
p1_2c = 1 * bar_to_Pa
gamma_2c = gamma_air
R_2c = R_univ

def task_2c(T1, M1, p1, D, fbar, gamma, R):
    
    rho1 = rho_func(p1, R, T1)
    print(f'rho1_1b: {rho1:.2f}')    
    
    frac_4fLstar1_D = 3.04997e-01 # from A.4, M1 = 2.0
    Lstar1 = frac_4fLstar1_D * D / (4 * fbar)
    print(f'Lstar1: {Lstar1:.2f}')
    
    L = 1.5 * Lstar1
    print(f'L: {L:.2f}')
    
task_2c(T1_2c, M1_2c, p1_2c, D_2c, fbar_2c, gamma_2c, R_2c)

#%%