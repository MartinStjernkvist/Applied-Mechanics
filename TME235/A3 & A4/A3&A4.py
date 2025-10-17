#%%
# %matplotlib widget

import numpy as np
import matplotlib.pyplot as plt
from sympy import *
import math
import pandas as pd

from IPython.display import display, Math
from mpl_toolkits.mplot3d import axes3d
from mpl_toolkits.mplot3d import Axes3D
from numpy.random import rand
from IPython.display import HTML
from matplotlib import animation
import scipy.io as sio
from scipy.optimize import fsolve
from matplotlib import rcParams # for changing default values
import matplotlib.ticker as ticker

import calfem.core as cfc
import calfem.vis_mpl as cfv
import calfem.mesh as cfm
import calfem.utils as cfu

from scipy.sparse import coo_matrix, csr_matrix
import matplotlib.cm as cm

from pathlib import Path

def new_prob(string):
    print_string = '\n--------------------------------------------\n' + 'Assignment ' + str(string) + '\n--------------------------------------------\n'
    return print(print_string)

SMALL_SIZE = 10
MEDIUM_SIZE = 12
BIGGER_SIZE = 14
dpi = 500

# Set the global font sizes
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.rc('figure', figsize=(8,4))

script_dir = Path(__file__).parent

def sfig(fig_name):
    fig_output_file = script_dir / "figures" / fig_name
    fig_output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_output_file, dpi=dpi, bbox_inches='tight')
    
def fig(fig_name):
    '''
    standard matplotlib commands
    '''
    plt.legend()
    plt.grid(True, alpha = 0.3)
    sfig(fig_name)
    plt.show()

#%%
####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################



# GIVEN INFORMATION



####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################

# --------------------------------------------
# ASSIGNMENT 3 - Axisymmetric disc
# --------------------------------------------
q_num = 15 * 10**6 # Pa
b_radius = 0.3 # m
a_radius = 0.1 # m
h_num = a_radius / 4
E_num = 210 * 10**9 # Pa
nu_num = 0.3
sigma_y_num = 400 * 10**6 # Pa
F_num = 0

# --------------------------------------------
# ASSIGNMENT 4 - Body interactions
# --------------------------------------------

L = 2 # m
h2 = 0.05 # m
b2 = h2
E_steel = 210 * 10**9 # Pa
nu2 = 0.3
E_rubber = 20 * 10**6 # Pa
nu_rubber = 0.45

#%%
####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################



# Assignment 3 - numerical



####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################
new_prob('3 - numerical')

# define symbols
r, q, b, a, E, nu, h, z, A1, A2, A3, A4 = symbols('r q b a E nu h z A1 A2 A3 A4', real = True)

# bending stiffness
D = E * h**3 / (12 * (1 - nu**2))

# deflection field
w = integrate(1 /r * integrate(r * integrate(1 / r * integrate(q * r / D, r), r), r), r) + A1 * r**2 * log(r / b) + A2 * r**2 + A3 * log(r / b) + A4

# radial and circumferential bending moment
M_r = D *(- w.diff(r, 2) - nu * w.diff(r, 1) / r)
M_phi = D * (-w.diff(r, 1) / r - nu * w.diff(r, 2))

# shear force field
V = M_r.diff(r, 1) +  1 / r * (M_r - M_phi)

sigma_rr = E / (1 - nu**2) * ((-z * w.diff(r, 2)) + nu * (-z * w.diff(r, 1) / r))
sigma_phiphi = E / (1 - nu**2) * (nu * (- z * w.diff(r, 2)) + (- z * w.diff(r, 1) / r))

boundary_conditions = [
    M_r.subs(r, a),  # inner boundary radial bending moment free
    w.subs(r, b), # outer boundary zero deflection
    M_r.subs(r, b), # outer boundary radial bending moment free
    V.subs(r, a)
]

unknowns = (A1, A2, A3, A4)

# solve for the integration constants through the boundary conditions
integration_constants = solve(boundary_conditions, unknowns, real = True)
print('integration constants:')
display(integration_constants)

# displacement function
w_solution = simplify(w.subs(integration_constants))
print('w(r): ')
display(w_solution)
w_func = lambdify((r, q, b, a, E, nu, h), w_solution, 'numpy')
print(w_func)

# normal stress function
sigma_rr_solution = simplify(sigma_rr.subs(integration_constants))
print('sigma_rr: ')
display(sigma_rr_solution)
sigma_rr_func = lambdify((r, q, b, a, E, nu, h, z), sigma_rr_solution, 'numpy')
print(sigma_rr_func)

# circumferential stress function
sigma_phiphi_solution = simplify(sigma_phiphi.subs(integration_constants))
print('sigma_phiphi: ')
display(sigma_phiphi_solution)
sigma_phiphi_func = lambdify((r, q, b, a, E, nu, h, z), sigma_phiphi_solution, 'numpy')
print(sigma_phiphi_func)

def numerical_analysis(a, h):
    
    r_vals = np.linspace(a, b_radius, 401)
    z_num = h / 2

    w_vals = w_func(r_vals, q_num, b_radius, a, E_num, nu_num, h)
    sigma_rr_vals = sigma_rr_func(r_vals, q_num, b_radius, a, E_num, nu_num, h, z_num)
    sigma_phiphi_vals = sigma_phiphi_func(r_vals, q_num, b_radius, a, E_num, nu_num, h, z_num)
    
    sigma_vm_vals = np.sqrt((sigma_rr_vals - sigma_phiphi_vals)**2)

    plt.figure()
    plt.plot(r_vals, w_vals, label=fr'normalized values, a: {a/a_radius}, h: {h/h_num}')
    plt.axvline(a, color='black', linestyle='--', label='a')
    plt.axvline(b_radius, color='grey', linestyle='--', label='b')
    plt.title('Radial deflection')
    plt.xlabel('r [m]')
    plt.ylabel('w [m]')
    fig('radial deflection a' + str(int(a_radius/a)) + 'h' + str(int(h_num/h)))

    plt.figure()
    plt.plot(r_vals, sigma_rr_vals, color='red', label=fr'normalized values, a: {a/a_radius}, h: {h/h_num}')
    plt.axvline(a, color='black', linestyle='--', label='a')
    plt.axvline(b_radius, color='grey', linestyle='--', label='b')
    plt.axhline(sigma_y_num, color='orange', linestyle='--', label='yield strength')
    plt.title('Radial stress')
    plt.xlabel('r [m]')
    plt.ylabel('sigma [Pa]')
    fig('radial stress a' + str(int(a_radius/a)) + 'h' + str(int(h_num/h)))
    
    plt.figure()
    plt.plot(r_vals, sigma_vm_vals, color='green', label=fr'normalized values, a: {a/a_radius}, h: {h/h_num}')
    plt.axvline(a, color='black', linestyle='--', label='a')
    plt.axvline(b_radius, color='grey', linestyle='--', label='b')
    plt.axhline(sigma_y_num, color='orange', linestyle='--', label='yield strength')
    plt.title('von Mises stress')
    plt.xlabel('r [m]')
    plt.ylabel('sigma [Pa]')
    fig('von mises stress a' + str(int(a_radius/a)) + 'h' + str(int(h_num/h)))
    
    return r_vals, w_vals, sigma_rr_vals, sigma_vm_vals
    
a1h1_r_vals, a1h1_w_vals, a1h1_sigma_rr_vals, a1h1_sigma_vm_vals = numerical_analysis(a_radius, h_num)
a2h1_r_vals, a2h1_w_vals, a2h1_sigma_rr_vals, a2h1_sigma_vm_vals = numerical_analysis(a_radius/2, h_num)
a1h2_r_vals, a1h2_w_vals, a1h2_sigma_rr_vals, a1h2_sigma_vm_vals = numerical_analysis(a_radius, h_num/2)

#%%

####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################



# Assignment 3 - abaqus



####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################
new_prob('3 - abaqus')

def read_abaqus_data(results_file):
    
    """
    Parse Abaqus .rpt file with multiple columns.
    """
    with open(results_file, 'r') as f:
        lines = f.readlines()
    
    # Find header line (starts with X and contains column names)
    header_line = None
    data_start_idx = None
    
    for i, line in enumerate(lines):
        tokens = line.strip().split()
        if len(tokens) >= 2 and tokens[0] == 'X':
            header_line = tokens
            data_start_idx = i + 1
            break
    
    if header_line is None:
        raise ValueError("Could not find header line starting with 'X'")
    
    print(f"Found header: {header_line}")
    print(f"Number of columns: {len(header_line)}")
    
    # Parse data
    data = []
    for line in lines[data_start_idx:]:
        line = line.strip()
        if not line:  # Skip empty lines
            continue
        try:
            # Replace 'E' with 'e' for scientific notation and split
            values = [float(x.replace('E', 'e')) for x in line.split()]
            if len(values) == len(header_line):  # Match number of columns
                data.append(values)
        except ValueError:
            # Skip lines that can't be parsed as numbers
            continue
    
    data = np.array(data)
    print(f"Parsed {len(data)} data rows")
    
    # Create dictionary with column names
    datasets = {}
    for i, col_name in enumerate(header_line):
        datasets[col_name] = data[:, i]
    
    print(f"\nAvailable columns:")
    for name in datasets.keys():
        print(f"  - '{name}'")
    print()
    
    # Extract specific columns
    X = datasets['X']
    u2 = datasets['u2_mid_X']
    s11 = datasets['s11_top_X']
    svm = datasets['svm_top_X']
    
    # datasets = {}
    # current_data = []
    # current_name = None
    
    # with open(results_file, 'r') as f:
    #     for line in f:
    #         # Detect dataset header: line with at least two words, first is 'X'
    #         tokens = line.strip().split()
    #         if len(tokens) >= 2 and tokens[0] == 'X':
    #             # Save previous dataset
    #             if current_name and current_data:
    #                 datasets[current_name] = np.array(current_data)
    #             # Start new dataset
    #             current_name = ' '.join(tokens)
    #             current_data = []
    #         elif line.strip() and not (line.strip().startswith('X') or line.strip() == ''):
    #             # Try to parse data lines
    #             try:
    #                 values = [float(x.replace('E', 'e')) for x in line.split()]
    #                 if len(values) == 2:
    #                     current_data.append(values)
    #             except Exception:
    #                 pass  # skip lines that can't be parsed
    #     # Save last dataset
    #     if current_name and current_data:
    #         datasets[current_name] = np.array(current_data)

    # print("Available datasets from" + results_file)
    # for name in datasets.keys():
    #     print(f"  - '{name}'")
    # print()
    
    # X = datasets['X u2_mid_X'][:, 0]
    # u2 = datasets['X u2_mid_X'][:, 1]
    # s11 = datasets['X s11_top_X'][:, 1]
    # svm = datasets['X svm_top_X'][:, 1]
    
    plt.figure()
    plt.plot(X, u2, color='blue', label=results_file)
    # plt.axvline(a, color='black', linestyle='--', label='a')
    # plt.axvline(b_radius, color='grey', linestyle='--', label='b')
    plt.title('Radial deflection')
    plt.xlabel('r [m]')
    plt.ylabel('w [m]')
    fig(results_file.strip('.rpt') + 'radial deflection a')

    plt.figure()
    plt.plot(X, s11, color='red', label=results_file)
    # plt.axvline(a, color='black', linestyle='--', label='a')
    # plt.axvline(b_radius, color='grey', linestyle='--', label='b')
    plt.axhline(sigma_y_num, color='orange', linestyle='--', label='yield strength')
    plt.title('Radial stress')
    plt.xlabel('r [m]')
    plt.ylabel('sigma [Pa]')
    fig(results_file.strip('.rpt') + 'radial stress a')
    
    plt.figure()
    plt.plot(X, svm, color='green', label=results_file)
    # plt.axvline(a, color='black', linestyle='--', label='a')
    # plt.axvline(b_radius, color='grey', linestyle='--', label='b')
    plt.axhline(sigma_y_num, color='orange', linestyle='--', label='yield strength')
    plt.title('von Mises stress')
    plt.xlabel('r [m]')
    plt.ylabel('sigma [Pa]')
    fig(results_file.strip('.rpt') + 'von mises stress a')
    
    return datasets, X, u2, s11, svm

_, a1h1_X, a1h1_u2, a1h1_s11, a1h1_svm = read_abaqus_data('a1h1_results.rpt') # left bottom support
_, a1h2_X, a1h2_u2, a1h2_s11, a1h2_svm = read_abaqus_data('a1h2_results.rpt') # left bottom support
_, a2h1_X, a2h1_u2, a2h1_s11, a2h1_svm = read_abaqus_data('a2h1_results.rpt') # left bottom support
# _, a1h1_X, a1h1_u2, a1h1_s11, a1h1_svm = read_abaqus_data('a1h1_left_mid_results.rpt') # left middle support
_, a1h1_radialfix_X, a1h1_radialfix_u2, a1h1_radialfix_s11, a1h1_radialfix_svm = read_abaqus_data('a1h1_radialfix_results.rpt') # right bottom, radial fixed support

#%%


####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################



# Assignment 3 - comparison plot



####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################
new_prob('3 - comparison plot')


# plt.figure()
#     plt.plot(a1h1_r_vals, a1h1_w_vals, color='blue', label='numerical')
#     plt.plot(X, u2, color='blue', label ='abaqus')
#     # plt.axvline(a, color='black', linestyle='--', label='a')
#     # plt.axvline(b_radius, color='grey', linestyle='--', label='b')
#     plt.title('Radial deflection')
#     plt.xlabel('r [m]')
#     plt.ylabel('w [m]')
#     fig('comparison radial deflection')

#     plt.figure()
#     plt.plot(X, s11, color='red', label=results_file)
#     # plt.axvline(a, color='black', linestyle='--', label='a')
#     # plt.axvline(b_radius, color='grey', linestyle='--', label='b')
#     plt.axhline(sigma_y_num, color='orange', linestyle='--', label='yield strength')
#     plt.title('Radial stress')
#     plt.xlabel('r [m]')
#     plt.ylabel('sigma [Pa]')
#     fig('comparison radial stress')
    
#     plt.figure()
#     plt.plot(X, svm, color='green', label=results_file)
#     # plt.axvline(a, color='black', linestyle='--', label='a')
#     # plt.axvline(b_radius, color='grey', linestyle='--', label='b')
#     plt.axhline(sigma_y_num, color='orange', linestyle='--', label='yield strength')
#     plt.title('von Mises stress')
#     plt.xlabel('r [m]')
#     plt.ylabel('sigma [Pa]')
#     fig('comparison von mises stress')


#%%


####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################



# Assignment 4 - simulate rubber behaviour



####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################
# new_prob('4 - simulate rubber behaviour')

'''
# --------------------------------------------
# Example 17
# --------------------------------------------
# Define symbols
lambda_ = Symbol('lambda', positive=True)
Gmod, lambdamod, c2, c3, Kb = symbols('Gmod lambdamod c2 c3 Kb', real=True)

# Define deformation gradient F
F = Matrix([
    [lambda_, 0, 0],
    [0, 1/sqrt(lambda_), 0],
    [0, 0, 1/sqrt(lambda_)]
])

# Compute C = Fᵀ * F
C = F.T * F

# Compute determinant J = det(F)
J = F.det()

# Define S = Gmod*(I - C⁻¹) + lambdamod*log(det(F))*C⁻¹
S = Gmod * (eye(3) - C.inv()) + lambdamod * log(J) * C.inv()

# Compute sigma = (1/det(F)) * F * S * Fᵀ
sigma = (1/J) * F * S * F.T

# Optionally, simplify results
sigma_simplified = simplify(sigma)

# Display results
print("C =")
display(C)
print("\nJ =")
display(J)
print("\nS =")
display(S)
print("\nsigma =")
display(sigma_simplified)

# --------------------------------------------
# Own attempt
# --------------------------------------------
# Emod = 20 # Young's modulus
# v = 0.45 # Poisson's ratio
# G = Emod / (2 * (1 + v)) # Shear modulus
G_rubber = E_rubber / (2 * (1 + nu_rubber))

npoints = 1000
lambda_vals = np.linspace(0.1, 1.9, npoints)

sigma11 = np.zeros(npoints)

for i in range(npoints):
    sigma11[i] = G_rubber * (1 - 1/lambda_vals[i]**2)
    
plt.figure()
plt.plot(lambda_vals, sigma11, 'b-')
plt.xlabel('lambda')
plt.ylabel('σ₁₁ (stress)')
plt.title('Uniaxial stress response for natural rubber')
fig('rubber simulation')


# --- Material parameters for natural rubber ---
Emod = 20          # Young's modulus
v = 0.45           # Poisson's ratio
G = Emod / (2 * (1 + v))                           # Shear modulus
lmbda = v * Emod / ((1 + v) * (1 - 2 * v))         # Lame's first parameter
c2 = -G / 10
c3 = G / 30
Kb = lmbda + 2 * G / 3

# --- Deformation setup ---
npoints = 1000
epsilon = np.linspace(-0.5, 1.5, npoints)

# --- Preallocate arrays ---
sigma11NH = np.zeros(npoints)
sigma11Y = np.zeros(npoints)

# --- Loop over strains ---
for i in range(npoints):
    # Deformation gradient
    F = np.eye(3)
    F[0, 0] += epsilon[i]
    
    # Right Cauchy–Green tensor
    C = F.T @ F
    J = np.linalg.det(F)
    Cinv = np.linalg.inv(C)
    
    # --- Neo-Hookean model ---
    S = G * (np.eye(3) - Cinv) + lmbda * np.log(J) * Cinv
    sigma = (1 / J) * F @ S @ F.T
    sigma11NH[i] = sigma[0, 0]
    
    # --- Yeoh model ---
    I1 = np.trace(C)
    S = (G * (np.eye(3) - Cinv)
         + lmbda * np.log(J) * Cinv
         + (4 * c2 * (I1 - 3) + 6 * c3 * (I1 - 3)**2) * np.eye(3))
    sigma = (1 / J) * F @ S @ F.T
    sigma11Y[i] = sigma[0, 0]

# --- Plot results ---
plt.plot(epsilon, sigma11NH, 'b-', label='Neo-Hookean')
plt.plot(epsilon, sigma11Y, 'r--', label='Yeoh')
plt.xlabel('Engineering strain ε')
plt.ylabel('σ₁₁ (stress)')
plt.legend()
plt.grid(True)
plt.title('Uniaxial stress–strain response for natural rubber')
plt.show()
'''


npoints = 1000
lambda_vals = np.linspace(0.1, 1.9, npoints)

# G = Emod / (2 * (1 + v)) # Shear modulus
G_rubber = E_rubber / (2 * (1 + nu_rubber))

# Two formulations
sigma_eq96 = G_rubber * (lambda_vals**2 - 1)  # Eq. (9.6)
# sigma_uniaxial = G_rubber * (lambda_vals**2 - 1/lambda_vals)  # Uniaxial stress

# Deviatoric stress (should be identical)
sigma22_eq96 = G_rubber * (1 / lambda_vals - 1)
mean_96 = (sigma_eq96 + 2 * sigma22_eq96) / 3
dev_96 = sigma_eq96 - mean_96
# dev_uni = sigma_uniaxial - sigma_uniaxial / 3

plt.figure()
plt.plot(lambda_vals, sigma_eq96, 'b-', label='Eq. (9.6)')
# plt.plot(lambda_vals, sigma_uniaxial, 'r--', label='Uniaxial')
plt.axhline(0, color='k', linestyle='--', alpha=0.3)
plt.axvline(1, color='k', linestyle='--', alpha=0.3)
plt.xlabel('Stretch λ')
plt.ylabel('Stress σ₁₁ (Pa)')
plt.title('Material behaviour')
fig('Material behaviour')

plt.figure()
plt.plot(lambda_vals, dev_96, 'b-', linewidth=2.5, label='Eq. (9.6) deviatoric')
# plt.plot(lambda_vals, dev_uni, 'r--', label='Uniaxial deviatoric')
plt.axhline(0, color='k', linestyle='--', alpha=0.3)
plt.axvline(1, color='k', linestyle='--', alpha=0.3)
plt.xlabel('Stretch λ')
plt.ylabel('Deviatoric Stress (Pa)')
plt.title('Deviatoric Stress')
fig('Deviatoric Stress')


'''
# Incompressible neo-Hookean: σ₁₁ = G(λ² - λ⁻¹)
# This assumes uniaxial stress state (σ₂₂ = σ₃₃ = 0)
sigma11 = G_rubber * (lambda_vals**2 - 1/lambda_vals)

# Plot stress-stretch response
plt.figure(figsize=(10, 7))
plt.plot(lambda_vals, sigma11, 'b-', linewidth=2.5)
plt.axhline(0, color='k', linestyle='--', linewidth=1, alpha=0.5)
plt.axvline(1, color='k', linestyle='--', linewidth=1, alpha=0.5)
plt.grid(True, alpha=0.3)
plt.xlabel('Stretch λ₁', fontsize=12)
plt.ylabel('Cauchy Stress σ₁₁ (MPa)', fontsize=12)
plt.title('Incompressible Neo-Hookean Rubber: Uniaxial Loading', fontsize=14, fontweight='bold')
plt.xlim([0.1, 1.9])

# Annotations
plt.text(0.4, -10, 'Compression', fontsize=11, ha='center', color='red')
plt.text(1.5, 15, 'Tension', fontsize=11, ha='center', color='green')
plt.text(1.0, -2, 'λ = 1\n(undeformed)', fontsize=10, ha='center')
'''
#%%


####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################



# Assignment 4 - neo-Hooke parameters



####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################
new_prob('4 - neo-Hooke parameters')

lam = nu_rubber * E_rubber / ((1 + nu_rubber) * (1 - 2 * nu_rubber))
mu = E_rubber / (2 * (1 + nu_rubber))
G = mu

C_10 = G / 2
_D_1 = lam / 2

print('C_10: ', f'{C_10:.0f}')
print('D_1: 1/', f'{_D_1:.0f}')

traction = 10000 / (b2 * h2)
print(traction)
print(10_000 / (0.05 * 0.05))

#%%


####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################



# Assignment 4 - abaqus, P = 10 kN, P = -10 kN



####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################
new_prob('4 - abaqus, P = 10 kN, P = -10 kN')

def read_abaqus_data_2(results_file):
    
    datasets = {}
    current_data = []
    current_name = None
    
    with open(results_file, 'r') as f:
        for line in f:
            # Detect dataset header: line with at least two words, first is 'X'
            tokens = line.strip().split()
            if len(tokens) >= 2 and tokens[0] == 'X':
                # Save previous dataset
                if current_name and current_data:
                    datasets[current_name] = np.array(current_data)
                # Start new dataset
                current_name = ' '.join(tokens)
                current_data = []
            elif line.strip() and not (line.strip().startswith('X') or line.strip() == ''):
                # Try to parse data lines
                try:
                    values = [float(x.replace('E', 'e')) for x in line.split()]
                    if len(values) == 2:
                        current_data.append(values)
                except Exception:
                    pass  # skip lines that can't be parsed
        # Save last dataset
        if current_name and current_data:
            datasets[current_name] = np.array(current_data)

    print("Available datasets from" + results_file)
    for name in datasets.keys():
        print(f"  - '{name}'")
    print()
    
    X = datasets['X u2_mid_X'][:, 0]
    u2 = datasets['X u2_mid_X'][:, 1]
    
    plt.figure()
    plt.plot(X, u2, color='blue', label=results_file)
    plt.title('Deflection')
    plt.xlabel('r [m]')
    plt.ylabel('w [m]')
    fig(results_file.strip('.rpt') + 'deflection')
    
    return datasets, X, u2

read_abaqus_data_2('a4_p_10_results.rpt')
read_abaqus_data_2('a4_p_minus_10_results.rpt')

#%%

####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################



# Assignment 4 - comparison with euler-bernoulli



####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################
new_prob('4 - comparison with euler-bernoulli')


# Beam properties
I = (b2 * h2**3) / 12  # Second moment of area
EI = E_steel * I  # Bending stiffness


print("ANALYTICAL SOLUTIONS: EULER-BERNOULLI BEAM")

# Example load (you should replace with your actual load)
# Assuming distributed load q or point load P
q = 0  # N/m (example distributed load)
P = 10 * 10**3  # N (example point load at end)

# ============================================================================
# CASE 1: Cantilever beam WITHOUT support (free end)
# ============================================================================
print("CASE 1: Cantilever WITHOUT Support")

delta_point_no_support = (P * L**3) / (3 * EI)

print(f"  δ_max = PL³/(3EI) = {delta_point_no_support:.2f} m")

# ============================================================================
# CASE 2: Cantilever beam WITH rubber support
# ============================================================================
print("CASE 2: Cantilever WITH Rubber Support")

# APPROACH A: Linear elastic spring at the support location
print("\n--- Approach A: Elastic Spring Support ---")

# Rubber support dimensions
h_rubber = 3 * h2  # m (height of rubber support)
A_rubber = 4 * L / 20 * h2  # m² (contact area)

# Spring stiffness: k = EA/L for compression
k_spring = (E_rubber * A_rubber) / h_rubber

print(f"  Spring stiffness k = EA/h = {k_spring:.2f} N/m")

# For cantilever with spring support at tip under point load P:
# δ = P/(k + 3EI/L³)
# This comes from: k*δ + (P - k*δ) causes beam deflection
delta_with_spring = P / (k_spring + 3*EI/L**3)

print(f"  Deflection with spring support: δ = {delta_with_spring:.2f} m")
print(f"  Reduction: {(1 - delta_with_spring/delta_point_no_support)*100:.1f}%")


# For cantilever with roller support at distance a from fixed end
a = L  # Support at tip
# Deflection is essentially zero at support point
delta_rigid_support = 0.0

print(f"  Deflection at support ≈ {delta_rigid_support} mm (rigid)")
print("  This is an upper bound on support stiffness effect")

# ============================================================================
# VISUALIZATION
# ============================================================================
print("DEFLECTION COMPARISON")

x = np.linspace(0, L, 100)

# Deflection shapes (for point load P at tip)
# Without support
w_no_support = (P / (6*EI)) * x**2 * (3*L - x)

# With spring support (approximate)
# This is approximate - exact solution more complex
delta_tip_spring = delta_with_spring
w_with_spring = w_no_support * (delta_tip_spring / delta_point_no_support)

# With rigid support
w_rigid = np.zeros_like(x)

# Plot
plt.figure()
plt.plot(x, w_no_support, 'b-', label='No support')
plt.plot(x, w_with_spring, 'r--', label=f'Rubber support (k={k_spring/1e6:.1f} MN/m)')
plt.plot(x, w_rigid, 'g:', label='Rigid support (ideal)')
plt.xlabel('Position x (m)')
plt.ylabel('Deflection (m)')
plt.title('Beam Deflection: Comparison of Support Conditions')

plt.axhline(0, color='k')

# Mark fixed end and support location
plt.plot(0, 0, 'ks', label='Fixed end')
plt.plot(L, 0, 'ro', label='Support location')
fig('Beam Deflection Comparison of Support Conditions')

# ============================================================================
# SUMMARY TABLE
# ============================================================================
print("\nSummary of tip deflections:")
print(f"{'Case':<30} | {'Deflection (m)':<15} | {'Ratio':<10}")
print(f"{'No support':<30} | {delta_point_no_support:>15.2f} | {1.0:>10.2f}")
print(f"{'With rubber (spring k)':<30} | {delta_with_spring:>15.2f} | {delta_with_spring/delta_point_no_support:>10.2f}")
















L=2
a = 11/20 * L        # spring location [m]
P = 10 * 10**3      # end load [N]
n = 200        # number of nodes

I = (b2 * h2**3) / 12  # Second moment of area

# Rubber support dimensions
h_rubber = 3 * h2  # m (height of rubber support)
A_rubber = 4 * L / 20 * h2  # m² (contact area)

# Spring stiffness: k = EA/L for compression
k_spring = (E_rubber * A_rubber) / h_rubber

# Discretization
x = np.linspace(0, L, n)
print('x: ', x)
dx = x[1] - x[0]
EI = E_steel * I

# Finite difference matrix for 4th derivative
D4 = np.zeros((n, n))
for i in range(2, n-2):
    D4[i, i-2:i+3] = [1, -4, 6, -4, 1]
D4 /= dx**4

# Apply boundary conditions
A = EI * D4
b = np.zeros(n)

# Fixed end: w(0)=0, w'(0)=0
A[0, :] = 0
A[0, 0] = 1
A[1, :] = 0
A[1, 0:3] = [-3, 4, -1] / (2*dx)

# Free end: w''(L)=0, w'''(L)=-P/EI
A[-2, :] = 0
A[-2, -5:] = [1, -2, 1, 0, 0] / dx**2
A[-1, :] = 0
A[-1, -5:] = [-1, 3, -3, 1, 0] / dx**3
b[-1] = -P / EI

# Add spring condition: R_s = k * w(a)
idx_a = np.argmin(np.abs(x - a))
A[idx_a, idx_a] += k_spring / EI  # acts like a distributed spring

# Solve system
w = np.linalg.solve(A, b)

# Plot
plt.figure()
plt.plot(x, w)
plt.xlabel('x [m]')
plt.ylabel('Deflection [m]')
plt.title('Cantilever with Spring Support')
fig('Cantilever with Spring Support')

# Reaction at the spring
R_s = k_spring * w[idx_a]
print(f"Spring reaction force = {R_s:.3f} N")
print(f"Tip deflection = {w[-1]:.6e} m")


"""
CANTILEVER BEAM WITH ROLLER SUPPORT
====================================
Beam configuration:
- Fixed support at x = 0
- Roller support at x = a
- Point load P at x = L (free end)

Method: Solve using Euler-Bernoulli beam equation
EI * d⁴w/dx⁴ = q(x)
"""

# ============================================================================
# BEAM PARAMETERS
# ============================================================================
L = 2.0        # Total beam length [m]
a = 1.5        # Roller support position [m]
P = 5000.0     # Point load at free end [N]
E = 210e9      # Young's modulus [Pa] (steel)
h = 0.05       # Beam height [m]
b = 0.05       # Beam width [m]

I = (b * h**3) / 12  # Second moment of area [m⁴]
EI = E * I           # Bending stiffness [N·m²]

print("="*70)
print("CANTILEVER BEAM WITH ROLLER SUPPORT - NUMERICAL SOLUTION")
print("="*70)
print(f"Beam length L = {L} m")
print(f"Roller support at a = {a} m")
print(f"Point load P = {P} N at x = {L} m")
print(f"Bending stiffness EI = {EI:.2e} N·m²")

# ============================================================================
# METHOD 1: SUPERPOSITION (ANALYTICAL-NUMERICAL HYBRID)
# ============================================================================
print("\n" + "="*70)
print("METHOD 1: SUPERPOSITION")
print("="*70)

def solve_superposition(L, a, P, EI):
    """
    Solve using superposition:
    1. Cantilever with load P at tip → deflection w₁(x)
    2. Find reaction R at roller such that w₁(a) + w₂(a) = 0
    3. w₂(x) is deflection due to reaction R at position a
    """
    
    # Step 1: Find roller reaction R
    # Deflection of cantilever at x=a due to P at x=L:
    # w₁(a) = P/(6EI) * a² * (3L - a)
    
    # Deflection at x=a due to upward force R at x=a:
    # For cantilever with point load at distance 'a', deflection at load point:
    # w₂(a) = -R*a³/(3EI)
    
    # Boundary condition: w₁(a) + w₂(a) = 0
    # P/(6EI) * a² * (3L - a) - R*a³/(3EI) = 0
    
    R = P * (3*L - a) / (2*a)
    
    print(f"\nRoller reaction R = {R:.2f} N")
    print(f"Check: Fixed end reaction = {P - R:.2f} N")
    print(f"       Fixed end moment = {P*L - R*a:.2f} N·m")
    
    # Step 2: Compute deflection along beam
    x = np.linspace(0, L, 200)
    w = np.zeros_like(x)
    
    for i, xi in enumerate(x):
        # Region 1: 0 ≤ x ≤ a (before roller)
        if xi <= a:
            # Deflection due to P at x=L
            w1 = (P / (6*EI)) * xi**2 * (3*L - xi)
            # Deflection due to R at x=a
            w2 = -(R / (6*EI)) * xi**2 * (3*a - xi)
            w[i] = w1 + w2
            
        # Region 2: a < x ≤ L (after roller)
        else:
            # Deflection due to P at x=L
            w1 = (P / (6*EI)) * xi**2 * (3*L - xi)
            # Deflection due to R at x=a (now in other segment)
            w2 = -(R / (6*EI)) * (a**2 * (3*xi - a))
            w[i] = w1 + w2
    
    return x, w, R

x_super, w_super, R_super = solve_superposition(L, a, P, EI)

print(f"\nDeflection at roller (should be ~0): {np.interp(a, x_super, w_super)*1000:.6f} mm")
print(f"Deflection at tip: {w_super[-1]*1000:.4f} mm")























init_printing()

"""
CANTILEVER BEAM WITH ROLLER SUPPORT - SYMBOLIC SOLUTION
========================================================
Configuration:
- Fixed at x = 0
- Roller support at x = a (unknown reaction R)
- Point load P at x = L

Strategy: Solve in two segments
- Segment 1: 0 ≤ x ≤ a (with reaction R at x=a)
- Segment 2: a ≤ x ≤ L (after roller)
"""

print("="*80)
print("SYMBOLIC SOLUTION: CANTILEVER WITH ROLLER SUPPORT")
print("="*80)

# Define symbols
x, L, a, P, E, I, R = symbols('x L a P E I R', real=True, positive=True)

# ============================================================================
# SEGMENT 1: 0 ≤ x ≤ a (before roller)
# ============================================================================
print("\n" + "="*80)
print("SEGMENT 1: 0 ≤ x ≤ a")
print("="*80)

w1 = Function('w1')(x)

# Differential equation: EI * d⁴w/dx⁴ = 0 (no distributed load)
diffeq1 = Eq(E * I * diff(w1, x, 4), 0)
print("\nDifferential equation:")
print(diffeq1)

# General solution
w1_general = dsolve(diffeq1, w1).rhs
print("\nGeneral solution w1(x):")
print(w1_general)

# Moment and shear for segment 1
M1 = -E * I * w1_general.diff(x, 2)
V1 = -E * I * w1_general.diff(x, 3)

# ============================================================================
# SEGMENT 2: a ≤ x ≤ L (after roller)
# ============================================================================
print("\n" + "="*80)
print("SEGMENT 2: a ≤ x ≤ L")
print("="*80)

w2 = Function('w2')(x)

# Differential equation: EI * d⁴w/dx⁴ = 0
diffeq2 = Eq(E * I * diff(w2, x, 4), 0)

# General solution
w2_general = dsolve(diffeq2, w2).rhs
print("\nGeneral solution w2(x):")
print(w2_general)

# Moment and shear for segment 2
M2 = -E * I * w2_general.diff(x, 2)
V2 = -E * I * w2_general.diff(x, 3)

# ============================================================================
# BOUNDARY CONDITIONS
# ============================================================================
print("\n" + "="*80)
print("BOUNDARY CONDITIONS")
print("="*80)

# We have 8 unknowns (C1, C2, C3, C4 for w1 and C5, C6, C7, C8 for w2)
# Need 8 equations:

# At x = 0 (fixed support):
BC1 = w1_general.subs(x, 0)  # w1(0) = 0
BC2 = w1_general.diff(x).subs(x, 0)  # w1'(0) = 0 (slope = 0)

# At x = a (roller support - continuity and support conditions):
BC3 = w1_general.subs(x, a)  # w1(a) = 0 (roller constrains deflection)
BC4 = Eq(w1_general.diff(x).subs(x, a), w2_general.diff(x).subs(x, a))  # θ1(a) = θ2(a) (slope continuity)
BC5 = Eq(M1.subs(x, a), M2.subs(x, a))  # M1(a) = M2(a) (moment continuity)
# Jump in shear at roller due to reaction R:
BC6 = Eq(V1.subs(x, a) - V2.subs(x, a), R)  # V1(a) - V2(a) = R

# At x = L (free end):
BC7 = M2.subs(x, L)  # M2(L) = 0 (no moment at free end)
BC8 = Eq(V2.subs(x, L), -P)  # V2(L) = -P (shear force = -P)

boundary_conditions = [BC1, BC2, BC3, BC4, BC5, BC6, BC7, BC8]

print("\nBoundary conditions:")
for i, bc in enumerate(boundary_conditions, 1):
    print(f"BC{i}: {bc}")

# ============================================================================
# SOLVE FOR INTEGRATION CONSTANTS AND REACTION R
# ============================================================================
print("\n" + "="*80)
print("SOLVING FOR INTEGRATION CONSTANTS...")
print("="*80)

# Get the constants from the general solutions
from sympy import symbols as syms
C1, C2, C3, C4, C5, C6, C7, C8 = syms('C1 C2 C3 C4 C5 C6 C7 C8', real=True)

# Substitute constant symbols into general solutions
w1_general = w1_general.subs({'C1': C1, 'C2': C2, 'C3': C3, 'C4': C4})
w2_general = w2_general.subs({'C1': C5, 'C2': C6, 'C3': C7, 'C4': C8})

# Recompute M and V with new constants
M1 = -E * I * w1_general.diff(x, 2)
V1 = -E * I * w1_general.diff(x, 3)
M2 = -E * I * w2_general.diff(x, 2)
V2 = -E * I * w2_general.diff(x, 3)

# Recreate boundary conditions with substituted constants
BC1 = w1_general.subs(x, 0)
BC2 = w1_general.diff(x).subs(x, 0)
BC3 = w1_general.subs(x, a)
BC4 = Eq(w1_general.diff(x).subs(x, a), w2_general.diff(x).subs(x, a))
BC5 = Eq(M1.subs(x, a), M2.subs(x, a))
BC6 = Eq(V1.subs(x, a) - V2.subs(x, a), R)
BC7 = M2.subs(x, L)
BC8 = Eq(V2.subs(x, L), -P)

boundary_conditions = [BC1, BC2, BC3, BC4, BC5, BC6, BC7, BC8]

# Solve for constants and R
print("Solving system of 8 equations for 8 unknowns + R...")
constants_solution = solve(boundary_conditions, [C1, C2, C3, C4, C5, C6, C7, C8, R])

print("\nIntegration constants and roller reaction:")
for key, val in constants_solution.items():
    print(f"{key} = {simplify(val)}")

# ============================================================================
# FINAL SOLUTIONS
# ============================================================================
print("\n" + "="*80)
print("FINAL DEFLECTION FUNCTIONS")
print("="*80)

# Substitute constants back
w1_solution = simplify(w1_general.subs(constants_solution))
w2_solution = simplify(w2_general.subs(constants_solution))
R_solution = simplify(constants_solution[R])

print("\nSegment 1 (0 ≤ x ≤ a):")
print(f"w1(x) = {w1_solution}")

print("\nSegment 2 (a ≤ x ≤ L):")
print(f"w2(x) = {w2_solution}")

print("\nRoller reaction:")
print(f"R = {R_solution}")

# Moment solutions
M1_solution = simplify(M1.subs(constants_solution))
M2_solution = simplify(M2.subs(constants_solution))

print("\nBending moments:")
print(f"M1(x) = {M1_solution}")
print(f"M2(x) = {M2_solution}")

# ============================================================================
# NUMERICAL EVALUATION
# ============================================================================
print("\n" + "="*80)
print("NUMERICAL EVALUATION")
print("="*80)

# Convert to numerical functions
w1_func = lambdify((x, L, a, P, E, I), w1_solution, 'numpy')
w2_func = lambdify((x, L, a, P, E, I), w2_solution, 'numpy')
R_func = lambdify((L, a, P, E, I), R_solution, 'numpy')
M1_func = lambdify((x, L, a, P, E, I), M1_solution, 'numpy')
M2_func = lambdify((x, L, a, P, E, I), M2_solution, 'numpy')

# Numerical parameters
L_val = 2.0        # m
a_val = 1.5        # m
P_val = 5000.0     # N
E_val = 210e9      # Pa
h_val = 0.05       # m
b_val = 0.05       # m
I_val = (b_val * h_val**3) / 12  # m⁴

print(f"\nNumerical parameters:")
print(f"L = {L_val} m")
print(f"a = {a_val} m")
print(f"P = {P_val} N")
print(f"EI = {E_val * I_val:.2e} N·m²")

# Calculate roller reaction
R_val = R_func(L_val, a_val, P_val, E_val, I_val)
print(f"\nRoller reaction R = {R_val:.2f} N")
print(f"Fixed support reaction = {P_val - R_val:.2f} N")

# Generate deflection curve
x1_vals = np.linspace(0, a_val, 100)
x2_vals = np.linspace(a_val, L_val, 100)

w1_vals = w1_func(x1_vals, L_val, a_val, P_val, E_val, I_val)
w2_vals = w2_func(x2_vals, L_val, a_val, P_val, E_val, I_val)

M1_vals = M1_func(x1_vals, L_val, a_val, P_val, E_val, I_val)
M2_vals = M2_func(x2_vals, L_val, a_val, P_val, E_val, I_val)

print(f"\nDeflection at roller: {w1_func(a_val, L_val, a_val, P_val, E_val, I_val)*1000:.6f} mm")
print(f"Deflection at tip: {w2_vals[-1]*1000:.4f} mm")

# ============================================================================
# PLOTTING
# ============================================================================
fig, axes = plt.subplots(2, 1, figsize=(12, 10))

# Plot 1: Deflection
axes[0].plot(x1_vals, w1_vals*1000, 'b-', linewidth=2.5, label='Segment 1 (0 ≤ x ≤ a)')
axes[0].plot(x2_vals, w2_vals*1000, 'r-', linewidth=2.5, label='Segment 2 (a ≤ x ≤ L)')
axes[0].axhline(0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
axes[0].axvline(a_val, color='g', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Roller at x={a_val} m')
axes[0].plot(0, 0, 'ks', markersize=15, label='Fixed support')
axes[0].plot(a_val, 0, 'go', markersize=12, label='Roller support')
axes[0].plot(L_val, w2_vals[-1]*1000, 'rv', markersize=12, label=f'Load P={P_val} N')
axes[0].set_xlabel('Position x [m]', fontsize=12)
axes[0].set_ylabel('Deflection w [mm]', fontsize=12)
axes[0].set_title('Beam Deflection (Symbolic Solution)', fontsize=14, fontweight='bold')
axes[0].grid(True, alpha=0.3)
axes[0].legend(fontsize=10)

# Plot 2: Bending Moment
axes[1].plot(x1_vals, M1_vals/1000, 'b-', linewidth=2.5, label='Segment 1')
axes[1].plot(x2_vals, M2_vals/1000, 'r-', linewidth=2.5, label='Segment 2')
axes[1].axhline(0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
axes[1].axvline(a_val, color='g', linestyle='--', linewidth=1.5, alpha=0.7)
axes[1].set_xlabel('Position x [m]', fontsize=12)
axes[1].set_ylabel('Bending Moment M [kN·m]', fontsize=12)
axes[1].set_title('Bending Moment Diagram', fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3)
axes[1].legend(fontsize=10)

plt.tight_layout()
plt.savefig('cantilever_roller_symbolic.png', dpi=300)
plt.show()

print("\n" + "="*80)
print("SOLUTION COMPLETE")
print("="*80)



#%%

####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################



# Assignment 4 - comparison with euler-bernoulli



####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################
new_prob('4 - comparison with euler-bernoulli')

x, L, q0, P, E, I = symbols('x L q0 P E I')

w = Function('w')(x) # w is a function of x

diffeq1 = Eq(E * I * diff(w, x, 4), q0)

w = dsolve(diffeq1, w).rhs

M = -E * I * w.diff(x, 2)

# Boundary conditions for distributed load
boundary_conditions = [ 
                        w.subs(x, 0),                               #w(0) = 0
                        w.diff(x).subs(x, 0),                       #w'(0) = 0
                        M.subs(x, 11 * L / 20),                     #w''(L) = 0
                        w.subs(x, 11 * L / 20)      #w'''(L) = -P
                        ]
print('\nboundary conditions:')
display(boundary_conditions)

integration_constants = solve(boundary_conditions, 'C1, C2, C3, C4', real=True)
print('\nintegration constants:')
display(integration_constants)

# displacement function with constants substituted
solution = w.subs(integration_constants)
display(simplify(solution))
w_func = lambdify((x, L, q0, P, E, I), solution, 'numpy')

# moment function with constants substituted
M_solution = M.subs(integration_constants)
M_func = lambdify((x, L, q0, P, E, I), M_solution, 'numpy')

#%%
