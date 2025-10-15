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
    
numerical_analysis(a_radius, h_num)
numerical_analysis(a_radius/2, h_num)
numerical_analysis(a_radius, h_num/2)

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
    s11 = datasets['X s11_top_X'][:, 1]
    svm = datasets['X svm_top_X'][:, 1]
    
    plt.figure()
    plt.plot(X, u2, color='blue', label=results_file)
    plt.axvline(a, color='black', linestyle='--', label='a')
    plt.axvline(b_radius, color='grey', linestyle='--', label='b')
    plt.title('Radial deflection')
    plt.xlabel('r [m]')
    plt.ylabel('w [m]')
    fig(results_file.strip('.') + 'radial deflection a')

    plt.figure()
    plt.plot(X, s11, color='red', label=results_file)
    plt.axvline(a, color='black', linestyle='--', label='a')
    plt.axvline(b_radius, color='grey', linestyle='--', label='b')
    plt.axhline(sigma_y_num, color='orange', linestyle='--', label='yield strength')
    plt.title('Radial stress')
    plt.xlabel('r [m]')
    plt.ylabel('sigma [Pa]')
    fig(results_file.strip('.') + 'radial stress a')
    
    plt.figure()
    plt.plot(X, svm, color='green', label=results_file)
    plt.axvline(a, color='black', linestyle='--', label='a')
    plt.axvline(b_radius, color='grey', linestyle='--', label='b')
    plt.axhline(sigma_y_num, color='orange', linestyle='--', label='yield strength')
    plt.title('von Mises stress')
    plt.xlabel('r [m]')
    plt.ylabel('sigma [Pa]')
    fig(results_file.strip('.rpt') + 'von mises stress a')
    
    return datasets, X, u2, s11, svm

read_abaqus_data('a1h1_results.rpt') # left bottom support
read_abaqus_data('a1h2_results.rpt') # left bottom support
read_abaqus_data('a2h1_results.rpt') # left bottom support
read_abaqus_data('a1h1_left_mid_results.rpt') # left middle support
read_abaqus_data('a1h1_right_bottom_results.rpt') # right bottom support


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


# eqn (9.6)
# sigma = G * (lam**2 - 1)



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

'''
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

# Two formulations
sigma_eq96 = G_rubber * (lambda_vals**2 - 1)  # Eq. (9.6)
sigma_uniaxial = G_rubber * (lambda_vals**2 - 1/lambda_vals)  # Uniaxial stress

# Deviatoric stress (should be identical)
sigma22_eq96 = G_rubber * (1 / lambda_vals - 1)
mean_96 = (sigma_eq96 + 2 * sigma22_eq96) / 3
dev_96 = sigma_eq96 - mean_96
dev_uni = sigma_uniaxial - sigma_uniaxial / 3

plt.figure()
plt.plot(lambda_vals, sigma_eq96, 'b-', label='Eq. (9.6)')
plt.plot(lambda_vals, sigma_uniaxial, 'r--', label='Uniaxial')
plt.axhline(0, color='k', linestyle='--', alpha=0.3)
plt.axvline(1, color='k', linestyle='--', alpha=0.3)
plt.xlabel('Stretch λ')
plt.ylabel('Stress σ₁₁ (Pa)')
plt.title('Neo-Hookean: Two Formulations')
fig('Neo-Hookean: Two Formulations')

plt.figure()
plt.plot(lambda_vals, dev_96, 'b-', linewidth=2.5, label='Eq. (9.6) deviatoric')
plt.plot(lambda_vals, dev_uni, 'r--', label='Uniaxial deviatoric')
plt.axhline(0, color='k', linestyle='--', alpha=0.3)
plt.axvline(1, color='k', linestyle='--', alpha=0.3)
plt.xlabel('Stretch λ')
plt.ylabel('Deviatoric Stress (Pa)')
plt.title('Deviatoric Stress: Identical')
fig('Deviatoric Stress: Identical')

print("\nBoth give IDENTICAL deviatoric stress → same material behavior!")


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
fig('Beam Deflection: Comparison of Support Conditions')

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
#%%
