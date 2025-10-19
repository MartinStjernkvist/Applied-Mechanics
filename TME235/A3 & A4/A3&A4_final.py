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
    
    plt.figure()
    plt.plot(X, u2, color='blue', label=results_file)
    plt.title('Radial deflection')
    plt.xlabel('r [m]')
    plt.ylabel('w [m]')
    fig(results_file.strip('.rpt') + 'radial deflection a')

    plt.figure()
    plt.plot(X, s11, color='red', label=results_file)
    plt.axhline(sigma_y_num, color='orange', linestyle='--', label='yield strength')
    plt.title('Radial stress')
    plt.xlabel('r [m]')
    plt.ylabel('sigma [Pa]')
    fig(results_file.strip('.rpt') + 'radial stress a')
    
    plt.figure()
    plt.plot(X, svm, color='green', label=results_file)
    plt.axhline(sigma_y_num, color='orange', linestyle='--', label='yield strength')
    plt.title('von Mises stress')
    plt.xlabel('r [m]')
    plt.ylabel('sigma [Pa]')
    fig(results_file.strip('.rpt') + 'von mises stress a')
    
    return datasets, X, u2, s11, svm

_, a1h1_X, a1h1_u2, a1h1_s11, a1h1_svm = read_abaqus_data('a1h1_results.rpt')
_, a1h2_X, a1h2_u2, a1h2_s11, a1h2_svm = read_abaqus_data('a1h2_results.rpt') 
_, a2h1_X, a2h1_u2, a2h1_s11, a2h1_svm = read_abaqus_data('a2h1_results.rpt') 
_, a1h_mid_X, a1h1_mid_u2, a1h1_mid_s11, a1h1_mid_svm = read_abaqus_data('a1h1_mid_results.rpt') # middle support
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



plt.figure()
plt.plot(a1h1_r_vals, a1h1_w_vals, color='brown', alpha=0.75, label=fr'normalized values, a: {1}, h: {1}')
plt.plot(a1h2_r_vals, a1h2_w_vals, color='brown', alpha=0.75, linestyle='--', label=fr'normalized values, a: {1}, h: {0.5}')
plt.plot(a2h1_r_vals, a2h1_w_vals, color='brown', alpha=0.75, linestyle=':', label=fr'normalized values, a: {0.5}, h: {1}')
plt.title('Radial deflection')
plt.xlabel('r [m]')
plt.ylabel('w [m]')
fig('comparison radial deflection numerical')

plt.figure()
plt.plot(a1h1_r_vals, a1h1_w_vals, color='brown', alpha=0.75, label=fr'normalized values, a: {1}, h: {1}')
plt.plot(a1h2_r_vals, a1h2_w_vals, color='brown', alpha=0.75, linestyle='--', label=fr'normalized values, a: {1}, h: {0.5}')
plt.plot(a2h1_r_vals, a2h1_w_vals, color='brown', alpha=0.75, linestyle=':', label=fr'normalized values, a: {0.5}, h: {1}')
plt.title('Radial stress')
plt.xlabel('r [m]')
plt.ylabel('sigma [Pa]')
fig('comparison radial stress numerical')

plt.figure()
plt.plot(a1h1_r_vals, a1h1_sigma_vm_vals, color='brown', alpha=0.75, label=fr'normalized values, a: {1}, h: {1}')
plt.plot(a1h2_r_vals, a1h2_sigma_vm_vals, color='brown', alpha=0.75, linestyle='--', label=fr'normalized values, a: {1}, h: {0.5}')
plt.plot(a2h1_r_vals, a2h1_sigma_vm_vals, color='brown', alpha=0.75, linestyle=':', label=fr'normalized values, a: {0.5}, h: {1}')
plt.title('von Mises stress')
plt.xlabel('r [m]')
plt.ylabel('sigma [Pa]')
fig('comparison von mises stress numerical')


plt.figure()
plt.plot(a1h1_r_vals, a1h1_w_vals, color='grey', label='numerical')
plt.plot(a1h1_X+0.1, a1h1_u2, color='magenta', label ='abaqus')
plt.title('Radial deflection')
plt.xlabel('r [m]')
plt.ylabel('w [m]')
fig('comparison radial deflection')

plt.figure()
plt.plot(a1h2_r_vals, a1h2_sigma_rr_vals, color='grey', label='numerical')
plt.plot(a1h2_X+0.1, a1h2_s11, color='magenta', label ='abaqus')
plt.title('Radial stress')
plt.xlabel('r [m]')
plt.ylabel('sigma [Pa]')
fig('comparison radial stress')

plt.figure()
plt.plot(a2h1_r_vals, a2h1_sigma_vm_vals, color='grey', label='numerical')
plt.plot(a2h1_X+0.05, a2h1_svm, color='magenta', label ='abaqus')
plt.title('von Mises stress')
plt.xlabel('r [m]')
plt.ylabel('sigma [Pa]')
fig('comparison von mises stress')

plt.figure()
plt.plot(a1h1_X+0.1, a1h1_u2, color='black', alpha=0.75, label ='abaqus, simple, bottom')
plt.plot(a1h1_X+0.1, a1h1_radialfix_u2, color='black',alpha=0.75, linestyle='--', label ='abaqus, pinned, bottom')
plt.plot(a1h1_X+0.1, a1h1_mid_u2, color='black',alpha=0.75, linestyle=':', label ='abaqus, simple middle')
plt.title('Radial deflection')
plt.xlabel('w [m]')
plt.ylabel('sigma [Pa]')
fig('comparison support deflection')

plt.figure()
plt.plot(a1h1_X+0.1, a1h1_s11, color='black',alpha=0.75, label ='abaqus, simple, bottom')
plt.plot(a1h1_X+0.1, a1h1_radialfix_s11, color='black',alpha=0.75, linestyle='--', label ='abaqus, pinned, bottom')
plt.plot(a1h1_X+0.1, a1h1_mid_s11, color='black',alpha=0.75, linestyle=':', label ='abaqus, simple middle')
plt.title('Radial stress')
plt.xlabel('r [m]')
plt.ylabel('sigma [Pa]')
fig('comparison support radial stress')

plt.figure()
plt.plot(a1h1_X+0.1, a1h1_svm, color='black',alpha=0.75, label ='abaqus, simple, bottom')
plt.plot(a1h1_X+0.1, a1h1_radialfix_svm, color='black',alpha=0.75, linestyle='--', label ='abaqus, pinned, bottom')
plt.plot(a1h1_X+0.1, a1h1_mid_svm, color='black',alpha=0.75, linestyle=':', label ='abaqus, simple middle')
plt.title('von Mises stress')
plt.xlabel('r [m]')
plt.ylabel('sigma [Pa]')
fig('comparison support von mises stress')


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
    plt.xlabel('x [m]')
    plt.ylabel('w [m]')
    fig(results_file.strip('.rpt') + 'deflection')
    
    return datasets, X, u2

_, X_10, u2_10 = read_abaqus_data_2('a4_p_10_results.rpt')
_, X_minus_10, u2_minus_10 = read_abaqus_data_2('a4_p_minus_10_results.rpt')


plt.figure()
plt.plot(X_10, u2_10, label='P = 10 kN')
plt.plot(X_minus_10, u2_minus_10, label='P = -10 kN')
plt.plot(X_minus_10, -u2_minus_10, label='P = -10 kN, negative sign')
plt.title('Deflection')
plt.xlabel('x [m]')
plt.ylabel('w [m]')
fig('difference deflection')

#%%

####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################



# Assignment 4 - numerical solution, roller support



####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################
new_prob('4 - numerical solution, roller support')

# Define symbols
x, Fr, P, L1, L2, E4, I4, C1, C2, C3, C4 = symbols('x Fr P L1 L2 V4 E4 C1 C2 C3 C4', real=True) 

# functions
M1 = - (P + Fr) * x + (Fr * L1 + P * L2)
M2 = - P * (L2 - x)

V1 = M1.diff(x, 1)
V2 = M2.diff(x, 1)

# valid for 0 =< x =< L1
w1 = 1 / (E4 * I4) * (integrate(integrate(M1, x), x)) + C1 * x + C2 

# valid for L1 < x =< L2
w2 = 1 / (E4 * I4) * (integrate(integrate(M2, x), x)) + C3 * x + C4

bc = [
    w1.subs(x, 0),
    w1.diff(x).subs(x, 0),
    w1.subs(x, L1),
    w1.subs(x, L1) - w2.subs(x, L1),
    w1.diff(x).subs(x, L1) - w2.diff(x).subs(x, L1)
]

unknowns = [Fr, C1, C2, C3, C4]
ic = solve(bc, unknowns)
print('integration constants:')
display(ic)
w1_ = w1.subs(ic)
w2_ = w2.subs(ic)

print('w1_:')
display(simplify(w1_))
print('w2_:')
display(simplify(w2_))

w1_func = lambdify((x, P, L1, L2, E4, I4), w1_, 'numpy')
w2_func = lambdify((x, P, L1, L2, E4, I4), w2_, 'numpy')

E4_num = E_steel
L2_num = 2
L1_num = 11/20 * L2_num
I4_num = b2 * h2 *3 / 12 #moment of inertia

P_num = 4_000_000

# validation:
x_num = L1_num
print(w1_func(x_num, P_num, L1_num, L2_num, E4_num, I4_num))
print(w2_func(x_num, P_num, L1_num, L2_num, E4_num, I4_num))

x1_vals = np.linspace(0, L1_num, 401)
x2_vals = np.linspace(L1_num, L2_num, 401)

w1_vals_pos = w1_func(x1_vals, P_num, L1_num, L2_num, E4_num, I4_num)
w2_vals_pos = w1_func(x2_vals, P_num, L1_num, L2_num, E4_num, I4_num)

plt.figure()
plt.plot(x1_vals, w1_vals_pos, label='numerical solution, roller support, w1')
plt.plot(x2_vals, w2_vals_pos, label='numerical solution, roller support, w2')
plt.title('Deflection')
plt.xlabel('x [m]')
plt.ylabel('w [m]')
fig('deflection numerical minus 10')

P_num = -4_000_000

x1_vals = np.linspace(0, L1_num, 401)
x2_vals = np.linspace(L1_num, L2_num, 401)

w1_vals = w1_func(x1_vals, P_num, L1_num, L2_num, E4_num, I4_num)
w2_vals = w1_func(x2_vals, P_num, L1_num, L2_num, E4_num, I4_num)

plt.figure()
plt.plot(x1_vals, w1_vals, label='numerical solution, roller support, w1')
plt.plot(x2_vals, w2_vals, label='numerical solution, roller support, w2')
plt.title('Deflection')
plt.xlabel('x [m]')
plt.ylabel('w [m]')
fig('deflection numerical 10')

#%%
####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################



# Assignment 4 - comparison plots



####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################
new_prob('4 - comparison plots')

x_vals = np.linspace(0, L2_num, 401)

def euler_bernoulli(x, P):
    return (P * x**3) / (3 * E4_num * I4_num)

P_num = 4_000_000

euler_bernoulli_10_vals = euler_bernoulli(x_vals, P_num)
euler_bernoulli_minus_10_vals = euler_bernoulli(x_vals, -P_num)

plt.figure()
plt.plot(x1_vals, w1_vals_pos, label='numerical solution, roller support, w1')
plt.plot(x2_vals, w2_vals_pos, label='numerical solution, roller support, w2')
plt.plot(X_10, u2_10, color='red', label='abaqus results')
plt.plot(x_vals, euler_bernoulli_10_vals, color='purple', label='numerical, no support')
plt.title('Deflection')
plt.xlabel('x [m]')
plt.ylabel('w [m]')
fig('deflection comparison 10')

plt.figure()
plt.plot(x1_vals, w1_vals, label='numerical solution, roller support, w1')
plt.plot(x2_vals, w2_vals, label='numerical solution, roller support, w2')
plt.plot(X_minus_10, u2_minus_10, color='red', label='abaqus results')
plt.plot(x_vals, euler_bernoulli_minus_10_vals, color='purple', label='numerical, no support')
plt.title('Deflection')
plt.xlabel('x [m]')
plt.ylabel('w [m]')
fig('deflection comparison minus 10')


#%%