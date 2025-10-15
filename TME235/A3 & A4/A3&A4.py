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
# ASSIGNMENT 3 - Axisymmetric disc
# --------------------------------------------
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
    fig(results_file.strip('.') + 'von mises stress a')
    
    return datasets, X, u2, s11, svm

read_abaqus_data('a1h1_results.rpt') # left bottom support
read_abaqus_data('a1h2_results.rpt') # left bottom support
read_abaqus_data('a2h1_results.rpt') # left bottom support
read_abaqus_data('a1h1_left_mid_results.rpt') # left middle support
read_abaqus_data('a1h1_right_bottom_results.rpt') # right bottom support

#%%