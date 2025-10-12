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
z_num = -h_num / 2
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
M_phi = D * (-w.diff(r, 1) - nu * w.diff(r, 2))

# shear force field
V = M_r.diff(r, 1) +  1 / r * (M_r - M_phi)

sigma_rr = E / (1 - nu**2) * (-z * w.diff(r, 2) - nu * z * w.diff(r, 1))
sigma_phiphi = E / (1 - nu**2) * (-nu * z * w.diff(r, 2) - z * w.diff(r, 1))

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

r_vals = np.linspace(a_radius, b_radius, 401)

w_vals = w_func(r_vals, q_num, b_radius, a_radius, E_num, nu_num, h_num)
sigma_rr_vals = sigma_rr_func(r_vals, q_num, b_radius, a_radius, E_num, nu_num, h_num, z_num)


plt.figure()
plt.plot(r_vals, w_vals)
plt.axvline(a_radius, color='black', linestyle='--', label='a')
plt.axvline(b_radius, color='grey', linestyle='--', label='b')
plt.title('Radial deflection')
plt.xlabel('r [m]')
plt.ylabel('w [m]')
fig('test')


plt.figure()
plt.plot(r_vals, sigma_rr_vals)
plt.axvline(a_radius, color='black', linestyle='--', label='a')
plt.axvline(b_radius, color='grey', linestyle='--', label='b')
plt.axhline(-sigma_y_num, color='orange', linestyle='--', label='yield strength')
plt.title('Radial stress')
plt.xlabel('r [m]')
plt.ylabel('sigma [Pa]')
fig('test')

#%%






















































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

# Define symbols
F_sym, q0_sym, a_sym, b_sym, E_sym, nu_sym, h_sym, r_sym, A1_sym, A2_sym, A3_sym, A4_sym = symbols('F q0 a b E nu h r A1 A2 A3 A4', real = True)

D_sym = 1/12/(1-nu_sym**2)*E_sym*(h_sym**3) # bending stiffness

q_sym = q0_sym*(r_sym-a_sym)/(b_sym-a_sym)          # distributed load

# Formulate general solutions
w = integrate(1 / r_sym * integrate(r_sym * integrate(1 / r_sym * integrate(q * r_sym / D_sym, r_sym), r_sym), r_sym), r_sym)+\
    A1_sym * r_sym**2 * log(r_sym / b_sym) + A2_sym * r_sym**2 + A3_sym * log(r_sym / b_sym) + A4_sym # deflection field

wprime = diff(w, r_sym)                          # rotation field

Mr = D_sym * (-diff(wprime, r_sym) - nu_sym / r_sym * wprime )   # radial bending moment field
Mphi = D_sym * (-1 / r_sym * wprime - nu_sym * diff(wprime, r_sym))  # circumferential bending moment field
V = diff(Mr, r_sym) + 1 / r_sym * (Mr - Mphi)           # shear force field

# Apply the boundary conditions
boundary_conditions = [
Mr.subs(r_sym, a_sym),     # inner boundary radial bending moment free
V.subs(r_sym, a_sym) - F_sym,    # inner boundary shear force applied
w.subs(r_sym, b_sym),      # outer boundary deflection fixed
wprime.subs(r_sym, b_sym), # outer boundary rotation fixed
]

# Solve for unknown constants
unknowns = (A1_sym, A2_sym, A3_sym, A4_sym)
integration_constants= solve(boundary_conditions, unknowns)
print('\nintegration constants:')
display(integration_constants)

# Formulate the deflection field
w_ = w.subs(integration_constants) # constants substituted
print("w_(r) = ")
display(w_)

# Plot the deflection field for a given set of parameters
w_func = lambdify((F_sym, q0_sym, E_sym, nu_sym, h_sym, r_sym), w_, 'numpy')

# w_eval = simplify(w_.subs({F_sym:1., q_sym:0., E:200e3, nu_sym:0.3, a_sym:100, b_sym:500, h_sym:4})) # parameters substituted

r_vals = np.linspace(a_radius, b_radius, 401)
w_vals = w_func(F, q, E, nu, h, r_vals)

plt.figure()
plt.plot(r_vals, w_vals, "b-")
plt.xlabel(r"$r$ [mm]")
plt.ylabel(r"$w$ [mm]")
fig('displacement')

#%%