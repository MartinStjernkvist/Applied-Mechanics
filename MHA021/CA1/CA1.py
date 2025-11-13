#%%
import sys
import os
# Add parent directory to sys.path
parent_dir = os.path.abspath(os.path.join(os.getcwd(),'..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
    
from mha021 import *

from colorama import Fore
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
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
from matplotlib import rcParams
import matplotlib.ticker as ticker

import calfem.core as cfc
import calfem.vis_mpl as cfv
import calfem.mesh as cfm
import calfem.utils as cfu

from scipy.sparse import coo_matrix, csr_matrix
import matplotlib.cm as cm

from pathlib import Path

def new_task(string):
    print_string = Fore.YELLOW + '\n' + '=' * 80 + '\n' + 'Task ' + str(string) + '\n' + '=' * 80 + '\n'
    return print(print_string)

def new_subtask(string):
    print_string = Fore.CYAN + '\n' + '-' * 80 + '\n' + 'Subtask ' + str(string) + '\n' + '-' * 80 + '\n'
    return print(print_string)

SMALL_SIZE = 10
MEDIUM_SIZE = 12
BIGGER_SIZE = 14
dpi = 500

plt.rc('font', size=SMALL_SIZE)          
plt.rc('axes', titlesize=BIGGER_SIZE)     
plt.rc('axes', labelsize=MEDIUM_SIZE)    
plt.rc('xtick', labelsize=SMALL_SIZE)    
plt.rc('ytick', labelsize=SMALL_SIZE)    
plt.rc('legend', fontsize=SMALL_SIZE)    
plt.rc('figure', titlesize=BIGGER_SIZE)

plt.rc('figure', figsize=(8,4))

script_dir = Path(__file__).parent

def fig(fig_name):
    fig_output_file = script_dir / "fig" / fig_name
    fig_output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.legend()
    plt.grid(True, alpha = 0.3)
    plt.savefig(fig_output_file, dpi=dpi, bbox_inches='tight')
    plt.show()
    print('figure name: ', fig_name)
    
def printt(**kwargs):
    for name, value in kwargs.items():
        print('\n')
        print(f"\033[94m{name}\033[0m:")
        print(f"\033[92m{value}\033[0m")
        print('\n')
    
#%%
####################################################################################################
####################################################################################################
####################################################################################################



# Task 1 - Planar truss



####################################################################################################
####################################################################################################
####################################################################################################

new_task('1 - Planar truss')

# Define input data
L = 1.6             # Element length [m]
P = 60e3            # Force [N]
E = 200e9           # Young's modulus [Pa]
A = 1e-3            # Cross-sectional area
sigma_y = 250e6     # Yielding [Pa]

#---------------------------------------------------------------------------------------------------
# 1a)
#---------------------------------------------------------------------------------------------------
new_subtask('1a)')

# See labeling of elements in the report

Ly = 3 * L / 4
Ex = np.array([
    [0, 0],
    [0, L],
    [0, L],
    [0, L],
    [L, L],
    [L, 2 * L],
    [L, 2 * L]
])
Ey = np.array([ 
    [0, Ly],
    [0, 0],
    [0, Ly],
    [Ly, Ly],
    [0, Ly],
    [0, Ly],
    [Ly, Ly]
])

# Plot the truss geometry
fig = draw_discrete_elements(
    Ex, Ey,
    title="Geometry with nodes and elements",
    xlabel="x [m]",
    ylabel="y [m]",
    annotate="both"
)
fig.show()

# Topology matrix (connectivity)
Edof = np.array([
    [1, 2, 3, 4],   # Element 1
    [1, 2, 5, 6],   # Element 2
    [1, 2, 7, 8],   # Element 3
    [3, 4, 7, 8],   # Element 4
    [5, 6, 7, 8],   # Element 5
    [5, 6, 9, 10],  # Element 6
    [7, 8, 9, 10]   # Element 7
])
# Number of elements
num_el = Edof.shape[0] # => (num_rows, num_columns) => select first one 
num_dofs = np.max(np.max(Edof))

print(f"number of dofs = {num_dofs}")
print(f"number of elements = {num_el}")

#---------------------------------------------------------------------------------------------------
# 1b)
#---------------------------------------------------------------------------------------------------
new_subtask('1b)')

# Assemble stiffness matrix and load vector, first allocate space
K = np.zeros((num_dofs, num_dofs))  # Stiffness matrix
f = np.zeros((num_dofs))            # Load vector

# Loop over all elements to assemble global stiffness matrix
for el in range(num_el):
    Ke = bar2e(Ex[el, :], Ey[el, :], E = E, A=A)  # Element stiffness matrix
    dofs = Edof[el, :]   # DOFs for the element
    assem(K, Ke, dofs)
displayvar("K", K)

# External forces
f[10-1] = -P  # Add a vertical force at node 5 (= dof 10)
displayvar("f", f)

# Boundary conditions
bc_dofs = np.array([1, 3, 4]) # DOFs fixed: 1, 3, 4
bc_vals = np.array([0.0, 0.0, 0.0])

# Solve the system of equations
a, r = solve_eq(K, f, bc_dofs, bc_vals)
displayvar("a", a)
displayvar("r", np.round(r))

#---------------------------------------------------------------------------------------------------
# 1c)
#---------------------------------------------------------------------------------------------------
new_subtask('1c)')

# Displacement for each element (Edof to extract dofs for each element)
Ed = extract_dofs(a, Edof)
fig = draw_discrete_elements(
    Ex, Ey,
    title="Deformed truss",
    xlabel="x [m]",
    ylabel="y [m]",
    line_style="dashed"
)
plot_deformed_bars(fig, Ex, Ey, Ed)
fig.show()

#---------------------------------------------------------------------------------------------------
# 1d)
#---------------------------------------------------------------------------------------------------
new_subtask('1d)')

# Compute normal forces
N = np.zeros((num_el))
for el in range(num_el):
    N[el] = bar2s(Ex[el, :], Ey[el, :], E, A, Ed[el, :]) 
displayvar("N", np.round(N))

# Normal stresses
sigma = N / A
displayvar("\sigma", np.round(sigma*1e-6)) # stresses in MPa

# Factor of safety
displayvar("FOS", sigma_y / np.max(np.abs(sigma))) # stresses in Pa

#%%
####################################################################################################
####################################################################################################
####################################################################################################



# Task 2 - Planar frame



####################################################################################################
####################################################################################################
####################################################################################################

new_task('2 - Planar frame')

# Define input data
L = 1.6             # Element length [m]
P = 60e3            # Force [N]
E = 200e9           # Young's modulus [Pa]
A = 1e-3            # Cross-sectional area
sigma_y = 250e6     # Yielding [Pa]

#---------------------------------------------------------------------------------------------------
# 2a)
#---------------------------------------------------------------------------------------------------
new_subtask('2a)')

# See labeling of elements in the report

Ly = 3 * L / 4
Ex = np.array([
    [0, 0],
    [0, L],
    [0, L],
    [0, L],
    [L, L],
    [L, 2 * L],
    [L, 2 * L]
])
Ey = np.array([ 
    [0, Ly],
    [0, 0],
    [0, Ly],
    [Ly, Ly],
    [0, Ly],
    [0, Ly],
    [Ly, Ly]
])

# Plot the truss geometry
fig = draw_discrete_elements(
    Ex, Ey,
    title="Geometry with nodes and elements",
    xlabel="x [m]",
    ylabel="y [m]",
    annotate="both"
)
fig.show()

# Radius and area moment of inertia for cross-sectional area
r = np.sqrt(A / np.pi)
I = np.pi / 4 * r**4
displayvar("I", I)

q = 0

# Topology matrix (connectivity)
Edof = np.array([
    [1, 2, 3, 4, 5, 6],         # Element 1
    [1, 2, 3, 7, 8, 9],         # Element 2
    [1, 2, 3, 10, 11, 12],      # Element 3
    [4, 5, 6, 10, 11, 12],      # Element 4
    [7, 8, 9, 10, 11, 12],      # Element 5
    [7, 8, 9, 13, 14, 15],      # Element 6
    [10, 11, 12, 13, 14, 15]    # Element 7
])
# Number of elements
num_el = Edof.shape[0] # => (num_rows, num_columns) => select first one 
num_dofs = np.max(np.max(Edof))

num_el = Edof.shape[0]
qxy = np.zeros((num_el, 2)) # Distributed loads 
qxy[:, 1] = -q # All elements in vertical direction

ep = [E, A, I]
num_dofs = np.max(np.max(Edof))

#---------------------------------------------------------------------------------------------------
# 2b)
#---------------------------------------------------------------------------------------------------
new_subtask('2b)')

K = np.zeros((num_dofs, num_dofs))  # Stiffness matrix
f = np.zeros((num_dofs))            # Load vector

for el in range(num_el):
    dofs = Edof[el, :]   # DOFs for the element
    Ke, fe = beam2e(Ex[el, :], Ey[el, :], E, A, I, qxy[el, :])  # Element stiffness matrix
    assem(K, Ke, dofs)
    assem(f, fe, dofs)

# External forces
f[14-1] = -P  # Add a vertical force at node 5 (= dof 14)
displayvar("f", f)

# Boundary conditions
# From Task 1: DOFs fixed: 1, 3, 4 
# --> 1, 4, 5 are fixed (new numbering)'
bc_dofs = [1, 4, 5]
bc_vals = [0.0, 0.0, 0.0]

# # Solve the system
a, r = solve_eq(K, f, bc_dofs, bc_vals)

displayvar("a", a, 2)
displayvar("r", np.round(r), 2)

# Vertical deflection p
displayvar("p", a[14-1]) 

#---------------------------------------------------------------------------------------------------
# 2c)
#---------------------------------------------------------------------------------------------------
new_subtask('2c)')

## Postprocessing
Ed = extract_dofs(a, Edof)

# Draw the deformed structure
fig = draw_discrete_elements(Ex, Ey, title="Deformed frame", line_style="dashed")
plot_deformed_beams(fig, Ex, Ey, Ed)
fig.show()

M = np.zeros((num_el, 2))
N = np.zeros((num_el, 2))
for el in range(num_el):
    data = beam2s(Ex[el, :], Ey[el, :], Ed[el, :], E, A, I)
    M[el, :] = data["M"]
    N[el, :] = data["N"]

displayvar("M", M) 
displayvar("N", N) 


#%%
####################################################################################################
####################################################################################################
####################################################################################################



# Task 3 - 1D bar with distributed load



####################################################################################################
####################################################################################################
####################################################################################################
new_task('3 - 1D bar with distributed load')

# Define input data
EA = 210e5      # Axial stiffness [N]
L = 2           # Element length [m]
q0 = 1000       # Distributed load at x = L [N / m]

def q(x):
    return q0 * x / L

#---------------------------------------------------------------------------------------------------
# 3e)
#---------------------------------------------------------------------------------------------------
new_subtask('e)')


#---------------------------------------------------------------------------------------------------
# 3g)
#---------------------------------------------------------------------------------------------------
new_subtask('g)')

def u(x):
    return q0 * L**2 / (6 * EA) * (3 * x / L - x**3 / L**3)

def N(x):
    return q0 * L / 2 * (1 - x**2 / L**2)

def e_u(u_exact, u_FEM):
    return (u_exact - u_FEM) / u_exact



#%%
####################################################################################################
####################################################################################################
####################################################################################################



# Task 4 - Beam on elastic foundation



####################################################################################################
####################################################################################################
####################################################################################################
new_task('4 - Beam on elastic foundation')

# Define input data
I_y = 30.3 * 10**(-6)   # [m^4]
E = 210 * 10**9         # [Pa]
K_w = 10.05 * 10**(6)   # [N / m^2]
L = 10                  # [m]
P = 70 * 10**3          # [N]

#%%
