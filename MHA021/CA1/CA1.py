#%%
# Code written by Martin Stjernkvist

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

def sfig(fig_name):
    fig_output_file = script_dir / "figures" / fig_name
    fig_output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_output_file, dpi=dpi, bbox_inches='tight')
    print('figure name: ', fig_name)


def fig(fig_name):
    fig_output_file = script_dir / "figures" / fig_name
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

# Vertical deflection p
displayvar("p_{truss}", a[10-1]) 


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

q = 0

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
radius = np.sqrt(A / np.pi)
I = np.pi / 4 * radius**4
displayvar("I", I)

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
displayvar("p_P", a[14-1]) 

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

sigma_top = np.zeros((num_el, 2))
sigma_bottom = np.zeros((num_el, 2))

# Navier's formula
for i in range(7):
    for j in range(2):
        sigma_top[i, j] = M[i, j] * radius / I
        sigma_bottom[i, j] = M[i, j] * (-radius) / I
        
displayvar("stress top", sigma_top) 
displayvar("stress bottom", sigma_bottom) 


#---------------------------------------------------------------------------------------------------
# 2d)
#---------------------------------------------------------------------------------------------------
new_subtask('2d)')

q = P / L

# New distributed loads
num_el = Edof.shape[0]
qxy = np.zeros((num_el, 2)) # Distributed loads 
qxy[7-1, 1] = -q # Element 7, in vertical direction

# Same number of dofs

# Reset 
K = np.zeros((num_dofs, num_dofs))  # Stiffness matrix
f = np.zeros((num_dofs))            # Load vector

for el in range(num_el):
    dofs = Edof[el, :]   # DOFs for the element
    Ke, fe = beam2e(Ex[el, :], Ey[el, :], E, A, I, qxy[el, :])  # Element stiffness matrix
    assem(K, Ke, dofs)
    assem(f, fe, dofs)

# Same boundary conditions

# # Solve the system
a, r = solve_eq(K, f, bc_dofs, bc_vals)

displayvar("a", a, 2)
displayvar("r", np.round(r), 2)

# Vertical deflection p
displayvar("p_q", a[14-1]) 


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

n_elem = 1
iteration = 5

x_analytical = np.linspace(0, L, 100)
u_analytical = (q0 * L**2 / (6 * EA)) * (3 * x_analytical / L - (x_analytical / L)**3)
N_analytical = q0 * L / 2 * (1 - x_analytical**2 / L**2)
u_exact = u_analytical[-1]
N_exact = N_analytical[0]


def FEM_3(n_elem_start, iterations):
    u_error_list = []
    N_error_list = []
    n_elem_list = []
    
    for iteration in range(iterations + 1):
        print('\nIteraration: ', iteration)
        
        n_elem = n_elem_start * 2**iteration
        # Mesh generation
        n_nodes = n_elem + 1
        x_nodes = np.linspace(0, L, n_nodes)
        Le = L / n_elem  # element length

        # Initialize global stiffness matrix and load vector
        K_global = np.zeros((n_nodes, n_nodes))
        fl_global = np.zeros(n_nodes)

        # Assemble element contributions
        for e in range(n_elem):
            print(f'element {e}')
            # Element nodes
            i = e
            j = e + 1
            
            # Element stiffness matrix (linear bar element)
            K_e = (EA / Le) * np.array([[1, -1],
                                        [-1, 1]])
            
            # Element load vector using midpoint rule
            x_mid = (x_nodes[i] + x_nodes[j]) / 2
            fl_e = q0 * x_mid * Le / (2 * L) * np.array([1, 1])
            
            # Assemble into global system
            K_global[i : j + 1, i : j + 1] += K_e
            fl_global[i : j + 1] += fl_e
            
            # displayvar('K_{global}', K_global)
            # displayvar('fl_{global}', fl_global)

        # Apply boundary condition: u(0) = 0
        # Remove first DOF (fixed at node 0)
        K_reduced = K_global[1:, 1:]
        fl_reduced = fl_global[1:]
        
        # displayvar('K_{reduced}', K_reduced)
        # displayvar('fl_{reduced}', fl_reduced)

        # Solve system
        a_reduced = np.linalg.solve(K_reduced, fl_reduced)

        # Full displacement vector
        a = np.zeros(n_nodes)
        a[1:] = a_reduced

        # Results
        print(f"Number of elements: {n_elem}")
        print(f"Nodal coordinates: {x_nodes}")
        print(f"Displacements: {a}")
        print(f"Displacement at free end u(L): {a[-1]:.2e} m")
        
        
        N_list = []
        # Compute normal forces at element ends
        print("Normal forces:")
        for e in range(n_elem):
            i = e
            j = e + 1
            N = (EA / Le) * (a[j] - a[i])
            print(f"Element {e+1}: N = {N:.2f} N")
            N_list.append(N)

        # # Plot displacement
        # plt.figure()
        # plt.plot(x_nodes, a * 1e3, 'o-', linewidth=2, markersize=8)
        # plt.xlabel('x (m)')
        # plt.ylabel('u (mm)')
        # plt.title('Displacement along bar')
        # plt.grid()
        # plt.show()
        # sfig('Displacement along bar.png')
        
        u_FEM = a[-1]
        N_FEM = N_list[0]
        
        u_error = (u_exact - u_FEM) / u_exact
        N_error = (N_exact - N_FEM) / N_exact
        
        if u_error <= 0.02:
            print('u convergence at iteration: ', iteration)
        if N_error <= 0.02:
            print('N convergence at iteration: ', iteration)
        
        u_error_list.append(u_error)
        N_error_list.append(N_error)
        n_elem_list.append(n_elem)
        
    return u_error_list, N_error_list, n_elem_list

n_elem_start = 1
iterations = 2
u_error_list, N_error_list, n_elem_list = FEM_3(n_elem_start, iterations)

# Plot displacement error
plt.figure()
plt.plot(n_elem_list, u_error_list, 'o-')
plt.xlabel('number of elements')
plt.ylabel('u_error')
plt.title('Displacement error')
plt.grid()
sfig('Displacement error.png')
plt.show()     
   
# Plot normal force error
plt.figure()
plt.plot(n_elem_list, N_error_list, 'o-')
plt.xlabel('number of elements')
plt.ylabel('N_error')
plt.title('Normal force error')
plt.grid()
sfig('Normal force error.png')
plt.show()

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
