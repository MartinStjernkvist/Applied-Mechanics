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

from pathlib import Path

def new_task(string):
    print_string = '\n' + '=' * 80 + '\n' + '=' * 80 + '\n' + 'Task ' + str(string) + '\n' + '=' * 80 + '\n' + '=' * 80 + '\n'
    return print(print_string)

def new_subtask(string):
    print_string = '\n' + '-' * 80 + '\n' + 'Subtask ' + str(string) + '\n' + '-' * 80 + '\n'
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
I = np.pi * radius**4 / 4 
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
        sigma_top[i, j] = M[i, j] * radius / I + N[i, j] / A
        sigma_bottom[i, j] = M[i, j] * (-radius) / I + N[i, j] / A
        
displayvar("stress top", sigma_top) 
displayvar("stress bottom", sigma_bottom) 

print('max tensile stress: ', max(np.max(sigma_top), np.max(sigma_bottom)))
print('max compressive stress: ', min(np.min(sigma_top), np.min(sigma_bottom)))

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

# Solve the system
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

        # Remove first DOF (fixed at node 0)
        K_reduced = K_global[1:, 1:]
        fl_reduced = fl_global[1:]
        
        # displayvar('K_{reduced}', K_reduced)
        # displayvar('fl_{reduced}', fl_reduced)
        
        # Solve the system
        a_reduced = np.linalg.solve(K_reduced, fl_reduced)

        # Displacement vector
        a = np.zeros(n_nodes)
        a[1:] = a_reduced

        print(f"Number of elements: {n_elem}")
        print(f"Nodal coordinates: {x_nodes}")
        print(f"Displacements: {a}")
        print(f"Displacement at free end u(L): {a[-1]:.2e} m")
        
        N_list = []
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
iterations = 5
u_error_list, N_error_list, n_elem_list = FEM_3(n_elem_start, iterations)

u_error_list_percent = [i  * 100 for i in u_error_list]
N_error_list_percent = [i * 100 for i in N_error_list]

# Plot displacement error
plt.figure()
plt.plot(n_elem_list, u_error_list_percent, 'o-')
plt.xlabel('number of elements')
plt.ylabel('u_error (%)')
plt.title('Displacement error')
plt.grid()
sfig('Displacement error.png')
plt.show()     
   
# Plot normal force error
plt.figure()
plt.plot(n_elem_list, N_error_list_percent, 'o-')
plt.xlabel('number of elements')
plt.ylabel('N_error (%)')
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

EI = E * I_y

def hermite_beam_stiffness(EI, Le):
    K_e = (EI / Le**3) * np.array([
        [12,      6*Le,    -12,     6*Le],
        [6*Le,     4*Le**2,  -6*Le,    2*Le**2],
        [-12,    -6*Le,     12,     -6*Le],
        [6*Le,     2*Le**2,  -6*Le,    4*Le**2]
    ])
    return K_e

def winkler_stiffness(K_w, Le):
    K_w_e = (K_w * Le / 420) * np.array([
        [156,     22*Le,    54,      -13*Le],
        [22*Le,    4*Le**2,  13*Le,    -3*Le**2],
        [54,      13*Le,    156,     -22*Le],
        [-13*Le,  -3*Le**2, -22*Le,    4*Le**2]
    ])
    return K_w_e

def solve_rail(n_elem, k_w, bc_type='semi-infinite'):
    
    n_nodes = n_elem + 1
    n_dof = 2 * n_nodes
    Le = L / n_elem
    
    K_global = np.zeros((n_dof, n_dof))
    f_global = np.zeros(n_dof)
    
    for e in range(n_elem):
        i = e
        j = e + 1
        
        K_e = hermite_beam_stiffness(EI, Le)
        K_w_e = winkler_stiffness(k_w, Le)
        K_total_e = K_e + K_w_e
        
        # DOFs: [w_i, th_i, w_j, th_j]
        dofs = [2 * i, 2 * i + 1, 2 * j, 2 * j + 1] 
        for ii in range(4):
            for jj in range(4):
                K_global[dofs[ii], dofs[jj]] += K_total_e[ii, jj]
        
    # Load at left end (x=0)
        f_global[0] = -P  # negative for downward
        
    # Apply boundary conditions
    fixed_dofs = []
    
    if bc_type == 'semi-infinite':
        fixed_dofs = []
        
    if bc_type == 'cantilever':
        # Fixed at x=0: w=0, theta=0
        fixed_dofs = [2 * (n_nodes - 1), 2 * (n_nodes - 1) + 1]
        
    # Remove fixed DOFs
    free_dofs = [i for i in range(n_dof) if i not in fixed_dofs]
    K_reduced = K_global[np.ix_(free_dofs, free_dofs)]
    f_reduced = f_global[free_dofs]
    
    # Solve the system
    a_reduced = np.linalg.solve(K_reduced, f_reduced)
    
    # Displacement vector
    a = np.zeros(n_dof)
    a[free_dofs] = a_reduced
    
    x = np.linspace(0, L, n_nodes)
    w = a[0::2]
    
    return x, w, a 

def compute_bending_moments(n_elem, a):
    Le = L / n_elem
    M = np.zeros((n_elem, 2))  # moment at start and end of each element
    
    for e in range(n_elem):
        i = e
        j = e + 1
        
        # Element DOFs
        a_e = np.array([a[2 * i], a[2 * i + 1], a[2 * j], a[2 * j + 1]])
        
        # Element stiffness
        K_e = hermite_beam_stiffness(EI, Le)
        
        # Element forces
        f_e = K_e @ a_e
        
        # Bending moments
        M[e, 0] = -f_e[1]  # node i
        M[e, 1] = f_e[3]   # node j
    return M

#---------------------------------------------------------------------------------------------------
# 4c)
#---------------------------------------------------------------------------------------------------
new_subtask('c)')

# Cantilever beam (point load P at tip)
w_analytical = -P * L**3 / (3 * EI)
print(f"Analytical max deflection: {w_analytical:.2e} m")

# Verification
n_elem_verify = 10
x_ver, w_ver, _ = solve_rail(n_elem=n_elem_verify, k_w=0, bc_type='cantilever')

print(f"FEM max deflection: {min(w_ver):.2e} m") # Min because w is negative downward
print(f"Relative error: {abs(min(w_ver) - w_analytical)/abs(w_analytical) * 100:.2f}%")

#---------------------------------------------------------------------------------------------------
# 4d)
#---------------------------------------------------------------------------------------------------
new_subtask('d)')

elem_counts = [2, 4, 8, 16, 32]
max_deflections = []

for n in elem_counts:
    x, w, a = solve_rail(n_elem=n, k_w=K_w)
    max_def = abs(min(w))
    max_deflections.append(max_def)
    print(f"Elements: {n}, Max deflection: {max_def:.2e} m")

# Check convergence
for i in range(1, len(elem_counts)):
    rel_change = abs(max_deflections[i] - max_deflections[i-1]) / max_deflections[i] * 100
    print(f"{elem_counts[i - 1]} -> {elem_counts[i]} elements: {rel_change:.3f}% change")

# Converged mesh
chosen_n_elem = 32
print(f"Chosen: {chosen_n_elem} elements (converged to <0.1%)")

#---------------------------------------------------------------------------------------------------
# 4e)
#---------------------------------------------------------------------------------------------------
new_subtask('e)')

x_final, w_final, a_final = solve_rail(n_elem=chosen_n_elem, k_w=K_w)

M_final = compute_bending_moments(chosen_n_elem, a_final)

print(f"Maximum deflection: {abs(min(w_final)):.2e} m")
print(f"Maximum bending moment: {np.max(np.abs(M_final)):.2f} Nm")

plt.figure()
plt.plot(x_final, w_final, 'b-o')
plt.xlabel('Position x [m]')
plt.ylabel('Deflection w [m]')
plt.title(f'Beam Deflection ({chosen_n_elem} elements)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
sfig('Beam Deflection.png')
plt.show()

x_M = []
M_plot = []
for e in range(chosen_n_elem):
    Le = L / chosen_n_elem
    x_M.extend([e*Le, (e+1)*Le])
    M_plot.extend([M_final[e, 0], M_final[e, 1]])
    
plt.figure()
plt.plot(x_M, M_plot, 'r-o')
plt.xlabel('Position x [m]')
plt.ylabel('Bending Moment [Nm]')
plt.title(f'Bending Moment Distribution ({chosen_n_elem} elements)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
sfig('Bending Moment Distribution.png')
plt.show()
#%%
