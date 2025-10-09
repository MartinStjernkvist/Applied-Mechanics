#%%
# %matplotlib widget
from quadmesh import *

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
# ASSIGNMENT 1 - Beam
# --------------------------------------------

L1 = 3
L2 = 0.3

E_num = 220 * 10**9
b_num = 0.05
h_num = b_num
I_num= b_num * h_num**3 / 12 #moment of inertia
rho_num = 7800 #density
g_num = 9.81
m_num = 130
P_num = -m_num * g_num
poisson_num = 0.3
q0_num = -h_num * b_num * rho_num * g_num

Ks_num = 5/6
A_num = b_num * h_num
G_num = E_num / (2 * (1 + poisson_num)) 

# --------------------------------------------
# ASSIGNMENT 2 - Axissymetry
# --------------------------------------------

a_num = 150 # mm
b_num = 250 # mm
h0_num = 20 # mm
p_num = 120 * 10**6 # Pa
E2_num = 200 * 10**9 # Pa

# for calfem calculations:
thickness = h0_num
Emod = E2_num
# lower left corner
p1 = [a_num, 0]
# upper right corner
p2 = [b_num, h0_num]


#%% 

####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################



# Assigment 1 - EULER-BERNOULLI


####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################
new_prob('1 - EULER-BERNOULLI')

x, L, q0, P, E, I = symbols('x L q0 P E I')

w = Function('w')(x) # w is a function of x

diffeq1 = Eq(E*I * diff(w, x, 4), q0)

w = dsolve(diffeq1, w).rhs

M = -E*I*w.diff(x, 2)

#C1, C2, C3, C4 = symbols('C1 C2 C3 C4')

# Boundary conditions for distributed load
boundary_conditions = [ 
                        w.subs(x, 0),                               #w(0) = 0
                        w.diff(x).subs(x, 0),                       #w'(0) = 0
                        M.subs(x, L),                               #w''(L) = 0
                        w.diff(x, 3).subs(x, L) - P / (-E * I)      #w'''(L) = -P
                        ]
print('\nboundary conditions:')
display(boundary_conditions)

integration_constants = solve(boundary_conditions, 'C1, C2, C3, C4', real=True)
print('\nintegration constants:')
display(integration_constants)

solution = w.subs(integration_constants)
display(simplify(solution))

w_func = lambdify((x, L, q0, P, E, I), solution, 'numpy')

L=L1
x_vals = np.linspace(0, L, 200)

x_vals = np.linspace(0, L, 200)
w_vals = w_func(x_vals, L, q0_num, P_num, E_num, I_num)

plt.figure()
plt.plot(x_vals, w_vals*1e3, 'b-')
plt.axhline(0, color='black', linestyle='--')
plt.title('Beam deflection, combined Loading L=3')
plt.xlabel('x (m)')
plt.ylabel('w (mm)')
plt.grid(True)
plt.ylim(bottom=min(w_vals)*1e3*1.1)
plt.xlim(0, L)
plt.show()

L=L2
x_vals = np.linspace(0, L, 200)
q0 = -h_num * b_num * rho_num * g_num
x_vals = np.linspace(0, L, 200)
w_vals = w_func(x_vals, L, q0, P_num, E_num, I_num)

plt.figure()
plt.plot(x_vals, w_vals*1e3, 'b-')
plt.axhline(0, color='black', linestyle='--')
plt.title('Beam deflection, combined Loading (L=0.3)')
plt.xlabel('x (m)')
plt.ylabel('w (mm)')
plt.grid(True)
plt.ylim(bottom=min(w_vals)*1e3*1.1)
plt.xlim(0, L)
plt.show()

# %%
####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################



# Assigment 1 - TIMOSHENKO



####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################
new_prob('1 - TIMOSHENKO')

#############
# Point load

x, q0, E, I, Ks, G, A, L, P = symbols('x q0 E I Ks G A L P', real=True)

f_phi = Function('phi') # phi is a function of x

## Define the differential equation in terms of phi
diffeq_phi = Eq(E * I * f_phi(x).diff(x, 3), 0)

## Solve the differential equation for phi(x) (eq. 3.35 LN)
phi = dsolve(diffeq_phi, f_phi(x)).rhs

## Solve the differential equation for w(x) (eq. 3.36 LN)
w = Function('w') # w is a function of x
diffeq_w = Eq(w(x).diff(x), -E * I / (G * Ks * A)*phi.diff(x,2) + phi)
w        = dsolve(diffeq_w, w(x)).rhs

## Define boundary conditions
M = -E*I*phi.diff(x)
bc_eqs = [
            Eq(w.subs(x, 0), 0),                        # w(0) = 0
            Eq(diff(w, x).subs(x, 0), 0),               # w'(0) = 0
            Eq(diff(w, x, 2).subs(x, L), 0),            # w''(L) = 0
            Eq(diff(w, x, 3).subs(x, L), -P/(E*I))      # w'''(L) = -P/(EI)
]

## Solve for the integration constants
integration_constants = solve(bc_eqs, 'C1, C2, C3, C4', real=True)

## Substitute the integration constants into the solution
solution1 = w.subs(integration_constants)
display(solution1)

##################
# Distributed load

## Define symbolic variables
x, q0, E, I, Ks, G, A, L = symbols('x q0 E I Ks G A L', real=True)

f_phi = Function('phi') # phi is a function of x

## Define the differential equation in terms of phi
diffeq_phi = Eq(E*I*f_phi(x).diff(x, 3), q0)

## Solve the differential equation for phi(x) (eq. 3.35 LN)
phi = dsolve(diffeq_phi, f_phi(x)).rhs

## Solve the differential equation for w(x) (eq. 3.36 LN)
w = Function('w') # w is a function of x
diffeq_w = Eq(w(x).diff(x), -E*I/(G*Ks*A)*phi.diff(x,2) + phi)
w        = dsolve(diffeq_w, w(x)).rhs

## Define boundary conditions
M = -E * I * phi.diff(x)
boundary_conditions1 = [ w.subs(x, 0), 0,               #w(0) = 0
                        w.diff(x).subs(x, 0),           #w'(0) = 0
                        M.subs(x, L), 0,                #w''(L) = 0
                        w.diff(x,3).subs(x, L), 0]      #w'''(L) = 0

## Solve for the integration constants
integration_constants = solve(boundary_conditions1, 'C1, C2, C3, C4', real=True)

## Substitute the integration constants into the solution
solution2 = w.subs(integration_constants)
display(solution2)

solution_total = solution1 + solution2
display(simplify(solution_total))

w_func = lambdify((x, L, q0, P, E, I, Ks, A, G), solution_total, 'numpy')

## Plugging in values for the length of the beam and plotting

L = L1
q0_num = -((m_num * g_num)/L)    

x_vals = np.linspace(0, L, 200)
w_vals = w_func(x_vals, L, q0_num, P_num, E_num, I_num, A_num, G_num, Ks_num)

plt.figure()
plt.plot(x_vals, w_vals * 1e3, 'b-')
plt.title('Beam Deflection, combined loading (L=3)')
plt.xlabel('x (m)')
plt.ylabel('w (mm)')
plt.grid(True)
plt.axhline(0, color='black', linestyle='--')
plt.ylim(bottom=min(w_vals) * 1e3 * 1.1)
plt.xlim(0, L)
plt.savefig('TIMOSHENKO_1', dpi=dpi, bbox_inches='tight')
plt.show()

L=L2
q0_num = -((m_num * g_num)/L2)

x_vals = np.linspace(0, L, 200)
w_vals = w_func(x_vals, L, q0_num, P_num, E_num, I_num, A_num, G_num, Ks_num)

plt.figure()
plt.plot(x_vals, w_vals*1e3, 'b-', linewidth=2)
plt.title('Beam deflection, combined loading (L=0.3)')
plt.xlabel('x (m)')
plt.ylabel('w (mm)')
plt.grid(True)
plt.axhline(0, color='black', linestyle='--')
plt.ylim(bottom=min(w_vals) * 1e3 * 1.1)
plt.xlim(0, L)
plt.savefig('TIMOSHENKO_2', dpi=dpi, bbox_inches='tight')
plt.show()


#%%
####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################



# Assignment 1 - calfem



####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################
new_prob('1 - calfem')

# Geometry
L1 = 3.0
L2 = 0.3
h = b = 0.05

# Material data
E = 220e9
poisson = 0.3
ptype = 1  # plane stress
ep = [ptype, b]  # [ptype, thickness]
Dmat = cfc.hooke(ptype, E, poisson)

# Physical properties
rho = 7800  # density
g = 9.81
m = 130
A = b * h

# Forces
P = -m * g  # Point load (downward, negative y-direction)
q0 = -h * b * rho * g  # Distributed load per unit length (downward)

# Create mesh
p1 = np.array([0., 0.])  # Lower left corner
p2 = np.array([L1, h])   # Upper right corner
nelx = 20
nely = 4
ndof_per_node = 2
nnode = (nelx + 1) * (nely + 1)
nDofs = ndof_per_node * nnode

# Create and plot rectangular mesh using quadmesh.py 

height = h
thickness = b

p1 = np.array([0., 0.]) # Lower left corner
p2 = np.array([L1, height]) # Upper right corner

nelx = 20; nely = 4 # Number of elements in x and y direction

ndof_per_node = 2 # Number of dofs per node
nnode = (nelx+1) * (nely+1) # Number of nodes
nDofs = ndof_per_node * nnode # Number of degrees of freedom

eq = [0, -q0]


# Generating mesh

Ex, Ey, Edof, B1, B2, B3, B4, P1, P2, P3, P4 = quadmesh(p1, p2, nelx, nely, ndof_per_node)
cfv.eldraw2(Ex, Ey) # Plotting the mesh

#display(Edof)
#display("Ex", Ex)

## Initializing boundary condition vectors

bc = np.array([], 'i')
bc_val = np.array([], 'f')
bc = B4
bcval = 0. * np.ones(np.size(bc))

display(bc)

# Initializing empty stiffness matrix
K = np.zeros([nDofs, nDofs])
f = np.zeros((nDofs, 1))

for eltopo, elx, ely, in zip(Edof, Ex, Ey):
    Ke = cfc.planqe(elx, ely, ep, Dmat)
    cfc.assem(eltopo, K, Ke)

top_elements = range(nelx * (nely - 1), nelx * nely)

for el_idx in top_elements:
    # Get element coordinates
    elx = Ex[el_idx, :]
    ely = Ey[el_idx, :]
    
    # Get element topology
    eltopo = Edof[el_idx, :]
    
    # For a quad element, the top edge connects nodes at indices 2 and 3
    # Calculate edge length
    edge_length = np.sqrt((elx[2] - elx[3])**2 + (ely[2] - ely[3])**2)
    
    # Equivalent nodal forces for uniform distributed load
    # For a uniformly distributed load q over length L:
    # Each node gets q*L/2
    nodal_force = q0 * edge_length / 2.0
    
    # Apply to the y-DOFs of nodes 3 and 4 (indices 4,5,6,7 in eltopo)
    # Node 3 (top-right): eltopo[4] = x-dof, eltopo[5] = y-dof
    # Node 4 (top-left): eltopo[6] = x-dof, eltopo[7] = y-dof
    f[eltopo[5] - 1] += nodal_force  # y-DOF of node 3 (0-indexed)
    f[eltopo[7] - 1] += nodal_force  # y-DOF of node 4 (0-indexed)

# Apply point load P at top right corner (P3)
# P3 contains the DOFs of the top-right corner node
f[P3[1] - 1] += P  # y-direction DOF (subtract 1 for 0-indexing)

print(f"Total distributed load: {q0 * L1:.2f} N")
print(f"Point load: {P:.2f} N")
print(f"Total load: {q0 * L1 + P:.2f} N")

# Apply boundary conditions - fixed left edge (B4)
bc = B4
bcval = np.zeros(np.size(bc))

# Solve the system
a, r = cfc.solveq(K, f, bc, bcval)

# Extract displacements
print(f"\nMaximum displacement: {np.min(a):.6e} m")
print(f"Maximum displacement (mm): {np.min(a)*1000:.4f} mm")



# Geometry
L1 = 3.0
L2 = 0.3
h = b = 0.05

# Material data
E = 220e9
poisson = 0.3
ptype = 1  # plane stress
ep = [ptype, b]  # [ptype, thickness]
Dmat = cfc.hooke(ptype, E, poisson)

# Physical properties
rho = 7800  # density
g = 9.81
m = 130
A = b * h

# Forces
P = -m * g  # Point load (downward, negative y-direction)
q0 = -h * b * rho * g  # Distributed load per unit length (downward)

# Create mesh
p1 = np.array([0., 0.])  # Lower left corner
p2 = np.array([L1, h])   # Upper right corner
nelx = 1000
nely = 10
ndof_per_node = 2
nnode = (nelx + 1) * (nely + 1)
nDofs = ndof_per_node * nnode

# Generate mesh
Ex, Ey, Edof, B1, B2, B3, B4, P1, P2, P3, P4 = quadmesh(p1, p2, nelx, nely, ndof_per_node)

# Plot mesh
cfv.figure()
cfv.eldraw2(Ex, Ey)
cfv.title('Mesh')
cfv.show()

display(B2)



# Initialize global stiffness matrix and force vector
K = np.zeros([nDofs, nDofs])
f = np.zeros([nDofs, 1])

# Assemble stiffness matrix
for eltopo, elx, ely in zip(Edof, Ex, Ey):
    Ke = cfc.planqe(elx, ely, ep, Dmat)
    cfc.assem(eltopo, K, Ke)

# Apply distributed load on top edge (B3)
# For a uniformly distributed load on a line element, the equivalent nodal forces
# are q*L/2 at each node (where L is the length of the edge)

# Find which elements have their top edge on B3
# Top edge elements are those in the top row (last nely row)
top_elements = range(nelx * (nely - 1), nelx * nely)


for el_idx in top_elements:
    # Get element coordinates
    elx = Ex[el_idx, :]
    ely = Ey[el_idx, :]
    
    # Get element topology
    eltopo = Edof[el_idx, :]
    
    # For a quad element, the top edge connects nodes at indices 2 and 3
    # Calculate edge length
    edge_length = np.sqrt((elx[2] - elx[3])**2 + (ely[2] - ely[3])**2)
    
    # Equivalent nodal forces for uniform distributed load
    # For a uniformly distributed load q over length L:
    # Each node gets q*L/2
    nodal_force = q0 * edge_length / 2.0
    
    # Apply to the y-DOFs of nodes 3 and 4 (indices 4,5,6,7 in eltopo)
    # Node 3 (top-right): eltopo[4] = x-dof, eltopo[5] = y-dof
    # Node 4 (top-left): eltopo[6] = x-dof, eltopo[7] = y-dof
    f[eltopo[5] - 1] += nodal_force  # y-DOF of node 3 (0-indexed)
    f[eltopo[7] - 1] += nodal_force  # y-DOF of node 4 (0-indexed)

# Apply point load P at top right corner (P3)
# P3 contains the DOFs of the top-right corner node
#f[P3[1] - 1] += P  # y-direction DOF (subtract 1 for 0-indexing)
f[B2[0]] += P


print(f"Total distributed load: {q0 * L1:.2f} N")
print(f"Point load: {P:.2f} N")
print(f"Total load: {q0 * L1 + P:.2f} N")

# Apply boundary conditions - fixed left edge (B4)
bc = B4
bcval = np.zeros(np.size(bc))

# Solve the system
a, r = cfc.solveq(K, f, bc, bcval)

# Extract displacements
print(f"\nMaximum displacement: {np.min(a):.6e} m")
print(f"Maximum displacement (mm): {np.min(a)*1000:.4f} mm")

Ed = cfc.extract_eldisp(Edof, a)

# Deformations
plt.figure()
plotpar = [2, 1, 0]  # Plotting parameters
cfv.eldraw2(Ex, Ey, plotpar)  # Drawing the original geometry
plt.title('Original geometry')
# Drawing the deformed structure
#sfac = cfv.scalfact2(Ex[2, :], Ey[2, :], Ed[2, :], 1)
#plotpar = [1, 2, 1]
#sfac = 40  # Scaling factor to see the actual deformations
#cfv.eldisp2(Ex, Ey, Ed, plotpar, sfac)
#plt.title('Displacement')

# -------------------------------
# Plot undeformed and deformed mesh manually
# -------------------------------

# Compute nodal coordinates
xv = np.linspace(p1[0], p2[0], nelx + 1)
yv = np.linspace(p1[1], p2[1], nely + 1)
coords = np.array([[x, y] for y in yv for x in xv])  # (N, 2) array

# Deformed coordinates
U = a[0::2].reshape(-1, 1)  # x-displacements
V = a[1::2].reshape(-1, 1)  # y-displacements

sfac = 1  # deformation scale
coords_def = coords + sfac * np.hstack([U, V])

# Plot
plt.figure()
for elx, ely in zip(Ex, Ey):
    # Undeformed
    plt.plot(np.append(elx, elx[0]), np.append(ely, ely[0]), 'k-', lw=0.8, alpha=0.5)

for e, (elx, ely) in enumerate(zip(Ex, Ey)):
    # Get the deformed coordinates for each element
    nodes = ((Edof[e, ::ndof_per_node] - 1) // ndof_per_node).astype(int)
    x_def = coords_def[nodes, 0]
    y_def = coords_def[nodes, 1]
    plt.plot(np.append(x_def, x_def[0]), np.append(y_def, y_def[0]), 'r-')

plt.gca().set_aspect('equal')
plt.title(f"Deformed mesh (scale factor = {sfac:.1f})")
plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.tight_layout()
plt.savefig(f"deformed_mesh_{L1 if p2[0]==L1 else L2}m.png", dpi=300, bbox_inches='tight')
plt.show()



# Geometry
L1 = 3.0
L2 = 0.3
h = b = 0.05

# Material data
E = 220e9
poisson = 0.3
ptype = 1  # plane stress
ep = [ptype, b]  # [ptype, thickness]
Dmat = cfc.hooke(ptype, E, poisson)

# Physical properties
rho = 7800  # density
g = 9.81
m = 130
A = b * h

# Forces
P = -m * g  # Point load (downward, negative y-direction)
q0 = -h * b * rho * g  # Distributed load per unit length (downward)

# Create mesh
p1 = np.array([0., 0.])  # Lower left corner
p2 = np.array([L2, h])   # Upper right corner
nelx = 20
nely = 4
ndof_per_node = 2
nnode = (nelx + 1) * (nely + 1)
nDofs = ndof_per_node * nnode

# Generate mesh
Ex, Ey, Edof, B1, B2, B3, B4, P1, P2, P3, P4 = quadmesh(p1, p2, nelx, nely, ndof_per_node)

# Plot mesh
cfv.figure()
cfv.eldraw2(Ex, Ey)
cfv.title('Mesh')
cfv.show()

display(B2)


# Initialize global stiffness matrix and force vector
K = np.zeros([nDofs, nDofs])
f = np.zeros([nDofs, 1])

# Assemble stiffness matrix
for eltopo, elx, ely in zip(Edof, Ex, Ey):
    Ke = cfc.planqe(elx, ely, ep, Dmat)
    cfc.assem(eltopo, K, Ke)

# Apply distributed load on top edge (B3)
# For a uniformly distributed load on a line element, the equivalent nodal forces
# are q*L/2 at each node (where L is the length of the edge)

# Find which elements have their top edge on B3
# Top edge elements are those in the top row (last nely row)
top_elements = range(nelx * (nely - 1), nelx * nely)


for el_idx in top_elements:
    # Get element coordinates
    elx = Ex[el_idx, :]
    ely = Ey[el_idx, :]
    
    # Get element topology
    eltopo = Edof[el_idx, :]
    
    # For a quad element, the top edge connects nodes at indices 2 and 3
    # Calculate edge length
    #edge_length = np.sqrt((elx[2] - elx[3])**2 + (ely[2] - ely[3])**2)
    L = np.hypot(elx[2] - elx[3], ely[2] - ely[3])

    # Equivalent nodal forces for uniform distributed load
    # For a uniformly distributed load q over length L:
    # Each node gets q*L/2
    nodal_force = q0 * L / 2.0
    
    # Apply to the y-DOFs of nodes 3 and 4 (indices 4,5,6,7 in eltopo)
    # Node 3 (top-right): eltopo[4] = x-dof, eltopo[5] = y-dof
    # Node 4 (top-left): eltopo[6] = x-dof, eltopo[7] = y-dof
    f[eltopo[5] - 1] += nodal_force  # y-DOF of node 3 (0-indexed)
    f[eltopo[7] - 1] += nodal_force  # y-DOF of node 4 (0-indexed)

# Apply point load P at top right corner (P3)
# P3 contains the DOFs of the top-right corner node
#f[P3[1] - 1] += P  # y-direction DOF (subtract 1 for 0-indexing)
f[B2[0]] += P


print(f"Total distributed load: {q0 * L1:.2f} N")
print(f"Point load: {P:.2f} N")
print(f"Total load: {q0 * L1 + P:.2f} N")

# Apply boundary conditions - fixed left edge (B4)
bc = B4
bcval = np.zeros(np.size(bc))

# Solve the system
a, r = cfc.solveq(K, f, bc, bcval)

# Extract displacements
print(f"\nMaximum displacement: {np.min(a):.6e} m")
print(f"Maximum displacement (mm): {np.min(a)*1000:.4f} mm")

Ed = cfc.extract_eldisp(Edof, a)

# Deformations
plt.figure()
plotpar = [2, 1, 0]  # Plotting parameters
cfv.eldraw2(Ex, Ey, plotpar)  # Drawing the original geometry
plt.title('Original geometry')
# Drawing the deformed structure
#sfac = cfv.scalfact2(Ex[2, :], Ey[2, :], Ed[2, :], 1)
#plotpar = [1, 2, 1]
#sfac = 40  # Scaling factor to see the actual deformations
#cfv.eldisp2(Ex, Ey, Ed, plotpar, sfac)
#plt.title('Displacement')

# -------------------------------
# Plot undeformed and deformed mesh manually
# -------------------------------


# Compute nodal coordinates
xv = np.linspace(p1[0], p2[0], nelx + 1)
yv = np.linspace(p1[1], p2[1], nely + 1)
coords = np.array([[x, y] for y in yv for x in xv])  # (N, 2) array

# Deformed coordinates
U = a[0::2].reshape(-1, 1)  # x-displacements
V = a[1::2].reshape(-1, 1)  # y-displacements

sfac = 40  # deformation scale
coords_def = coords + sfac * np.hstack([U, V])

# Plot
plt.figure()
for elx, ely in zip(Ex, Ey):
    # Undeformed
    plt.plot(np.append(elx, elx[0]), np.append(ely, ely[0]), 'k-', lw=0.8, alpha=0.5)

for e, (elx, ely) in enumerate(zip(Ex, Ey)):
    # Get the deformed coordinates for each element
    nodes = ((Edof[e, ::ndof_per_node] - 1) // ndof_per_node).astype(int)
    x_def = coords_def[nodes, 0]
    y_def = coords_def[nodes, 1]
    plt.plot(np.append(x_def, x_def[0]), np.append(y_def, y_def[0]), 'r-')

plt.gca().set_aspect('equal')
plt.title(f"Deformed mesh (scale factor = {sfac:.1f})")
plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.tight_layout()
plt.savefig(f"deformed_mesh_{L1 if p2[0]==L1 else L2}m.png", dpi=300, bbox_inches='tight')
plt.show()

# %%
####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################




# Assignment 1 - Read Abaqus data



####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################

datasets = {}
current_data = []
current_name = None

with open('abaqus_results.rpt', 'r') as f:
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
        
print(datasets.items())

# Plot all datasets
for name, data in datasets.items():
    plt.figure()
    plt.plot(data[:, 0], data[:, 1], label=name)
    plt.xlabel(name.split()[0])
    plt.ylabel(name.split()[1] if len(name.split()) > 1 else '')
    plt.title(name)
    plt.legend()
    plt.savefig(str(name), dpi=400, bbox_inches='tight')
    plt.show()

    
u2_3m = [-1.016e-4, -1.032e-4, -1.038e-4, -1.038e-4]
u2_03m = [-1.148e-1, -1.165e-1, -1.170e-1, -1.170e-1]
meshsize = [0.01, 0.005, 0.001, 0.0005]

name = 'Mesh convergence (3m)'
plt.figure()
plt.plot(meshsize, u2_3m,'X-')
plt.title(name)
plt.xlabel('meshsize (m)')
plt.ylabel('displacement $w$')
plt.savefig(str(name), dpi=400, bbox_inches='tight')
plt.show()

name = 'Mesh convergence (03m)'
plt.figure()
plt.title(name)
plt.plot(meshsize, u2_03m,'X-')
plt.xlabel('meshsize (m)')
plt.ylabel('displacement $w$')
plt.savefig(str(name), dpi=400, bbox_inches='tight')
plt.show()


datasets = {}
current_data = []
current_name = None

with open('abaqus_results_mises.rpt', 'r') as f:
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
        
print(datasets.items())

# Plot all datasets
for name, data in datasets.items():
    if name == 'X 3m':
        plt.figure()
        plt.plot(data[:, 0], data[:, 1], label= 'smises_bottom_X_03m')
        plt.xlabel(name.split()[0])
        plt.ylabel('smises_bottom_X_03m')
        plt.title('smises_bottom_X_03m')
        plt.legend()
        plt.savefig(str('smises_bottom_X_03m'), dpi=400, bbox_inches='tight')
        plt.show()
    if name == 'X m':
        plt.figure()
        plt.plot(data[:, 0], data[:, 1], label= 'smises_bottom_X_3m')
        plt.xlabel(name.split()[0])
        plt.ylabel('smises_bottom_X_3m')
        plt.title('smises_bottom_X_3m')
        plt.legend()
        plt.savefig(str('smises_bottom_X_3m'), dpi=400, bbox_inches='tight')
        plt.show()


#%%
####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################



# Assignment 2 - numerical v2



####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################

p_num = 120 * 10**6

# Define symbols
p, q0, a, b, E, nu, h, r, f_r, A1, A2 = symbols('p q0 a b E nu h r f_r A1 A2', real = True)

D = 1 / 12 / (1 - nu**2) * E * (h**3) # bending stiffness

# radial force = 0
f_r = 0
w = A1 * r/2 + A2 / r

w_prime = diff(w,r) # rotation field

eps_r = diff(w, r)
sigma_r = D * diff(w, r)

# Apply the boundary conditions
boundary_conditions = [
                        w.subs(r, a),               # inner boundary displacement 0
                        sigma_r.subs(r, b) - p,     # outer boundary stress
                       ]

# Solve for unknown constants
unknowns = (A1, A2)
sol= solve(boundary_conditions, unknowns)

# Formulate the deflection field
w_ = simplify(w.subs(sol)) # constants substituted

print("w(r) = ")
display(w_)

# Plot the deflection field for a given set of parameters
wp_f = simplify(w_.subs({f_r:0, p:p_num, q0:0, E:E2_num, nu:poisson_num, a:a_num, b:b_num, h:h0_num})) # parameters substituted

r_num  = np.linspace(150, 250, 401)
wr_num = [wp_f.subs({r:val}) for val in r_num]

print(r_num)
print(wr_num)

plt.figure()
plt.plot(r_num, wr_num, "b-")
plt.axvline(a_num, color='black', linestyle='--', label='a')
plt.axvline(b_num, color='grey', linestyle='--', label='b')
plt.title('Radial deflection')
plt.xlabel(r"$r$ [mm]")
plt.ylabel(r"$w$ [mm]")
plt.grid()
plt.legend()
plt.savefig('Radial deflection', dpi=dpi, bbox_inches='tight')
plt.show()

# Substitute the solution constants
sigma_rr_ = simplify(sigma_r.subs(sol))

print("sigma_rr(r) = ")
display(sigma_rr_)

# Plot the radial stress field for the same parameters
sigma_rr_f = simplify(sigma_rr_.subs({f_r: 0, p:p_num, q0:0, E:E2_num, nu:poisson_num, a:a_num, b:b_num, h:h0_num}))

sigma_rr_num = [sigma_rr_f.subs({r:val}) for val in r_num]

plt.figure()
plt.plot(r_num, sigma_rr_num, "r-")
plt.axvline(a_num, color='black', linestyle='--', label='a')
plt.axvline(b_num, color='grey', linestyle='--', label='b')
plt.axhline(0, color='gray', linestyle='-', linewidth=0.5)
plt.title('Normal stress')
plt.xlabel(r"$r$ [mm]")
plt.ylabel(r"$\sigma_{rr}$ [MPa]")
plt.grid()
plt.legend()
plt.savefig('Normal stress', dpi=dpi, bbox_inches='tight')
plt.show()

# %%
####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################




# Assigment 2 - Calfem, constant area



####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################
new_prob('2 - Calfem')

b_outer = 0.25
nu=0.3
E=200e9
h0 = 0.02
Width = b
num_el = 6
nnodes = num_el + 1
a_inner = 0.15

coords = np.linspace(b_outer, a_inner, nnodes)
#coords = coords.reshape(-1,1)

Edof = np.zeros((num_el, 2), dtype=int)
for i in range(num_el):
    Edof[i, 0] = i + 1 
    Edof[i, 1] = i + 2       

#display(coords)
display(Edof)

num_dofs = np.max(Edof)

#Dmat = E / (1-nu**2) * np.array([[1, nu],
                                 #[nu,1]])

num_dofs = np.max(Edof)
K = np.zeros((num_dofs, num_dofs))
f = np.zeros((num_dofs, 1))

print("K shape:", K.shape)

for el in range(num_el):
    r1 = coords[el]
    r2 = coords[el + 1]
    Le = r2 - r1
    r_mean = 0.5 * (r1 + r2)
    
    # Constant thickness
    h_e = h0

    print('h_e',np.shape(h_e))
    print('r_mean',np.shape(r_mean))

    
    Ke = (2 * np.pi * E * h_e * r_mean / Le) * np.array([[1, -1],
                                                        [-1,  1]])
    cfc.assem(Edof[el, :], K, Ke)

# Bcs
bc = np.array([nnodes])  
bcVal = np.array([0.0])

# Applying distributed load
sigma_r = -120e6  # 1 MPa
r_inner = coords[0]
h_inner = h0
f[0, 0] = 2 * np.pi * r_inner * h_inner * sigma_r  # negative radial direction


a, r= cfc.solveq(K, f, bc, bcVal)


plt.figure()
plt.plot(coords, a * 1e3, 'o-', label='$u_r(r)$ [mm]')
plt.xlabel('r [m]')
plt.ylabel('Radial displacement [mm]')
plt.title('Axisymmetric radial displacement')
plt.grid(True)
plt.legend()
plt.savefig('Axisymmetric radial displacement (constant height)', dpi=dpi, bbox_inches='tight')
plt.show()


for i in range(nnodes):
    print(f"Node {i+1}: r = {coords[i]:.4f} m, ur = {a[i,0]*1e6:.3f} μm")

# Computing radial stress
sigma_rr_vals = np.zeros(num_el)
r_centers = np.zeros(num_el)

for el in range(num_el):
    r1 = coords[el]
    r2 = coords[el + 1]
    Le = r2 - r1
    r_mean = 0.5 * (r1 + r2)
    r_centers[el] = r_mean

    u1 = a[el, 0]
    u2 = a[el + 1, 0]
    du_dr = (u2 - u1) / Le

    # Plane stress approximation
    sigma_rr = E / (1 - nu**2) * (du_dr + nu * (u1 + u2) / (2 * r_mean))
    sigma_rr_vals[el] = sigma_rr

# Plotting radial stress
plt.figure()
plt.plot(r_centers, sigma_rr_vals/1e6, 'o-', label=r'$\sigma_{rr}$ [MPa]')
plt.xlabel('r [m]')
plt.ylabel(r'$\sigma_{rr}$ [MPa]')
plt.title('Radial stress distribution σ_rr(r)')
plt.grid(True)
plt.legend()
plt.savefig('Radial stress distribution (constant height)', dpi=dpi, bbox_inches='tight')
plt.show()


# %%
####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################




# Assigment 2 - Calfem, varying area



####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################


b_outer = 0.25
nu=0.3
E=200e9
h0 = 0.02
Width = b
num_el = 6
nnodes = num_el + 1
a_inner = 0.15

coords = np.linspace(b_outer, a_inner, nnodes)

display(coords)
#coords = coords.reshape(-1,1)

Edof = np.zeros((num_el, 2), dtype=int)
for i in range(num_el):
    Edof[i, 0] = i + 1       # First column
    Edof[i, 1] = i + 2       # Second column

#display(coords)
display(Edof)

num_dofs = np.max(Edof)

#Dmat = E / (1-nu**2) * np.array([[1, nu],
                                 #[nu,1]])

num_dofs = np.max(Edof)
K = np.zeros((num_dofs, num_dofs))
f = np.zeros((num_dofs, 1))

print("K shape:", K.shape)

for el in range(num_el):
    r1 = coords[el]
    r2 = coords[el + 1]
    Le = r2 - r1
    r_mean = 0.5 * (r1 + r2)
    
    # variable thickness
    h_e = h0 * (r_mean - a_inner) / (b_outer - a_inner) + h0

    print('h_e',np.shape(h_e))
    print('r_mean',np.shape(r_mean))

    # local stiffness (axisymmetric linear element)
    Ke = (2 * np.pi * E * h_e * r_mean / Le) * np.array([[1, -1],
                                                        [-1,  1]])
    cfc.assem(Edof[el, :], K, Ke)

# --- Boundary conditions ---
bc = np.array([nnodes])  # fix outer radius (clamped)
bcVal = np.array([0.0])

# Applying distributed load
sigma_r = -120e6 
r_inner = coords[0]
h_inner = h0
f[0, 0] = 2 * np.pi * r_inner * h_inner * sigma_r  # negative radial direction


a, r= cfc.solveq(K, f, bc, bcVal)


plt.figure()
plt.plot(coords, a * 1e3, 'o-', label='$u_r(r)$ [mm]')
plt.xlabel('r [m]')
plt.ylabel('Radial displacement [mm]')
plt.title('Axisymmetric radial displacement')
plt.grid(True)
plt.legend()
plt.savefig('Axisymmetric radial displacement (varying height)', dpi=dpi, bbox_inches='tight')
plt.show()


for i in range(nnodes):
    print(f"Node {i+1}: r = {coords[i]:.4f} m, ur = {a[i,0]*1e6:.3f} μm")

# Computing normal stress, radial

sigma_rr_vals = np.zeros(num_el)
r_centers = np.zeros(num_el)

for el in range(num_el):
    r1 = coords[el]
    r2 = coords[el + 1]
    Le = r2 - r1
    r_mean = 0.5 * (r1 + r2)
    r_centers[el] = r_mean

    u1 = a[el, 0]
    u2 = a[el + 1, 0]
    du_dr = (u2 - u1) / Le

    # Plane stress approximation
    sigma_rr = E / (1 - nu**2) * (du_dr + nu * (u1 + u2) / (2 * r_mean))
    sigma_rr_vals[el] = sigma_rr

# Plotting radial stress
plt.figure()
plt.plot(r_centers, sigma_rr_vals/1e6, 'o-', label=r'$\sigma_{rr}$ [MPa]')
plt.xlabel('r [m]')
plt.ylabel(r'$\sigma_{rr}$ [MPa]')
plt.title('Radial stress distribution σ_rr(r)')
plt.grid(True)
plt.legend()
plt.savefig('Radial stress distribution (varying height)', dpi=dpi, bbox_inches='tight')
plt.show()
#%%