#%%
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import math
import pandas as pd
import sys
import os
import scipy.sparse.linalg as spla

import gmsh
import scipy.io
from matplotlib.widgets import Slider
from matplotlib.gridspec import GridSpec
from numpy import random
import time
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
import multiprocessing as mp
import tkinter as tk
from tkinter import simpledialog
import json
from __future__ import annotations
from typing import List, Optional, Sequence, Tuple
import plotly.graph_objects as go
from IPython.display import Math, display
from typing import Union, Dict
from typing import Literal 
from scipy.linalg import eigh
from plotly.express.colors import sample_colorscale
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
from scipy.sparse import coo_matrix, csr_matrix
import matplotlib.cm as cm
from pathlib import Path

# Add parent directory to sys.path
parent_dir = os.path.abspath(os.path.join(os.getcwd(),'..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

def new_task(string):
    print_string = '\n' + '=' * 80 + '\n' + '=' * 80 + '\n' + str(string) + '\n' + '=' * 80 + '\n' + '=' * 80 + '\n'
    return print(print_string)

def new_subtask(string):
    print_string = '\n' + '=' * 80 + '\n' + str(string) + '\n' + '=' * 80 + '\n'
    return print(print_string)

def printt(string):
    print()
    print('=' * 40)
    print(string)
    print('=' * 40)
    print()

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


def displayvar(name: str, var, post: Optional[str] = None, accuracy: Optional[int] = None) -> None:
    if isinstance(var, np.ndarray):
        var = sp.Matrix(var)
    if accuracy is None:
        display(Math(f"{name} = {sp.latex(var)}") )
    else:
        display(Math(f"{name} \\approx {sp.latex(sp.sympify(var).evalf(accuracy))}"))

#===================================================================================================
####################################################################################################
new_subtask('Task 2 - CE1 ')
####################################################################################################
#===================================================================================================

# ------------------------------------------------------------------
# Be, Ae
# ------------------------------------------------------------------

# Isoparametric coordinates
xi1, xi2 = sp.symbols('xi1 xi2', real=True)

# Node positions (vectors)
xi = sp.Matrix([xi1, xi2])
xe1_1, xe1_2 = sp.symbols('xe1_1 xe1_2', real=True)
xe2_1, xe2_2 = sp.symbols('xe2_1 xe2_2', real=True)
xe3_1, xe3_2 = sp.symbols('xe3_1 xe3_2', real=True)

xe1 = sp.Matrix([xe1_1, xe1_2])
xe2 = sp.Matrix([xe2_1, xe2_2])
xe3 = sp.Matrix([xe3_1, xe3_2])

# Shape functions
N1 = 1 - xi1 - xi2
N2 = xi1
N3 = xi2

# Partial derivatives
dN1_dxi = sp.Matrix([sp.diff(N1, xi1), sp.diff(N1, xi2)])
dN2_dxi = sp.Matrix([sp.diff(N2, xi1), sp.diff(N2, xi2)])
dN3_dxi = sp.Matrix([sp.diff(N3, xi1), sp.diff(N3, xi2)])

# Mapping and Jacobian
x = N1*xe1 + N2*xe2 + N3*xe3
J = x.jacobian(xi)
J_inv_T = J.inv().T

# Partial derivatives of shape function, wrt global coords
dN1_dx = sp.simplify(J_inv_T * dN1_dxi)
dN2_dx = sp.simplify(J_inv_T * dN2_dxi)
dN3_dx = sp.simplify(J_inv_T * dN3_dxi)

# B-matrix
Be = sp.Matrix([
[dN1_dx[0], 0, dN2_dx[0], 0, dN3_dx[0], 0],
[0, dN1_dx[1], 0, dN2_dx[1], 0, dN3_dx[1]],
[dN1_dx[1], dN1_dx[0], dN2_dx[1], dN2_dx[0], dN3_dx[1], dN3_dx[0]]
])

Ae = sp.simplify(0.5 * J.det())

# Callable functions
Be_func_cst = sp.lambdify((xe1, xe2, xe3), Be, modules="numpy")
Ae_func = sp.lambdify((xe1, xe2, xe3), Ae, modules="numpy")

# ------------------------------------------------------------------
# To save computation time during assembly:
# Precompute sparse pattern once
# ------------------------------------------------------------------
def precompute_pattern(Edof):
    Edof0 = Edof[:, 1:].astype(np.int64) - 1
    nel, ndofe = Edof0.shape
    nnz_per_el = ndofe * ndofe
    nnz_total  = nel * nnz_per_el

    ii = np.repeat(np.arange(ndofe), ndofe)
    jj = np.tile(np.arange(ndofe), ndofe)

    rows = np.empty(nnz_total, dtype=np.int64)
    cols = np.empty(nnz_total, dtype=np.int64)

    p = 0
    for el in range(nel):
        edof = Edof0[el]
        rows[p:p+nnz_per_el] = edof[ii]
        cols[p:p+nnz_per_el] = edof[jj]
        p += nnz_per_el

    return rows, cols

# ------------------------------------------------------------------
# To save computation time:
# precompute Be and Ae matrices for all elements and gauss points
# ------------------------------------------------------------------
def create_Be_Ae_matrix(nel, Ex, Ey):
    
    ngp=1 #number of Gauss points
    Be_matrix = np.zeros((nel,ngp, 3, 6))
    Ae_matrix = np.zeros((nel, ngp))

    for el in range(nel):
        x1 = np.array([Ex[el,0], Ey[el,0]])
        x2 = np.array([Ex[el,1], Ey[el,1]])
        x3 = np.array([Ex[el,2], Ey[el,2]])
        

        for gp in range(ngp):
            Be = Be_func_cst(x1, x2, x3)
            Be_matrix[el, gp, :, :] = Be
            Ae = Ae_func(x1, x2, x3)
            Ae_matrix[el, gp] = Ae
            
    return Be_matrix, Ae_matrix

# ------------------------------------------------------------------
# Assembling routine uses precomputed COO pattern 
# and precomputed Be and Ae matrices
# ------------------------------------------------------------------
def assemble_K_fint_coo(a, Edof, rows, cols, ndof, nel, Ex, Ey, D, body, thickness,my_element):
    
    Be_matrix, Ae_matrix = create_Be_Ae_matrix(nel, Ex, Ey)
    
    # Initialize global internal force vector
    f_ext = np.zeros(ndof, dtype=float)

    # Number of DOFs per element (e.g. 12 for tri6 with 2 DOF/node)
    ndofe = Edof.shape[1]-1

    # Each element contributes a dense (ndofe x ndofe) stiffness block
    nnz_per_el = ndofe * ndofe
    nnz_total = nel * nnz_per_el
    
    # Preallocate COO triplet arrays (row, col, value)
    data = np.empty(nnz_total, dtype=float)
    # Pointer into the preallocated triplet arrays
    p = 0

    # Element loop
    for el in range(nel):
        # Element DOF indices
        edof = Edof[el, 1:].astype(np.int64) -1
        # Compute element internal force and stiffness matrix
        ae = a[edof]
        fe, Ke, *_ = my_element(
            ae, el, Be_matrix, Ae_matrix, D, body, thickness) #Ex, Ey, D, body,  thickness
        
        # Assemble internal force contributions
        f_ext[edof] += fe

        # Assemble stiffness using COO triplets
        data[p:p + nnz_per_el] = Ke.ravel()
        p += nnz_per_el

    # Build global stiffness matrix and convert to CSR for efficient slicing/solves
    K = coo_matrix((data, (rows, cols)), shape=(ndof, ndof)).tocsr()

    # Sum duplicate entries (multiple elements contribute to same (i,j) location)
    K.sum_duplicates()

    return K, f_ext

# ------------------------------------------------------------------
# Constant strain element
# ------------------------------------------------------------------
def cst_element(ae, el, Be_matrix, Ae_matrix, D, body, h):
    ngp=1; fe=np.zeros(6); Ke=np.zeros((6,6))

    for gp in range(ngp):
        Ae = Ae_matrix[el, gp]
        Be = Be_matrix[el, gp, :, :]
        fe = np.tile([body[el], body[el]], 3) * Ae / 3
        Ke = Be.T @ D @ Be * Ae * h  
             
    return fe, Ke

#%%
#===================================================================================================
####################################################################################################
new_subtask('Task 2 a) - Extension of CE1')
####################################################################################################
#===================================================================================================
# Define inputs
H_val = 0.1 # m
B_val = 0.1 # m
h_val = 0.1 # m
E_val = 20 # MPa
nu_val = 0.45 

# Constitutive matrix
D = (E_val / (1 - nu_val**2)) * np.array([
            [1,   nu_val,       0],
            [nu_val,   1,       0],
            [0,   0,  (1-nu_val)/2]
        ])

# ------------------------------------------------------------------
# Read matlab file (from read_matfiles-1.py)
# ------------------------------------------------------------------
def read_toplogy_from_mat_file(filename):
    mat_file = sio.loadmat(filename)
    # edof_x=mat_file['Edof'] #note that edof matlab contains one column too much
    # Edof=edof_x[:,1:13]
    
    Edof=mat_file['Edof']; Edof = Edof.astype(int)
    Ex=mat_file['Ex']
    Ey=mat_file['Ey']
    
    dof_lower=mat_file['dof_lower'].ravel() 
    dof_lower=dof_lower.astype(int)
    dof_upper=mat_file['dof_upper'].ravel()
    dof_upper=dof_upper.astype(int) 
    dof_right=mat_file['dof_right'].ravel()
    dof_right=dof_right.astype(int)
    dof_left=mat_file['dof_left'].ravel()
    dof_left=dof_left.astype(int)
    dof_corner=mat_file['dof_corner'].ravel()
    dof_corner=dof_corner.astype(int)
    
    ndofs=(mat_file['ndofs']).item()
    nelem=mat_file['nelem'].item()
    nnodes=mat_file['nnodes'].item()
    
    return Ex, Ey, Edof, dof_upper, dof_lower, ndofs, nelem, nnodes

filename='topology_coarse_3node.mat'
Ex,Ey,Edof,dof_upper,dof_lower,ndofs,nelem,nnodes=read_toplogy_from_mat_file(filename)

# ------------------------------------------------------------------
# Plot undeformed mesh
# ------------------------------------------------------------------
from matplotlib.collections import PolyCollection

polygons = np.zeros((nelem, 3, 2))
for el in range(nelem):
    polygons[el,:,:] = [[Ex[el,0], Ey[el,0]], 
                        [Ex[el,1], Ey[el,1]], 
                        [Ex[el,2], Ey[el,2]]]
    
fig1, ax1 = plt.subplots()
pc1 = PolyCollection(
    polygons,
    facecolors='none',
    edgecolors='k'
)
ax1.add_collection(pc1)
ax1.autoscale()

# ------------------------------------------------------------------
# Precompute pattern
# ------------------------------------------------------------------
ndofs = 2 * nnodes
rows, cols= precompute_pattern(Edof)

# ------------------------------------------------------------------
# Simulation parameters, store results
# ------------------------------------------------------------------
n_steps = 10 # Number of time steps
tol = 1e-6 # Newton tolerance
max_iter = 10 # Max iterations
thickness = h_val # Thickness h
body = np.zeros(nelem) # No body force

disp_history = []
force_history = []
u_vals = []

# LC1: u_gamma = +20mm
# LC2: u_gamma = -15mm
target_displacement = 20e-3
load_case_name = 'LC1 - Tension'
# load_case_name = 'LC2 - Compression'

# Initialize global displacement vector
a = np.zeros(ndofs)

# ------------------------------------------------------------------
# Time stepping loop
# ------------------------------------------------------------------
for step in range(1, n_steps + 1):
    t = step / n_steps
    current_u_gamma = t * target_displacement
    
    # -- Define Boundary Conditions for this step --
    dof_C = [] 
    
    # Example BCs (adjust according to "define and motivate your choice" in Task 1a)
    # Symmetry/Fixed bottom:
    dof_C.extend(dof_lower) # Fix bottom
    
    # Prescribed displacement
    prescribed_dofs = dof_upper
    
    # Combine constrained DOFs
    bc_dofs = np.concatenate([dof_C, prescribed_dofs])
    bc_vals = np.zeros(len(bc_dofs))
    
    # Set values: 0 for fixed, current_u_gamma for prescribed
    bc_vals[len(dof_C):] = current_u_gamma
    
    # Free DOFs
    dof_all = np.arange(ndofs)
    dof_F = np.setdiff1d(dof_all, bc_dofs)
    
    # Update 'a' with prescribed values immediately
    a[bc_dofs] = bc_vals
    
    # Newton-Raphson Loop
    print(f"\nStep {step}/{n_steps}, disp: {current_u_gamma:.2e} m")
    
    for i in range(max_iter):
        K, f_ext = assemble_K_fint_coo(a, Edof, rows, cols, ndofs, nelem, Ex, Ey, D, body, h_val, my_element=cst_element)

        # Internal force vector
        f_int = K @ a
        
        # Residual
        r = f_int - f_ext
        
        # Partition residual
        r_F = r[dof_F]
        
        # Check convergence
        res_norm = np.linalg.norm(r_F)
        if res_norm < tol:
            print(f'Converged in {i} iterations, \nresidual: {res_norm:.2e}')
            break
            
        # Solve for increment
        K_F = K[dof_F, :][:, dof_F]
        da_F = spla.spsolve(K_F, -r_F)
        
        # Update solution
        a[dof_F] += da_F
        
        if i == max_iter - 1:
            print('Max iterations reached without convergence.')

    # Post-Processing
    r_tot = (K @ a) - f_ext
    
    # Sum vertical reaction forces on the upper boundary
    Ry_sum = np.sum(r_tot[dof_upper])
    
    # Store results
    u_vals.append(current_u_gamma)
    force_history.append(Ry_sum)

# ------------------------------------------------------------------
# Plot deformed mesh
# ------------------------------------------------------------------
def_polygons = np.zeros((nelem, 3, 2))

# Magnification factor
mag = 1000

for el in range(nelem):
    edofs = Edof[el,1:] - 1
    
    def_polygons[el,:,:] = [
        [Ex[el,0] + mag * a[edofs[0]], Ey[el,0] + mag * a[edofs[1]]],
        [Ex[el,1] + mag * a[edofs[2]], Ey[el,1] + mag * a[edofs[3]]],
        [Ex[el,2] + mag * a[edofs[4]], Ey[el,2] + mag * a[edofs[5]]]
    ]

fig2, ax2 = plt.subplots()
pc2 = PolyCollection(
    def_polygons,
    facecolors='none',
    edgecolors='r'
)
ax2.add_collection(pc2)
ax2.autoscale()
ax2.set_title("Deformed mesh")

# ------------------------------------------------------------------
# Plot stress
# ------------------------------------------------------------------
Es = np.zeros((nelem, 3))

for el in range(nelem):
    x1 = np.array([Ex[el,0], Ey[el,0]])
    x2 = np.array([Ex[el,1], Ey[el,1]])
    x3 = np.array([Ex[el,2], Ey[el,2]])

    Be = Be_func_cst(x1, x2, x3)
    edofs = Edof[el,1:] - 1

    Es[el,:] = D @ Be @ a[edofs]

fig3, ax3 = plt.subplots()
pc3 = PolyCollection(
    polygons,
    array=Es[:,0], 
    cmap='turbo', 
    edgecolors='k'
)
ax3.add_collection(pc3)
ax3.autoscale()
ax3.set_title("sigma xx")
fig2.colorbar(pc3, ax=ax3)

# ------------------------------------------------------------------
# Plot graph
# ------------------------------------------------------------------
title = 'vertical reaction force vs uΓ'
plt.figure()
plt.plot(u_vals, force_history, 'X-')
plt.title(title)
plt.xlabel('uΓ [m]')
plt.ylabel('vertical reaction force [N]')
sfig(title)
plt.show()

#%%
#===================================================================================================
####################################################################################################
new_subtask('Task 2 b) - Yeoh hyperelsatic material model')
####################################################################################################
#===================================================================================================

# Define inputs
F_11_min = 0.5
F_11_max = 1.5
F_11_vals = np.linspace(F_11_min, F_11_max, 100)
placeholder_sigma11_list = np.linspace(0, 100, 100)

G_val = E_val / (2 * (1 + nu_val))
lam_val = E_val * nu_val / ((1 + nu_val) * (1 - 2 * nu_val))

# Yeoh Parameters
c10 = G_val / 2
c20 = - G_val / 10
c30 = G_val / 30
D1 = 0.02 # 1 / Pa
D2 = 0.01
D3 = 0.01

def yeoh_functions():
    Fv = sp.Matrix(sp.symbols('Fv0:4', real=True)) 
    
    F = sp.Matrix([
        [Fv[0], Fv[2], 0],
        [Fv[1], Fv[3], 0], 
        [0,     0,     1]
    ])
    
    # Large deformation kinematics
    C = F.T * F
    J = F.det()
            
    U0_val = c10 * (J**(-sp.Rational(2, 3)) * sp.trace(C) - 3) \
            + c20 * (J**(-sp.Rational(2, 3)) * sp.trace(C) - 3)**2 \
            + c30 * (J**(-sp.Rational(2, 3)) * sp.trace(C) - 3)**3 \
            + (1 / D1) * (J - 1)**2 \
            + (1 / D2) * (J - 1)**4 \
            + (1 / D3) * (J - 1)**6

    P_vec = sp.diff(U0_val, Fv)
    dPvdFv = P_vec.jacobian(Fv)
    
    # Validation helper
    P_mat = sp.Matrix([
        [P_vec[0], P_vec[2], 0], 
        [P_vec[1], P_vec[3], 0], 
        [0,0,0]])
    S_mat = F.inv() * P_mat
    S_vec_out = sp.Matrix([S_mat[0,0], S_mat[1,1], S_mat[0,1], S_mat[1,0]])
    
    P_func = sp.lambdify([Fv], P_vec, modules="numpy", cse=True)
    dPvdFv_func = sp.lambdify([Fv], dPvdFv, modules="numpy", cse=True)
    S_func = sp.lambdify([Fv], S_vec_out, modules='numpy', cse=True)
    
    return P_func, dPvdFv_func, S_func

def neohooke_functions():
    Fv = sp.Matrix(sp.symbols('Fv0:4', real=True))
    F = sp.Matrix([[Fv[0], Fv[2], 0], [Fv[1], Fv[3], 0], [0, 0, 1]])
    C = F.T * F 
    invC = sp.simplify(C.inv())
    J = F.det()

    S = G_val * (sp.eye(3) - invC) + lam_val * sp.log(J) * invC
    P = F * S
    Pv = sp.Matrix([P[0,0], P[1,1], P[0,1], P[1,0]])
    dPvdFv = Pv.jacobian(Fv)

    S_vec_out = sp.Matrix([S[0,0], S[1,1], S[0,1], S[1,0]])
    
    dPdF_func = sp.lambdify([Fv], dPvdFv, modules='numpy', cse=True)
    P_func = sp.lambdify([Fv], Pv, modules='numpy', cse=True)
    S_func = sp.lambdify([Fv], S_vec_out, modules='numpy', cse=True)

    return P_func, dPdF_func, S_func

# Generate functions
P_Yeoh_func, dPvdFv_Yeoh_func, S_Yeoh_func = yeoh_functions()
P_Neo_func, dPvdFv_Neo_func, S_Neo_func = neohooke_functions()

# Solve
sigma11_yeoh = []
sigma11_neo = []

for f11 in F_11_vals:
    # F11 = varying, F22 = 1.0, F33 = 1.0
    F_vec_input = [f11, 0.0, 0.0, 1.0] 
    
    # Calculate Cauchy stress
    J = f11 * 1.0
    
    # Yeoh
    s_vals_yeoh = S_Yeoh_func(F_vec_input)
    s11_yeoh = s_vals_yeoh[0][0]
    
    sig_yeoh = (1 / J) * f11 * s11_yeoh * f11
    sigma11_yeoh.append(sig_yeoh)
    
    # Neo-Hooke
    s_vals_neo = S_Neo_func(F_vec_input)
    s11_neo = s_vals_neo[0][0]
    
    sig_neo = (1 / J) * f11 * s11_neo * f11
    sigma11_neo.append(sig_neo)

printt('Validate results:')
print(f'Yeoh sigma11: {sigma11_yeoh[-1]:.4e} MPa (ref: 1.2142e02)')
print(f'Neo-Hooke sigma11: {sigma11_neo[-1]:.4e} MPa (ref: 2.2525e01)')

# ------------------------------------------------------------------
# Plot graphs
# ------------------------------------------------------------------
title = 'Cauchy stress component sigma11 vs F11 - pure elongation'
plt.figure()
plt.plot(F_11_vals, sigma11_yeoh, 'o-')
plt.title(title)
plt.xlabel('F_11')
plt.ylabel('sigma11')
sfig(title)
plt.show()

title = 'Cauchy stress component sigma11 vs F11 - pure contraction'
plt.figure()
plt.plot(F_11_vals, sigma11_yeoh, 'o-')
plt.title(title)
plt.xlabel('F_11')
plt.ylabel('sigma11')
sfig(title)
plt.show()

#%%
#===================================================================================================
####################################################################################################
new_subtask('Task 2 c) - Own element function')
####################################################################################################
#===================================================================================================

def generate_fast_tri6():
    print("Compiling symbolic element function (runs once)...")
    
    xi, eta = sp.symbols('xi eta')
    x_nodes = [sp.symbols(f'x{i}') for i in range(6)] # x1..x6
    y_nodes = [sp.symbols(f'y{i}') for i in range(6)] # y1..y6
    
    # Shape Functions
    L = 1.0 - xi - eta
    Ns = [
        L*(2*L-1),       # N1
        xi*(2*xi-1),     # N2
        eta*(2*eta-1),   # N3
        4*xi*L,          # N4
        4*xi*eta,        # N5
        4*eta*L          # N6
    ]
    
    # Mappings
    x_map = sum(N*x for N, x in zip(Ns, x_nodes))
    y_map = sum(N*y for N, y in zip(Ns, y_nodes))
    
    # Jacobian
    J = sp.Matrix([[sp.diff(x_map, xi), sp.diff(x_map, eta)],
                   [sp.diff(y_map, xi), sp.diff(y_map, eta)]])
    detJ = J.det()
    invJ = J.inv()
    
    # Derivatives dN/dX (Reference Config)
    dN_dxi_list = [sp.Matrix([sp.diff(N, xi), sp.diff(N, eta)]) for N in Ns]
    dN_dX_list = [invJ.T @ dN for dN in dN_dxi_list]
    
    B = sp.zeros(4, 12)
    for i in range(6):
        dN_dX = dN_dX_list[i]
        # F11 (row 0), F21 (row 1) use dN/dX
        B[0, 2*i]   = dN_dX[0] # u
        B[1, 2*i+1] = dN_dX[0] # v
        # F12 (row 2), F22 (row 3) use dN/dY
        B[2, 2*i]   = dN_dX[1] # u
        B[3, 2*i+1] = dN_dX[1] # v

    # Compile
    return sp.lambdify([xi, eta] + x_nodes + y_nodes, [B, detJ], 'numpy')


def precompute_element(X_ref, fast_func, thickness=1.0):
    # Gauss points
    gps = [(1./6, 1./6, 1./6), (2./3, 1./6, 1./6), (1./6, 2./3, 1./6)]
    
    # Flatten coordinates: x1, x2... y1, y2...
    coords = list(X_ref[:, 0]) + list(X_ref[:, 1])
    
    element_data = []
    for xi, eta, w in gps:
        # Call the compiled function
        B_val, detJ_val = fast_func(xi, eta, *coords)
        
        # Store constant data
        dv = detJ_val * w * thickness
        element_data.append((B_val, dv))
        
    return element_data


def el6_yeoh(u, element_data):
    # Pre-allocate output arrays
    Ke = np.zeros((12, 12))
    fe = np.zeros(12)
    
    # Loop over pre-computed Gauss points
    for B, dv in element_data:
        # Kinematics
        F_vec = B @ u
        F_vec[0] += 1.0
        F_vec[3] += 1.0
        
        # Material Evaluation
        P = P_Yeoh_func(F_vec).flatten() 
        A = dPvdFv_Yeoh_func(F_vec)      
        
        # Integration
        fe += (B.T @ P) * dv
        
        Ke += (B.T @ A @ B) * dv
        
    return Ke, fe

X_ref = np.array([
    [0.0, 0.0], # 1
    [3.0, 0.0], # 2
    [0.0, 2.0], # 3
    [1.5, 0.0], # 4
    [1.5, 1.0], # 5
    [0.0, 1.0]  # 6
])

x_curr = np.array([
    [6.0, 0.7], # 1
    [7.0, 2.3], # 2
    [4.5, 1.8], # 3
    [6.4, 1.2], # 4
    [5.6, 2.0], # 5
    [5.2, 1.1]  # 6
])

ex_ref = X_ref[:, 0].T
ey_ref = X_ref[:, 1].T

u = (x_curr - X_ref).flatten()

# Run once
tri6_func = generate_fast_tri6()

# Run once for each mesh
element_data = precompute_element(X_ref, tri6_func, thickness=0.001)
    
Ke_val, fe_val, F_res, P_res = el6_yeoh(u, element_data)
# Ke_val, fe_val, F_res, P_res = el6_yeoh(ex_ref, ey_ref, u, thickness=0.001, return_validation=True)

printt('REFERENCE RESULTS, FOR VALIDATION:')
print('deformation_gradient_2d:')
print(F_res.T)
print('\nPiola_Kirchoff_2d:')
print(P_res.T)
print('\nfe_int:')
print(fe_val)
print('\nKe_int top (8x8):')
print(Ke_val[0:8, 0:8])

#%%
#===================================================================================================
####################################################################################################
new_subtask('Task 2 d) - Combine routines')
####################################################################################################
#===================================================================================================

import scipy.sparse as sps

def solve_task_2d(filename, n_steps=50, tol=1e-6, u_final=20, h=100, max_iter=25):
    # ------------------------------------------------------------------
    # Solver setup
    # ------------------------------------------------------------------
    
    # Load mesh
    Ex, Ey, Edof, dof_upper, dof_lower, ndof, nel, nnodes = read_toplogy_from_mat_file(filename)
    
    # Pre-processing
    Edof_indices = Edof[:, 1:].astype(np.int64) - 1
    
    # Precompute pattern
    rows_pattern, cols_pattern = precompute_pattern(Edof)
    nnz_per_el = 12 * 12
    nnz_total = nel * nnz_per_el
    
    # Pre-allocate data array for COO matrix
    data = np.empty(nnz_total, dtype=float)

    # Element Data Storage
    tri6_func = generate_fast_tri6()
    print('Pre-computing element matrices')
    element_store = []
    
    for el in range(nel):
        X_ref = np.vstack([Ex[el, :], Ey[el, :]]).T
        # Calculate B and detJ constants once
        element_store.append(precompute_element(X_ref, tri6_func, thickness=h))
    
    # Time stepping
    a = np.zeros(ndof)
    u_steps = np.linspace(0, u_final, n_steps + 1)
    force_history = [0.0]
    disp_history = [0.0]

    # ------------------------------------------------------------------
    print(f'Starting Solver')
    # ------------------------------------------------------------------
    for step, u_app in enumerate(u_steps):
        if step == 0: continue 
        
        # Boundary Conditions
        bc_dofs = []
        bc_vals = []
        
        # Dofs
        top_y_dofs = [d - 1 for d in dof_upper if d % 2 == 0]
        bot_y_dofs = [d - 1 for d in dof_lower if d % 2 == 0]
        bot_x_dofs = [d - 1 for d in dof_lower if d % 2 == 1]

        # Values
        bc_dofs.extend(bot_y_dofs)
        bc_vals.extend([0.0] * len(bot_y_dofs))
        
        bc_dofs.extend(top_y_dofs)
        bc_vals.extend([u_app] * len(top_y_dofs))
        
        if bot_x_dofs:
            bc_dofs.append(bot_x_dofs[0])
            bc_vals.append(0.0)
            
        bc_dofs = np.array(bc_dofs, dtype=int)
        bc_vals = np.array(bc_vals)
        
        # Apply BCs
        a[bc_dofs] = bc_vals
        
        # Free DOFs mask
        free_mask = np.ones(ndof, dtype=bool)
        free_mask[bc_dofs] = False
        free_dofs = np.where(free_mask)[0]

        # ------------------------------------------------------------------
        # Newton-Raphson loop
        # ------------------------------------------------------------------
        converged = False
        
        for it in range(max_iter):
            
            # Reset global force
            f_int = np.zeros(ndof)
            
            # Element Loop
            for el in range(nel):
                # Use pre-calculated indices
                edof_idx = Edof_indices[el]
                u_loc = a[edof_idx]
                
                Ke, fe = el6_yeoh(u_loc, element_store[el])
                
                # Assemble force
                f_int[edof_idx] += fe
                
                # Store stiffness data
                idx_start = el * nnz_per_el
                data[idx_start : idx_start + nnz_per_el] = Ke.ravel()
            
            # ------------------------------------------------------------------
            # Global solve
            # ------------------------------------------------------------------
            
            # Create CSR matrix
            K_global = sps.coo_matrix((data, (rows_pattern, cols_pattern)), shape=(ndof, ndof)).tocsr()
            
            # Residual
            r = -f_int[free_dofs]
            res_norm = np.linalg.norm(r)
            
            if res_norm < tol:
                converged = True
                print(f"Step {step}: Converged (Iter {it}, Res {res_norm:.2e})")
                break
            
            # Solve system
            K_free = K_global[free_dofs, :][:, free_dofs]
            
            try:
                da_free = spla.spsolve(K_free, r)
            except Exception as e:
                print(f"\n[Error] solver failed: {e}")
                return a, disp_history, force_history
            
            a[free_dofs] += da_free

        if not converged:
            print(f'\n[Warning] step {step} did not converge.')
            break
        
        # ------------------------------------------------------------------
        # Store reaction force
        # ------------------------------------------------------------------
        Ry_top = np.sum(f_int[top_y_dofs])
        force_history.append(Ry_top)
        disp_history.append(u_app)

    return a, disp_history, force_history

def plot_stress(a, filename, plot_title=''):
    # Load mesh
    Ex, Ey, Edof, _, _, _, _, _ = read_toplogy_from_mat_file(filename)
    nel = Edof.shape[0]

    # Compile/Get the fast element function
    tri6_func = generate_fast_tri6()
    
    # Centroid coordinates
    xi_c, eta_c = 1.0/3.0, 1.0/3.0
    
    Es = np.zeros((nel, 4))
    polygons = []
    
    for el in range(nel):
        edofs = Edof[el, 1:].astype(int) - 1
        u_el = a[edofs]
        
        ex = Ex[el, :]
        ey = Ey[el, :]
        
        # Flatten coords for the function call
        coords = list(ex) + list(ey)
        
        # Get B-matrix and detJ at the centroid
        B_cent, detJ_cent = tri6_func(xi_c, eta_c, *coords)
        
        # Compute kinematics
        grad_u = B_cent @ u_el
        F_vec = np.array([1.0, 0.0, 0.0, 1.0]) + grad_u
        
        # Reconstruct F Matrix (2x2) for stress push-forward
        F_mat = np.array([
            [F_vec[0], F_vec[2]],
            [F_vec[1], F_vec[3]]
        ])
        
        # Jacobian determinant
        J = F_mat[0,0]*F_mat[1,1] - F_mat[0,1]*F_mat[1,0]
        
        # Get First Piola-Kirchhoff stress (vector)
        P_out = P_Yeoh_func(F_vec)
        P_vals = np.array(P_out).flatten()
        
        P_tensor = np.array([
            [P_vals[0], P_vals[2]],
            [P_vals[1], P_vals[3]]
        ])
        
        # Compute Cauchy stress
        Sigma = (1.0/J) * P_tensor @ F_mat.T
        
        s11 = Sigma[0,0]
        s22 = Sigma[1,1]
        s12 = Sigma[0,1]
        
        # Von Mises
        s_vm = np.sqrt(s11**2 + s22**2 - s11*s22 + 3*s12**2)
        
        Es[el, :] = [s11, s22, s12, s_vm]
        
        # Deformed coordinates
        x_nodes = ex + u_el[0::2]
        y_nodes = ey + u_el[1::2]
        
        # Create polygon
        poly_coords = np.column_stack([x_nodes[[0, 1, 2]], y_nodes[[0, 1, 2]]])
        polygons.append(poly_coords)

    # Plotting
    title = 'Von Mises Stress' + plot_title
    fig, ax = plt.subplots(figsize=(10, 8))
    pc = PolyCollection(
        polygons,
        array=Es[:, 3], # Plot Von Mises
        cmap='turbo',
        edgecolors='black',
        linewidths=0.2
    )
    ax.add_collection(pc)
    ax.autoscale()
    ax.set_aspect('equal')
    ax.set_title(f'Von Mises Stress\n{plot_title}')
    ax.set_xlabel('X [mm]')
    ax.set_ylabel('Y [mm]')
    cb = fig.colorbar(pc, ax=ax)
    cb.set_label('Von Mises Stress [MPa]')
    sfig(title)
    plt.show()

#%%
# ------------------------------------------------------------------
printt('Run solver - coarse mesh')
# ------------------------------------------------------------------
filename = 'topology_coarse_6node.mat'

a, disp_history, force_history = solve_task_2d(filename, n_steps=150, tol=1e-5, u_final=20, h=100)

#%%
# ------------------------------------------------------------------
printt('Postprocess - coarse mesh')
# ------------------------------------------------------------------
title = 'Coarse mesh - Total Vertical Reaction Force vs Displacement'
plt.figure()
plt.plot(disp_history, force_history, '-o')
plt.title(title)
plt.xlabel('Displacement u_Gamma [m]')
plt.ylabel('Reaction Force [N]')
plt.grid(True)
sfig(title)
plt.show()

plot_stress(a, filename, plot_title='coarse')
#%%
# ------------------------------------------------------------------
printt('Run solver - medium mesh')
# ------------------------------------------------------------------
filename = 'topology_medium_6node.mat'

a, disp_history, force_history = solve_task_2d(filename, n_steps=150, tol=1e-5, u_final=20, h=100)

#%%
# ------------------------------------------------------------------
printt('Postprocess - medium mesh')
# ------------------------------------------------------------------
title = 'Medium mesh - Total Vertical Reaction Force vs Displacement'
plt.figure()
plt.plot(disp_history, force_history, '-o')
plt.title(title)
plt.xlabel('Displacement u_Gamma [m]')
plt.ylabel('Reaction Force [N]')
plt.grid(True)
sfig(title)
plt.show()
plot_stress(a, filename, plot_title='medium ')

#%%
# ------------------------------------------------------------------
printt('Run solver - fine mesh')
# ------------------------------------------------------------------
filename = 'topology_fine_6node.mat'

a, disp_history, force_history = solve_task_2d(filename, n_steps=150, tol=1e-5, u_final=20, h=100)

#%%
# ------------------------------------------------------------------
printt('Postprocess - fine mesh')
# ------------------------------------------------------------------
title = 'Fine mesh - Total Vertical Reaction Force vs Displacement'
plt.figure()
plt.plot(disp_history, force_history, '-o')
plt.title(title)
plt.xlabel('Displacement u_Gamma [m]')
plt.ylabel('Reaction Force [N]')
plt.grid(True)
sfig(title)
plt.show()
plot_stress(a, filename, plot_title='fine')

#%%