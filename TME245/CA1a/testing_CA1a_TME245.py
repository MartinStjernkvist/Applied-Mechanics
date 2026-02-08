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
    """Display a variable as LaTeX: ``name = value``.

    Uses SymPy's LaTeX printer. If ``var`` is a NumPy array, it is converted to
    a SymPy Matrix for crisp typesetting. If ``accuracy`` is given, the value is
    shown approximately with that many significant digits.

    Parameters
    ----------
    name : str
        Symbolic name to display.
    var : Any
        Value to display (number, array, sympy expression, ...).
    accuracy : int, optional
        Number of significant digits for approximate print. If ``None``, exact
        expressions are printed when possible.

    Examples
    --------
    >>> displayvar("P", 1)
    >>> import numpy as np; displayvar("K", np.eye(2))
    >>> displayvar("pi", sp.pi, accuracy=5)
    """
    if isinstance(var, np.ndarray):
        var = sp.Matrix(var)
    if accuracy is None:
        display(Math(f"{name} = {sp.latex(var)}") )
    else:
        display(Math(f"{name} \\approx {sp.latex(sp.sympify(var).evalf(accuracy))}"))

#%%
####################################################################################################
####################################################################################################
####################################################################################################



new_task('Task 1 - Nonlinear elastic analysis in 2D using Matlab/Python')



####################################################################################################
####################################################################################################
####################################################################################################

#===================================================================================================
#===================================================================================================
new_subtask('Task 1')
#===================================================================================================
#===================================================================================================

H_val = 0.1 # m
B_val = 0.1 # m
h_val = 0.1 # m


#%%
####################################################################################################
####################################################################################################
####################################################################################################



new_task('Task 2 - Nonlinear elastic analysis in 2D using Matlab/Python')



####################################################################################################
####################################################################################################
####################################################################################################

#===================================================================================================
####################################################################################################
new_subtask('Task 2 - CE1 ')
####################################################################################################
#===================================================================================================

# -------------------------------------------------
# Isoparametric coordinates
# -------------------------------------------------
xi1, xi2 = sp.symbols('xi1 xi2', real=True)

# -------------------------------------------------
# Node positions (vectors)
# -------------------------------------------------
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

# -------------------------------------------------
# Mapping and Jacobian
# -------------------------------------------------
x = N1*xe1 + N2*xe2 + N3*xe3
J = x.jacobian(xi)
J_inv_T = J.inv().T

# Partial derivatives of shape function, wrt global coords
dN1_dx = sp.simplify(J_inv_T * dN1_dxi)
dN2_dx = sp.simplify(J_inv_T * dN2_dxi)
dN3_dx = sp.simplify(J_inv_T * dN3_dxi)

# -------------------------------------------------
# B-matrix
# -------------------------------------------------
Be = sp.Matrix([
[dN1_dx[0], 0, dN2_dx[0], 0, dN3_dx[0], 0],
[0, dN1_dx[1], 0, dN2_dx[1], 0, dN3_dx[1]],
[dN1_dx[1], dN1_dx[0], dN2_dx[1], dN2_dx[0], dN3_dx[1], dN3_dx[0]]
])

Ae = sp.simplify(0.5 * J.det())

# -------------------------------------------------
# Callable functions
# -------------------------------------------------
Be_func_cst = sp.lambdify((xe1, xe2, xe3), Be, modules="numpy")
Ae_func = sp.lambdify((xe1, xe2, xe3), Ae, modules="numpy")

# Precompute sparse pattern once to save computation time during assembly
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

# To save computation time, we precompute Be and Ae matrices for all elements and gauss points
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

# This assembling routine uses precomputed COO pattern and precomputed Be and Ae matrices
def assemble_K_fint_coo(a, Edof, rows, cols, ndof, nel, Ex, Ey, D, body, thickness,my_element):
    
    Be_matrix, Ae_matrix = create_Be_Ae_matrix(nel, Ex, Ey)
    
    # ------------------------------------------------------------------
    # Initialize global internal force vector
    # ------------------------------------------------------------------
    f_ext = np.zeros(ndof, dtype=float)

    # Number of DOFs per element (e.g. 12 for tri6 with 2 DOF/node)
    ndofe = Edof.shape[1]-1

    # Each element contributes a dense (ndofe x ndofe) stiffness block
    nnz_per_el = ndofe * ndofe
    nnz_total = nel * nnz_per_el
    # ------------------------------------------------------------------
    # Preallocate COO triplet arrays (row, col, value)
    # ------------------------------------------------------------------
    data = np.empty(nnz_total, dtype=float)
    # Pointer into the preallocated triplet arrays
    p = 0

    # ------------------------------------------------------------------
    # Element loop
    # ------------------------------------------------------------------
    for el in range(nel):
        # Element DOF indices
        edof = Edof[el, 1:].astype(np.int64) -1
        # Compute element internal force and stiffness matrix
        ae = a[edof]
        fe, Ke, *_ = my_element(
            ae, el, Be_matrix, Ae_matrix, D, body, thickness) #Ex, Ey, D, body,  thickness
        
        # Assemble internal force contributions
        f_ext[edof] += fe

        # ------------------------------------------------------------------
        # Assemble stiffness using COO triplets
        # ------------------------------------------------------------------
        data[p:p + nnz_per_el] = Ke.ravel()
        p += nnz_per_el

    # ------------------------------------------------------------------
    # Build global stiffness matrix and convert to CSR for efficient slicing/solves
    # ------------------------------------------------------------------
    K = coo_matrix((data, (rows, cols)), shape=(ndof, ndof)).tocsr()

    # Sum duplicate entries (multiple elements contribute to same (i,j) location)
    K.sum_duplicates()

    return K, f_ext

def cst_element(ae, el, Be_matrix, Ae_matrix, D, body, h): # Ex, Ey, D, body, h):
    ngp=1; fe=np.zeros(6); Ke=np.zeros((6,6))
    #x1 = np.array([Ex[el,0], Ey[el,0]])
    #x2 = np.array([Ex[el,1], Ey[el,1]])
    #x3 = np.array([Ex[el,2], Ey[el,2]])

    for gp in range(ngp):
        Ae = Ae_matrix[el, gp]
        Be = Be_matrix[el, gp, :, :]  # Be_func_cst(x1, x2, x3)
        fe = np.tile(body[el], 3) * Ae / 3
        Ke = Be.T @ D @ Be * Ae * h  
             
    return fe, Ke

# Edge load contribution function
def fe_edge(coords1, coords2, p, h=1):
    
    vec = coords2 - coords1
    n = np.array([vec[1], -vec[0]]).T
    n_unit = n / np.linalg.norm(n)
    
    tn = -p * n_unit
    Le = np.linalg.norm(vec)
    
    force = tn * h * Le / 2
    fe = np.array([force[0], force[1], force[0], force[1]])
    return fe


#%%
#===================================================================================================
####################################################################################################
new_subtask('Task 2 a) - Extension of CE1')
####################################################################################################
#===================================================================================================

# ------------------------------------------------------------------
# Define inputs
# ------------------------------------------------------------------
E_val = 20e6 # Pa
nu_val = 0.45 

# ------------------------------------------------------------------
# Constitutive matrix
# ------------------------------------------------------------------
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

from matplotlib.collections import PolyCollection

polygons = np.zeros((nelem, 3, 2))
for el in range(nelem):
    polygons[el,:,:] = [[Ex[el,0], Ey[el,0]], 
                        [Ex[el,1], Ey[el,1]], 
                        [Ex[el,2], Ey[el,2]]]
    
# ------------------------------------------------------------------
# Plot undeformed mesh
# ------------------------------------------------------------------
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
# Simulation parameters
# ------------------------------------------------------------------
n_steps = 10 # Number of time steps
tol = 1e-6 # Newton tolerance
max_iter = 10 # Max iterations
thickness = h_val # Thickness h
body = np.zeros(nelem) # No body force

# ------------------------------------------------------------------
# Store results
# ------------------------------------------------------------------
disp_history = []
force_history = []
u_vals = []

# ------------------------------------------------------------------
# Target displacement
# ------------------------------------------------------------------
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
    print(f"Step {step}/{n_steps}, disp: {current_u_gamma:.4e} m")
    
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
            print(f'Converged in {i} iterations, residual: {res_norm:.2e}')
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

# -------------------------------------------------
# Plot deformed mesh
# -------------------------------------------------
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

# -------------------------------------------------
# Plot stress
# -------------------------------------------------
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
    array=Es[:,0], # values used for coloring 
    cmap='turbo', 
    edgecolors='k')
ax3.add_collection(pc3)
ax3.autoscale()
ax3.set_title("sigma xx")
fig2.colorbar(pc3, ax=ax3)

# title = 'vertical reaction force vs uΓ'
# plt.figure()
# plt.plot(..., ...)
# plt.title(title)
# plt.xlabel('uΓ [m]')
# plt.ylabel('vertical reaction force [N]')
# plt.show()
# sfig(title)

#%%
#===================================================================================================
####################################################################################################
new_subtask('Task 2 b) - Yeoh hyperelsatic material model')
####################################################################################################
#===================================================================================================

F_11_min = 0.5
F_11_max = 1.5

G_val = E_val / (2 * (1 + nu_val))
lam_val = E_val * nu_val / ((1 + nu_val) * (1 - 2 * nu_val))
c10 = G_val / 2
c20 = - G_val / 10
c30 = G_val / 30
D1 = 0.02 # 1 / MPa
D2 = 0.01
D3 = 0.01

def U0(C, J):
    ans = c10 * (J**(-2/3) * np.trace(C) -3) 
    + c20 * (J**(-2/3) * np.trace(C) -3)**2 
    + c30 * (J**(-2/3) * np.trace(C) -3)**3 
    + (1 / D1) * (J -1)**2 
    + (1 / D2) * (J -1)**4 
    + (1 / D3) * (J - 1)**6
    return ans

# Translation of matlab code in section 5.2.10:
def generate_deformation_gradient_functions():
    # Fv = sym(Fv,[4,1], real)
    Fv = sp.Matrix(sp.symbols('Fv0:4', real=True)) # Fv0, Fv1, Fv2, Fv3
    
    Gmod, lam = sp.symbols('Gmod lambda', real=True)

    # MATLAB: F=[Fv(1) Fv(3) 0; Fv(4) Fv(2) 0; 0 0 1];
    # Mapping indices: 1->0, 2->1, 3->2, 4->3
    F = sp.Matrix([
        [Fv[0], Fv[2], 0],
        [Fv[3], Fv[1], 0],
        [0,     0,     1]
    ])

    # MATLAB: C=F*F (Note: Standard mechanics is usually F.T * F, but translating literally)
    C = F * F 
    
    # MATLAB: invC=simplify(inv(C))
    invC = sp.simplify(C.inv())
    
    # MATLAB: J=det(F)
    J = F.det()

    # MATLAB: S=Gmod*( eye(3)-invC)+lambda*log(J)*invC;
    S = Gmod * (sp.eye(3) - invC) + lam * sp.log(J) * invC
    
    # MATLAB: P=F*S
    P = F * S

    # MATLAB: Pv=[P(1,1) P(2,2) P(1,2) P(2,1)]
    # Python indices: (0,0), (1,1), (0,1), (1,0)
    Pv = sp.Matrix([P[0,0], P[1,1], P[0,1], P[1,0]])

    P_NH_func = sp.lambdify((Fv, Gmod, lam), Pv, modules='numpy')

    # MATLAB: dPvdFv=sym(dPvFv,[4,4], real )
    # Loop i=1:4 ... gradient(Pv(i),Fv)
    
    # In SymPy, we can calculate the Jacobian matrix directly without a loop
    dPvdFv = Pv.jacobian(Fv)

    dPdF_NH_func = sp.lambdify((Fv, Gmod, lam), dPvdFv, modules='numpy')

    return P_NH_func, dPdF_NH_func

# --- Usage Example ---
# calc_P, calc_dPdF = generate_deformation_gradient_functions()

# Fv_val = np.array([1.1, 1.1, 0.1, 0.1]) 
# G_val = 100.0
# lam_val = 500.0

# P_result = calc_P(Fv_val, G_val, lam_val)
# print("\nCalculated P vector (Voigt):")
# print(P_result)

# K_result = calc_dPdF(Fv_val, G_val, lam_val)
# print("\nCalculated Tangent Stiffness Matrix:")
# print(K_result)



# Matlab example 8
# Xe1n=[0 0];Xe2n=[40];Xe3n=[03];
# Be0n=Be0_cst_largedef_func(Xe1n,Xe2n,Xe3n);
# ae=[4+3/2 3/2 3/2+3 3 4+3/2+3/2 3/2];
# F2d=[1 1 0 0]+Be0n*ae;
# %material model
# mu=3; lambda=2;
# Pn=P_NH_func(F2d,mu,lambda)
# =-5.1291e+00-2.5343e+00
# 4.2672e+00
# 4.8145e+00
# dPdFn=dPdF_NH_func(F2d,mu,lambda)
# 2.2439e+01 7.2004e+00-7.2898e+00-9.7197e+00
# 7.2004e+00 1.3935e+01-5.4673e+00-7.2898e+00-7.2898e+00-5.4673e+00 5.7337e+00 1.1024e+01-9.7197e+00-7.2898e+00 1.1024e+01 7.8598e+00
# %element area
# Ae=Ae_cst_func(Xe1n,Xe2n,Xe3n);
# %element thickness
# h0=1;
# %node forces due to deformation
# fe_int=Be0n*Pn*Ae*h0-8.4070e-01-2.1532e+00-7.6936e+00
# 7.2218e+00
# 8.5343e+00-5.0686e+00
# %corresponding stiffness
# Ke_int=Be0n*dPdFn*Be0n*Ae*h0
# 4.9474e+00 1.8224e+00-4.7699e+00-1.8671e+00-1.7756e-01 4.4660e-02
# 1.8224e+00 4.9474e+00 4.4660e-02 6.9744e-01-1.8671e+00-5.6449e+00-4.7699e+00 4.4660e-02 8.4148e+00-3.6449e+00-3.6449e+00 3.6002e+00-1.8671e+00 6.9744e-01-3.6449e+00 2.9474e+00 5.5120e+00-3.6449e+00-1.7756e-01-1.8671e+00-3.6449e+00 5.5120e+00 3.8224e+00-3.6449e+00
# 4.4660e-02-5.6449e+00 3.6002e+00-3.6449e+00-3.6449e+00 9.2898e+00

def generate_yeoh_functions():
    # -------------------------------------------------
    # Symbol definitions
    # -------------------------------------------------
    # Deformation gradient components (column-major vector form F11, F21, F12, F22)
    Fv = sp.Matrix(sp.symbols('Fv0:4', real=True)) 
    
    # Reconstruct 2x2 F matrix
    F = sp.Matrix([[Fv[0], Fv[2]], 
                   [Fv[1], Fv[3]]])
    
    # Large deformation kinematics
    C = F.T * F
    J = F.det()
    
    # -------------------------------------------------
    # Yeoh Strain Energy Potential U0(C, J)
    # -------------------------------------------------
    # Invariants for compressible Yeoh model
    # I1_bar = J^(-2/3) * tr(C)
    I1_bar = J**(-sp.Rational(2, 3)) * sp.trace(C)
    
    # Deviatoric part
    W_dev = c10 * (I1_bar - 3) + \
            c20 * (I1_bar - 3)**2 + \
            c30 * (I1_bar - 3)**3
            
    # Volumetric part
    W_vol = (1/D1) * (J - 1)**2 + \
            (1/D2) * (J - 1)**4 + \
            (1/D3) * (J - 1)**6
            
    U0_val = W_dev + W_vol

    # -------------------------------------------------
    # Derivatives
    # -------------------------------------------------
    # First Piola-Kirchhoff stress P = dU0/dF
    P = sp.diff(U0_val, Fv)
    
    # Tangent Stiffness A = dP/dF (4x4 matrix in Voigt notation)
    A = P.jacobian(Fv)

    # -------------------------------------------------
    # Lambdify for numerical efficiency
    # -------------------------------------------------
    print("Generating Yeoh numerical functions...")
    P_func = sp.lambdify((Fv), P, modules="numpy")
    A_func = sp.lambdify((Fv, c10, c20, c30, D1, D2, D3), A, modules="numpy")
    
    return P_func, A_func


def generate_neohooke_functions():
    """
    Generates numerical functions for the Neo-Hooke material model 
    using SymPy differentiation.
    """
    import sympy as sp
    
    # -------------------------------------------------
    # Symbol definitions
    # -------------------------------------------------
    # Deformation gradient components (column-major F11, F21, F12, F22)
    Fv = sp.Matrix(sp.symbols('Fv0:4', real=True)) 
    # Material parameters
    mu, lam = sp.symbols('mu lam', real=True)
    
    # Reconstruct 2x2 F matrix
    F = sp.Matrix([[Fv[0], Fv[2]], 
                   [Fv[1], Fv[3]]])
    
    # Kinematics
    C = F.T * F
    J = F.det()
    I1 = sp.trace(C)
    
    # -------------------------------------------------
    # Strain Energy Potential U0(C, J)
    # -------------------------------------------------
    # Standard Compressible Neo-Hooke
    U0 = (mu / 2) * (I1 - 3) - mu * sp.log(J) + (lam / 2) * (sp.log(J))**2
    
    # -------------------------------------------------
    # Derivatives
    # -------------------------------------------------
    # First Piola-Kirchhoff stress P = dU0/dF
    P = sp.diff(U0, Fv)
    
    # Material Tangent Stiffness A = dP/dF (4x4 matrix)
    A = P.jacobian(Fv)

    # -------------------------------------------------
    # Lambdify
    # -------------------------------------------------
    print("Generating Neo-Hooke numerical functions...")
    P_func = sp.lambdify((Fv, mu, lam), P, modules="numpy")
    A_func = sp.lambdify((Fv, mu, lam), A, modules="numpy")
    
    return P_func, A_func


# Generate the functions once
P_Yeoh_func, A_Yeoh_func = generate_yeoh_functions()

# Create the functions globally
P_Neo_func, A_Neo_func = generate_neohooke_functions()

printt('REFERENCE RESULTS, FOR VALIDATION:')
print('Neo-Hooke: Cauchy stress sigma11 for F11 = 1.5 is ', 2.2525e01, ' MPa')
print('Yeoh: Cauchy stress sigma11 for F11 = 1.5 is ',1.2142e02, ' MPa')


# title = 'Cauchy stress component σ11 vs F11 - pure elongation'
# plt.figure()
# plt.plot(..., ...)
# plt.title(title)
# plt.xlabel('uΓ [m]')
# plt.ylabel('total vertical [m]')
# plt.show()
# sfig(title)

# title = 'Cauchy stress component σ11 vs F11 - pure contraction'
# plt.figure()
# plt.plot(..., ...)
# plt.title(title)
# plt.xlabel('uΓ [m]')
# plt.ylabel('total vertical [m]')
# plt.show()
# sfig(title)


#%%
#===================================================================================================
####################################################################################################
new_subtask('Task 2 c) - Own element function')
####################################################################################################
#===================================================================================================


def deformation_gradient_2d(ae, Ex_el, Ey_el, xi, eta):
    """Computes F at a specific point (xi, eta) in a T6 element."""
    ue = ae.reshape(6, 2).T
    X_nodes = np.vstack([Ex_el, Ey_el])
    x_nodes = X_nodes + ue
    
    # Derivatives dN/dxi (Same as in t6_element above)
    dN_dxi = np.array([
            [4*xi + 4*eta - 3,  4*xi + 4*eta - 3],
            [4*xi - 1,          0               ],
            [0,                 4*eta - 1       ],
            [4 - 8*xi - 4*eta, -4*xi            ],
            [4*eta,             4*xi            ],
            [-4*eta,            4 - 4*xi - 8*eta]
    ]).T
    
    J_mat = X_nodes @ dN_dxi.T
    detJ = J_mat[0,0]*J_mat[1,1] - J_mat[0,1]*J_mat[1,0]
    J_inv = np.array([[J_mat[1,1], -J_mat[0,1]], [-J_mat[1,0], J_mat[0,0]]]) / detJ
    dN_dX = J_inv @ dN_dxi
    
    F = x_nodes @ dN_dX.T
    return F

def Piola_Kirchoff_2d():
    pass

def fe_int():
    pass

def Ke_int():
    pass

printt('REFERENCE RESULTS, FOR VALIDATION:')
print('See problem formulation')


#%%
#===================================================================================================
####################################################################################################
new_subtask('Task 2 d) - Combine routines')
####################################################################################################
#===================================================================================================

# title = 'total vertical on the upper boundary of the rubber profile vs uΓ'
# plt.figure()
# plt.plot(..., ...)
# plt.title(title)
# plt.xlabel('uΓ [m]')
# plt.ylabel('total vertical [m]')
# plt.show()
# sfig(title)

# title = 'von Mises equivalent stress - coarse mesh'
# plt.figure()
# plt.plot(..., ...)
# plt.title(title)
# plt.xlabel('uΓ [m]')
# plt.ylabel('total vertical [m]')
# plt.show()
# sfig(title)

# title = 'von Mises equivalent stress - fine mesh'
# plt.figure()
# plt.plot(..., ...)
# plt.title(title)
# plt.xlabel('uΓ [m]')
# plt.ylabel('total vertical [m]')
# plt.show()
# sfig(title)

#%%