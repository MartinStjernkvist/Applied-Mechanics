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

from matplotlib.collections import PolyCollection
from scipy.sparse import lil_matrix

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
    fig_output_file = script_dir / "figures_v2" / fig_name
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
####################################################################################################
####################################################################################################

new_task('Task 1 - Linear elastic plate analysis using MATLAB or Python')

####################################################################################################
####################################################################################################
####################################################################################################
#===================================================================================================
        
# ------------------------------------------------------------------
# Define inputs
# ------------------------------------------------------------------

# Geometry
h_snow = 0.25 # m
angle_roof = np.deg2rad(15) # rad
W_roof = 2 # m
L_roof = 3 # m

# Material properties aluminium
Emod_al = 80e9 # Pa
nu_al = 0.2
sigma_yield_al = 200e6 # Pa
alpha_al = 20e-6 # 1/K  

rho_snow = 500 # kmg/m^3

g = 9.81

q0 = rho_snow * g * h_snow
q_bar  = q0 * np.cos(angle_roof)**2
fx_bar = q0 * np.cos(angle_roof) * np.sin(angle_roof)

h_plate = 0.003
print(f'\nPlate thickness: {h_plate*1e3:.1f} mm')
        
#===================================================================================================
####################################################################################################
new_subtask('Task 1 d) - Code implementation')
####################################################################################################
#===================================================================================================

from kirchoff_funcs import bast_kirchoff_func
from N_kirchoff_func import N_kirchoff_func

def detFisop_4node_func(in1, in2, in3, in4, in5):
    # in1 = xi = location of integration point (vector with 2 components)    
    # in2 = xe1 = nodal coords of node 1 (vector with 2 components)    
    # in3 = xe2 = nodal coords of node 2 (vector with 2 components)    
    # in4 = xe3 = nodal coords of node 3 (vector with 2 components)
    # in5 = xe4 = nodal coords of node 4 (vector with 2 components)
    
    # Ensure inputs are 2D column vectors so [0, :] slicing works safely
    in1 = np.atleast_2d(in1).reshape(2, -1)
    in2 = np.atleast_2d(in2).reshape(2, -1)
    in3 = np.atleast_2d(in3).reshape(2, -1)
    in4 = np.atleast_2d(in4).reshape(2, -1)
    in5 = np.atleast_2d(in5).reshape(2, -1)

    xe11 = in2[0, :]
    xe12 = in2[1, :]
    xe21 = in3[0, :]
    xe22 = in3[1, :]
    xe31 = in4[0, :]
    xe32 = in4[1, :]
    xe41 = in5[0, :]
    xe42 = in5[1, :]
    xi1 = in1[0, :]
    xi2 = in1[1, :]

    # Derivatives of 4-node isoparametric shape functions w.r.t local coords (xi1, xi2)
    dN1_dxi1 = -0.25 * (1.0 - xi2)
    dN2_dxi1 =  0.25 * (1.0 - xi2)
    dN3_dxi1 =  0.25 * (1.0 + xi2)
    dN4_dxi1 = -0.25 * (1.0 + xi2)
    
    dN1_dxi2 = -0.25 * (1.0 - xi1)
    dN2_dxi2 = -0.25 * (1.0 + xi1)
    dN3_dxi2 =  0.25 * (1.0 + xi1)
    dN4_dxi2 =  0.25 * (1.0 - xi1)
    
    # Jacobian matrix components
    dx_dxi1 = dN1_dxi1 * xe11 + dN2_dxi1 * xe21 + dN3_dxi1 * xe31 + dN4_dxi1 * xe41
    dy_dxi1 = dN1_dxi1 * xe12 + dN2_dxi1 * xe22 + dN3_dxi1 * xe32 + dN4_dxi1 * xe42
    
    dx_dxi2 = dN1_dxi2 * xe11 + dN2_dxi2 * xe21 + dN3_dxi2 * xe31 + dN4_dxi2 * xe41
    dy_dxi2 = dN1_dxi2 * xe12 + dN2_dxi2 * xe22 + dN3_dxi2 * xe32 + dN4_dxi2 * xe42
    
    # Determinant of the Jacobian
    detFisop = dx_dxi1 * dy_dxi2 - dx_dxi2 * dy_dxi1
    
    return detFisop



def Bu_and_detJ(xin, xe_nodes):
    """Extract scalars from xin (handles both 1D and 2D input)."""
    xin_flat = np.atleast_1d(np.squeeze(xin))
    xi1v, xi2v = xin_flat[0], xin_flat[1]
    
    dN_dxi = 0.25 * np.array([
        [-(1.-xi2v),  (1.-xi2v), (1.+xi2v), -(1.+xi2v)],
        [-(1.-xi1v), -(1.+xi1v),(1.+xi1v),  (1.-xi1v)]
    ])                              # (2,4)
    J = dN_dxi @ xe_nodes        # (2,2)
    detJ = float(np.linalg.det(J))
    dN_dx = np.linalg.inv(J) @ dN_dxi  # (2,4)  [row0=dNi/dx, row1=dNi/dy]

    B_u = np.zeros((3, 8))
    for k in range(4):
        B_u[0, 2*k  ] = dN_dx[0, k]    # ∂N/∂x → εxx
        B_u[1, 2*k+1] = dN_dx[1, k]    # ∂N/∂y → εyy
        B_u[2, 2*k  ] = dN_dx[1, k]    # ∂N/∂y → γxy
        B_u[2, 2*k+1] = dN_dx[0, k]    # ∂N/∂x → γxy
        
    return B_u, detJ


def Nu_inplane(xin):
    """Extract scalars from xin (handles both 1D and 2D input)."""
    xin_flat = np.atleast_1d(np.squeeze(xin))
    xi1v, xi2v = xin_flat[0], xin_flat[1]
    
    Nv = [0.25*(1.-xi1v)*(1.-xi2v),
          0.25*(1.+xi1v)*(1.-xi2v),
          0.25*(1.+xi1v)*(1.+xi2v),
          0.25*(1.-xi1v)*(1.+xi2v)]
    N_u = np.zeros((2, 8))
    for k, Nk in enumerate(Nv):
        N_u[0, 2*k  ] = Nk
        N_u[1, 2*k+1] = Nk
        
    return N_u



def hooke_plane_stress(E, nu):
    """Constitutive matrix for plane stress."""
    return (E / (1 - nu**2)) * np.array([
        [1, nu, 0],
        [nu, 1, 0],
        [0, 0, (1 - nu) / 2]
    ])


def assem(edof, K, Ke, f, fe):
    """Assemble element stiffness and force into global matrices."""
    idx = edof - 1 # Convert to 0-based indexing
    for i in range(len(idx)):
        f[idx[i], 0] += fe[i, 0]
        for j in range(len(idx)):
            K[idx[i], idx[j]] += Ke[i, j]
    return K, f


# def coordxtr(Edof, Coord, Dof, nen):
#     """Extract element coordinates."""
#     nel = Edof.shape[0]
#     Ex = np.zeros((nel, nen))
#     Ey = np.zeros((nel, nen))
#     for i in range(nel):
#         # Find the node indices for this element based on Dof matching
#         # (Simplified for this specific structured mesh)
#         node_dofs = Edof[i, 1::3] # Take the first DOF of each node block
#         nodes = [np.where(Dof[:, 0] == dof)[0][0] for dof in node_dofs]
#         Ex[i, :] = Coord[nodes, 0]
#         Ey[i, :] = Coord[nodes, 1]
#     return Ex, Ey
    

def kirchoff_plate_element(ex, ey, h, Dbar, body_val):
    
    # Gauss points
    H_v = np.ones(4)
    xi_v = np.array([
        [-1/np.sqrt(3), -1/np.sqrt(3),  1/np.sqrt(3), 1/np.sqrt(3)],
        [-1/np.sqrt(3),  1/np.sqrt(3), -1/np.sqrt(3), 1/np.sqrt(3)]
    ])
    
    # Ke = np.zeros((12, 12))
    # fe_ext = np.zeros((12, 1))
    K_uu = np.zeros((8,  8))
    K_ww = np.zeros((12, 12))
    f_u  = np.zeros((8,  1))
    f_w  = np.zeros((12, 1))
    
    n1 = np.array([[ex[0]], [ey[0]]])
    n2 = np.array([[ex[1]], [ey[1]]])
    n3 = np.array([[ex[2]], [ey[2]]])
    n4 = np.array([[ex[3]], [ey[3]]])
    
    xe_nodes = np.column_stack((ex, ey))  # (4,2)
    
    for gp in range(4):
        Hgp = H_v[gp]
        
        # Extract as scalar coordinates (not 2D arrays)
        xi_vec = xi_v[:, gp]  # shape (2,) with float scalars
        xin = xi_vec.reshape(2, 1)  # shape (2, 1) for functions that expect 2D
        
        # =============================================================================
        # In-plane membrane
        # =============================================================================
        B_u, detJ = Bu_and_detJ(xi_vec, xe_nodes)
        N_u = Nu_inplane(xi_vec)
        f_body = np.array([[fx_bar], [0.0]])

        K_uu += B_u.T @ (h * D) @ B_u * detJ * Hgp
        f_u  += N_u.T @ f_body * detJ * Hgp
        
        # =============================================================================
        # Bending (Kirchhoff)
        # =============================================================================
        detFisop = detFisop_4node_func(xin, n1, n2, n3, n4)[0] # Extract scalar
        N_w = N_kirchoff_func(xin, n1, n2, n3, n4)
        dNdx, Bastn, _ = bast_kirchoff_func(xin, n1, n2, n3, n4)
        
        # fe_ext += (N.T * body_val * detFisop * Hgp)
        # Ke += (Bastn.T @ Dbar @ Bastn * detFisop * Hgp)
        f_w += (N_w.T * body_val * detFisop * Hgp)
        K_ww += (Bastn.T @ Dbar @ Bastn * detFisop * Hgp)
        
    # return Ke, fe_ext
    return K_uu, K_ww, f_u, f_w



#===================================================================================================
####################################################################################################
new_subtask('Task 1 e) - Small FE-program for plate analysis')
####################################################################################################
#===================================================================================================

xmin = 0
xmax = 0.75
ymin = 0
ymax = 0.50
nelx = 15
nely = 10

"""
xmin = 0;   xmax = 0.75;
ymin = 0;   ymax = 0.50;

nelx = 15;   % elements along x (≈5 cm element size)
nely = 10;   % elements along y (≈5 cm element size)

[mesh, coord, Edof_ip, Edof_oop] = rectMesh(xmin, xmax, ymin, ymax, nelx, nely);

save('testfil','-v7');
"""

data = sio.loadmat('testfil.mat', 
                   squeeze_me=True,
                   struct_as_record=False)

mesh = data['mesh']
Coord = data['coord']
Edof_ip = data['Edof_ip']
Edof_oop = data['Edof_oop']

# Convert from sparse to dense if needed
from scipy.sparse import issparse
if issparse(Edof_ip):
    Edof_ip = Edof_ip.toarray()
if issparse(Edof_oop):
    Edof_oop = Edof_oop.toarray()
Edof_ip = Edof_ip.astype(int)
Edof_oop = Edof_oop.astype(int)

nel = Edof_ip.shape[0]

node_idx = (mesh.T - 1).astype(int)

Ex = Coord[node_idx, 0]
Ey = Coord[node_idx, 1]

polygons = np.dstack((Ex, Ey))

# Plot undeformed mesh
fig1, ax1 = plt.subplots(figsize=(8, 6))
pc1 = PolyCollection(
    polygons,
    facecolors='none',
    edgecolors='k',
    linewidths=1.2
)
ax1.add_collection(pc1)
ax1.autoscale()
ax1.set_aspect('equal')
ax1.set_title("Undeformed mesh")
ax1.set_xlabel("x (m)")
ax1.set_ylabel("y (m)")
plt.grid(True, alpha=0.3)
plt.show()

# =============================================================================
# Initialize sparse global matrices
# =============================================================================
nel = mesh.shape[1]
nnodes = Coord.shape[0]
dofs_per_node = 5
ndofs = dofs_per_node * nnodes

K_global = lil_matrix((ndofs, ndofs)) 
f_global = np.zeros(ndofs)  # 1D array
a = np.zeros(ndofs)  # 1D array


D = hooke_plane_stress(Emod_al, nu_al)
Dbar = (h_plate**3 / 12) * D

# =============================================================================
# Boundary conditions
# =============================================================================
# boundary_nodes = np.where(
#     (np.abs(Coord[:,0] - xmin) < 1e-8) | (np.abs(Coord[:,0] - xmax) < 1e-8) |
#     (np.abs(Coord[:,1] - ymin) < 1e-8) | (np.abs(Coord[:,1] - ymax) < 1e-8)
# )[0]

# fixed = []
# for node in boundary_nodes:
#     for d in range(3):
#         fixed.append(node * 3 + d)
# dof_C = np.array(fixed)

# dof_F = np.setdiff1d(np.arange(ndofs), fixed)
# a_C = np.zeros((len(fixed), 1))

x_all, y_all = Coord[:, 0], Coord[:, 1]
x_min, x_max = x_all.min(), x_all.max()
y_min, y_max = y_all.min(), y_all.max()
tol = 1e-10

nodes_top   = np.where(x_all < x_min + tol)[0]   # ridge (x=0)
nodes_left  = np.where(y_all < y_min + tol)[0]   # left side (y=0)
nodes_right = np.where(y_all > y_max - tol)[0]   # right side (y=ymax)
nodes_bot   = np.where(x_all > x_max - tol)[0]   # gliding (x=xmax)

clamped_nodes = np.unique(np.concatenate([nodes_top, nodes_left, nodes_right]))

prescribed = set()
# Clamped: all 5 DOFs = 0
for n in clamped_nodes:
    for d in range(5):
        prescribed.add(5*n + d)
# Gliding edge: w (DOF+2) and θx (DOF+4)
for n in nodes_bot:
    prescribed.add(5*n + 2)
    prescribed.add(5*n + 4)

dof_C = np.array(sorted(prescribed), dtype=int)
dof_F = np.setdiff1d(np.arange(ndofs), prescribed)
a_C = np.zeros(len(prescribed))  # 1D array

print(f"\nBoundary conditions:")
print(f"  Prescribed DOFs : {len(dof_C )}")
print(f"  Free DOFs       : {len(dof_F)}")

# body = np.ones((nel, 1)) * q
body = q_bar

# =============================================================================
# Setup and solve FE equations
# =============================================================================
for el in range(nel):
    
    # Ke, fe_ext = kirchoff_plate_element(Ex[el, :], Ey[el, :], h_plate, Dbar, body)
    K_uu, K_ww, f_u, f_w = kirchoff_plate_element(Ex[el, :], Ey[el, :], h_plate, Dbar, body)
    # Pass Edof row, excluding the element number at index 0
    
    # 1. Get the 4 nodes for this specific element
    nodes = node_idx[el, :] 
    
    # 2. Map Membrane DOFs (ux, uy)
    d_ip = np.empty(8, dtype=int)
    d_ip[0::2] = 5 * nodes + 0  # ux
    d_ip[1::2] = 5 * nodes + 1  # uy
    
    # 3. Map Bending DOFs (w, theta_y, theta_x)
    # Note: Double check your local element formulation's rotation order. 
    # This matches your extraction logic at the end of the script.
    d_oop = np.empty(12, dtype=int)
    d_oop[0::3] = 5 * nodes + 2  # w
    d_oop[1::3] = 5 * nodes + 3  # theta_y
    d_oop[2::3] = 5 * nodes + 4  # theta_x
    
    # K, f_ext_area = assem(Edof_ip[el, 1:], K, Ke, f_ext_area, fe_ext)

    # K_global[np.ix_(d_ip,  d_ip )] += K_uu
    # K_global[np.ix_(d_oop, d_oop)] += K_ww
    # f_global[d_ip ] += f_u.flatten()
    # f_global[d_oop] += f_w.flatten()
    
    # Assembly for Membrane
    for i in range(8):
        f_global[d_ip[i]] += f_u[i, 0]
        for j in range(8):
            K_global[d_ip[i], d_ip[j]] += K_uu[i, j]
            
    # Assembly for Bending
    for i in range(12):
        f_global[d_oop[i]] += f_w[i, 0]
        for j in range(12):
            K_global[d_oop[i], d_oop[j]] += K_ww[i, j]

# Convert to CSR format
K = K_global.tocsr()

# Block partitioning and solving: K_FF * a_F = f_F - K_FC * a_C
K_FF = K[np.ix_(dof_F, dof_F)]
K_FC = K[np.ix_(dof_F, dof_C)]
K_CF = K[np.ix_(dof_C, dof_F)]
K_CC = K[np.ix_(dof_C, dof_C)]

# =============================================================================
# Solve for free displacements
# =============================================================================
f_F = f_global[dof_F] - K_FC @ a_C
import scipy.sparse.linalg as spla
a_F = spla.spsolve(K_FF, f_F)
if a_F.ndim == 0:  # scalar case
    a_F = np.array([a_F])
a_F = np.atleast_1d(a_F).ravel()  # Ensure 1D

# =============================================================================
# Calculate reaction forces
# =============================================================================
f_extC = K_CF @ a_F + K_CC @ a_C - f_global[dof_C]

# Reconstruct total displacement vector
a[dof_F] = a_F
a[dof_C] = a_C

# =============================================================================
# Extract displacement fields
# =============================================================================
w_nodes  = a[2::5]    # out-of-plane deflection at every node
ux_nodes = a[0::5]
uy_nodes = a[1::5]
ty_nodes = a[3::5]    # θy
tx_nodes = a[4::5]    # θx

w_max = np.abs(w_nodes).max()
print(f"\n{'='*60}")
print(f"  Max |w|  = {w_max*1e3:.3f} mm   (limit: 25 mm)")
if w_max*1e3 <= 25.0:
    print("  Deflection criterion:  SATISFIED ✓")
else:
    print("  Deflection criterion:  VIOLATED  ✗  → increase thickness")
print(f"{'='*60}")


# =============================================================================
# Displacement contour plots
# =============================================================================
def element_average(nodal_field):
    """Compute element-average of a nodal field for patch colouring."""
    return nodal_field[node_idx].mean(axis=1)

polygons = Coord[node_idx]   # (nel, 4, 2)

fig, axes = plt.subplots(3,1, figsize=(8, 16))
fig.suptitle(f"Displacements  (h = {h_plate*1e3:.1f} mm, snow load)", fontsize=13)

configs = [
    (w_nodes*1e3,    'w [mm]',   'RdBu_r'),
    (ux_nodes*1e6,   'ux [μm]',  'viridis'),
    (uy_nodes*1e6,   'uy [μm]',  'viridis'),
]
for ax, (field, label, cmap) in zip(axes, configs):
    el_vals = element_average(field)
    pc = PolyCollection(polygons, array=el_vals,
                        cmap=cmap, edgecolors='none')
    ax.add_collection(pc)
    ax.autoscale()
    ax.set_aspect('equal')
    ax.set_title(label)
    ax.set_xlabel('x [m]  (down slope)')
    ax.set_ylabel('y [m]')
    plt.colorbar(pc, ax=ax)

plt.tight_layout()
plt.savefig('task1_displacements.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: task1_displacements.png")

# =============================================================================
# Stress computation at integration points
# =============================================================================

def von_mises_ps(sigma):
    """Von Mises for plane-stress σ = [s11, s22, s12]."""
    s11, s22, s12 = sigma
    return np.sqrt(s11**2 - s11*s22 + s22**2 + 3.0*s12**2)

_gp   = 1.0/np.sqrt(3.)
_xi_gp = np.array([[-_gp,-_gp], [_gp,-_gp], [_gp,_gp], [-_gp,_gp]])

def element_stress_at_z(ex, ey, a_u_el, a_w_el, h_pl, E, nu, z_coord):
    
    D = hooke_plane_stress(E, nu)
    xe_nodes = np.column_stack((ex, ey))
    n1=np.array([[ex[0]],[ey[0]]])
    n2=np.array([[ex[1]],[ey[1]]])
    n3=np.array([[ex[2]],[ey[2]]])
    n4=np.array([[ex[3]],[ey[3]]])

    sigma_vM = np.zeros(4)
    for i in range(4):
        xi1v, xi2v = _xi_gp[i]
        xin = np.array([[xi1v],[xi2v]])

        # Membrane strain
        B_u, _ = Bu_and_detJ(xi1v, xi2v, xe_nodes)
        eps0   = B_u @ a_u_el                          # (3,)

        # Curvature  κ = Bast · aω  (Bast has shape (3,12))
        _, Bast, _ = bast_kirchoff_func(xin, n1, n2, n3, n4)
        kappa  = Bast @ a_w_el                         # (3,)

        # Total strain at depth z  (Kirchhoff: ε = ε⁰ - z·κ)
        eps_z  = eps0 - z_coord * kappa

        # Stress and von Mises
        sigma = D @ eps_z
        sigma_vM[i] = von_mises_ps(sigma)

    return sigma_vM


# ── Evaluate stresses for all elements ──────────────────────────────────────
print("\nComputing element stresses at integration points …")

z_levels   = [-h_plate/2.0, 0.0, h_plate/2.0]
z_labels   = ['-h/2 (bottom)', 'z=0 (mid-plane)', '+h/2 (top)']
vM_contour = {}             # dict  z_val → (nel,) array of element-average σ_vM

for z in z_levels:
    el_avg_vM = np.zeros(nel)
    for el in range(nel):
        ex = Coord[node_idx[el,:], 0]
        ey = Coord[node_idx[el,:], 1]

        d_ip  = Edof_ip [el, 1:] - 1
        d_oop = Edof_oop[el, 1:] - 1

        a_u_el = a[d_ip ]
        a_w_el = a[d_oop]

        svm = element_stress_at_z(ex, ey, a_u_el, a_w_el,
                                   h_plate, Emod_al, nu_al, z)
        el_avg_vM[el] = svm.mean()

    vM_contour[z] = el_avg_vM

print("  Stress computation complete.")


# ── Von Mises contour plots ──────────────────────────────────────────────────
fig2, axes2 = plt.subplots(1, 3, figsize=(16, 4))
fig2.suptitle(
    f"Von Mises effective stress [MPa]  (h = {h_plate*1e3:.1f} mm, snow load)",
    fontsize=13)

sigma_max_overall = 0.0
for ax, (z, lbl) in zip(axes2, zip(z_levels, z_labels)):
    el_vals = vM_contour[z] / 1e6      # Pa → MPa
    sigma_max_overall = max(sigma_max_overall, el_vals.max())

    pc = PolyCollection(polygons, array=el_vals,
                        cmap='hot_r', edgecolors='none')
    ax.add_collection(pc)
    ax.autoscale()
    ax.set_aspect('equal')
    ax.set_title(f'z = {lbl}')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    plt.colorbar(pc, ax=ax)

plt.tight_layout()
plt.savefig('task1_stress_vonMises.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: task1_stress_vonMises.png")


# =============================================================================
# Load vector verification
# =============================================================================
plate_area = (Coord[:,0].max() - Coord[:,0].min()) * \
             (Coord[:,1].max() - Coord[:,1].min())

w_dof_mask  = np.arange(2, ndofs, 5)
ux_dof_mask = np.arange(0, ndofs, 5)

sum_fw  = f_global[w_dof_mask ].sum()
sum_fux = f_global[ux_dof_mask].sum()

print(f"Sum of nodal w-forces = {sum_fw:.4f} N"
      f"expected {q_bar * plate_area:.4f} N")
print(f"Sum of nodal ux-forces  = {sum_fux:.4f} N"
      f"expected {fx_bar * plate_area:.4f} N")

#%%