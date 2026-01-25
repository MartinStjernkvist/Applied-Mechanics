#%%
from TunnelMeshGen import TunnelMeshGen
from polygon_plot import PolyCollection

import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import math
import pandas as pd
import sys
import os

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



new_task('Task 1')



####################################################################################################
####################################################################################################
####################################################################################################

#===================================================================================================
new_subtask('Task 1 - Problem definition & define inputs')
#===================================================================================================


E = 30e9 # Pa
nu = 0.2 
g = 9.82
rho_w = 1e3 # kg / m^3
rho_c = 2200 # kg / m^3

# dimensions (l_ = length)
l_Z = 50
l_B = 15
l_H = 22
l_D = 8
l_b = 6
l_h = 7
l_r = 2
l_L = 1

b = - rho_c * g
p_w = l_Z * rho_w * g

#%%
#===================================================================================================
new_subtask('Task 1 - b) Constant strain element')
#===================================================================================================

factor = E / ((1 + nu) * (1 - 2 * nu))
    
D = factor * np.array([
        [1.0 - nu,     nu,            0.0],
        [    nu,   1.0 - nu,          0.0],
        [   0.0,       0.0,  (1.0 - 2.0*nu)/2.0]
], dtype=float)

xi1, xi2 = sp.symbols('xi1 xi2', real=True)

xi = sp.Matrix([xi1, xi2])
xe1_1, xe1_2 = sp.symbols('xe1_1 xe1_2', real=True)
xe2_1, xe2_2 = sp.symbols('xe2_1 xe2_2', real=True)
xe3_1, xe3_2 = sp.symbols('xe3_1 xe3_2', real=True)

# Coordinate matrices
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

x = N1*xe1 + N2*xe2 + N3*xe3
J = x.jacobian(xi)
J_inv_T = J.inv().T

dN1_dx = sp.simplify(J_inv_T * dN1_dxi)
dN2_dx = sp.simplify(J_inv_T * dN2_dxi)
dN3_dx = sp.simplify(J_inv_T * dN3_dxi)

Be = sp.Matrix([
[dN1_dx[0], 0, dN2_dx[0], 0, dN3_dx[0], 0],
[0, dN1_dx[1], 0, dN2_dx[1], 0, dN3_dx[1]],
[dN1_dx[1], dN1_dx[0], dN2_dx[1], dN2_dx[0], dN3_dx[1], dN3_dx[0]]
])

Be_func_cst = sp.lambdify((xe1, xe2, xe3), Be, modules="numpy")

Ae = sp.simplify(0.5 * J.det())

Ae_func = sp.lambdify((xe1, xe2, xe3), Ae, modules="numpy")

def cst(coords1, coords2, coords3, D, h=1):
    
    Be_val = Be_func_cst(coords1, coords2, coords3)
    Ae_val = Ae_func(coords1, coords2, coords3)

    Ke = Be_val.T @ D @ Be_val * h * Ae_val

    nodal_force = Ae_val * h * b / 3

    fe = np.zeros((6, 1))
    fe[1::2, :] = nodal_force 
    
    return Ke, fe
    
coords1 = [0.00, 0.00]
coords2 = [1.00, 0.25]
coords3 = [0.50, 1.00]

Ke, fe = cst(coords1, coords2, coords3, D)

displayvar('K^e', np.round(Ke), accuracy=3)
displayvar('f^e', fe, accuracy=3)

#%%
#===================================================================================================
new_subtask('Task 1 - c) Constant strain element')
#===================================================================================================

def fe_edge(coords1, coords2, p, h=1):
    
    vec = coords2 - coords1
    n = np.array([vec[1], -vec[0]]).T
    n_unit = n / np.linalg.norm(n)
    
    tn = -p * n_unit
    Le = np.linalg.norm(vec)
    
    force = tn * h * Le / 2
    fe = np.array([force[0], force[1], force[0], force[1]])
    return fe

coords1 = np.array([0, 0]).T
coords2 = np.array([1, 1]).T
p_val = 5

fe_val = fe_edge(coords1, coords2, p_val)
displayvar('f', fe_val)
#%%
#===================================================================================================
new_subtask('Task 1 - d) solve elasticity problem')
#===================================================================================================

# Constant pressure
p = l_Z * rho_w * g

# Define geometry
Nr, Nt = 10, 20
elemtype = 1

output = TunnelMeshGen(l_H, l_B, l_D, l_b, l_h, l_r, Nr, Nt, elemtype)
Edof, Coord, Ex, Ey, LeftSide_nodes, TopSide_nodes, RightSide_nodes, BottomSide_nodes = output

num_nodes = Coord.shape[0]
num_dofs = 2 * num_nodes
num_el = Edof.shape[0]
num_ed_right = len(RightSide_nodes) - 1
num_ed_top = len(TopSide_nodes) - 1

polygons = np.zeros((num_el, 3, 2))
for el in range(num_el):
    node_ids = (Edof[el, [1, 3, 5]] - 1) // 2
    Ex[el,:] = Coord[node_ids,0]
    Ey[el,:] = Coord[node_ids,1]
    polygons[el,:,:] = [[Ex[el,0],Ey[el,0]], [Ex[el,1],Ey[el,1]], [Ex[el,2],Ey[el,2]]]
    
# Plot undeformed mesh
fig1, ax1 = plt.subplots()
pc1 = PolyCollection(
    polygons,
    facecolors='none',
    edgecolors='k'
)
ax1.add_collection(pc1)
ax1.autoscale()
ax1.set_title("Undeformed mesh")

# Boundary conditions
dof_C = []
a_C = []

# Bottom
for node in BottomSide_nodes:
    n = node - 1
    dof_C.extend([2 * n, 2 * n + 1])
    a_C.extend([0, 0])

# Left
for node in LeftSide_nodes:
    n = node - 1
    dof_C.extend([2 * n])
    a_C.extend([0])

dof_C = np.array(dof_C)
a_C = np.array(a_C)

_, idx = np.unique(dof_C, return_index=True)

dof_C = dof_C[idx]
a_C = a_C[idx]

all_dofs = np.arange(num_dofs)
dof_F = np.setdiff1d(all_dofs, dof_C)

# Initialize stiffness matrix and load vector
K = scipy.sparse.lil_matrix((num_dofs, num_dofs))
f = np.zeros(num_dofs)
a = np.zeros(num_dofs)

for el in range(num_el):
    
    coords1 = np.array([Ex[el,0], Ey[el,0]])
    coords2 = np.array([Ex[el,1], Ey[el,1]])
    coords3 = np.array([Ex[el,2], Ey[el,2]])
    
    Ke, fe = cst(coords1, coords2, coords3, D)
    
    dofs = Edof[el, 1:] - 1
    
    K[np.ix_(dofs, dofs)] += Ke
    
    f[dofs] += fe.flatten()

K = K.tocsr()

# Right edge contributions
for ed in range(num_ed_right):
    n1 = RightSide_nodes[ed] - 1
    n2 = RightSide_nodes[ed + 1] - 1
    
    dofs = np.array([2 * n1, 2 * n1 + 1, 2 * n2, 2 * n2 + 1])
    coords1 = Coord[n1]
    coords2 = Coord[n2]
    
    fe = fe_edge(coords1, coords2, p)
    f[dofs] += fe

# Top edge contributions
for ed in range(num_ed_top):
    n1 = TopSide_nodes[ed] - 1
    n2 = TopSide_nodes[ed + 1] - 1
    
    dofs = np.array([2 * n1, 2 * n1 + 1, 2 * n2, 2 * n2 + 1])
    coords1 = Coord[n1]
    coords2 = Coord[n2]
    
    fe = fe_edge(coords1, coords2, p)
    f[dofs] += fe


a_F = scipy.sparse.linalg.spsolve(
K[np.ix_(dof_F, dof_F)],
f[dof_F] - K[np.ix_(dof_F, dof_C)] @ a_C
)

f_C = (
K[np.ix_(dof_C, dof_F)] @ a_F +
K[np.ix_(dof_C, dof_C)] @ a_C -
f[dof_C]
)

a[dof_F] = a_F
a[dof_C] = a_C

uy = a[1::2]
uy_max = np.max(np.abs(uy))
print('\nmax vertical displacement [mm]:')
displayvar('u_y', uy_max * 1000, accuracy=3)

mag = 2000

polygons = np.zeros((num_el, 3, 2))
for el in range(num_el):
    edofs = Edof[el,1:] - 1
    polygons[el,:,:] = [
        [Ex[el,0] + mag * a[edofs[0]], Ey[el,0] + mag * a[edofs[1]]],
        [Ex[el,1] + mag * a[edofs[2]], Ey[el,1] + mag * a[edofs[3]]],
        [Ex[el,2] + mag * a[edofs[4]], Ey[el,2] + mag * a[edofs[5]]]
    ]

# Plot deformed mesh
fig2, ax2 = plt.subplots()
pc2 = PolyCollection(
    polygons,
    facecolors='none',
    edgecolors='r'
)
ax2.add_collection(pc2)
ax2.autoscale()
ax2.set_title(f"Deformed mesh, magnification = {mag}")

#%%
#===================================================================================================
new_subtask('Task 1 - e) stress plot')
#===================================================================================================

def sigma(coords1, coords2, coords3, a):
    
    Be = Be_func_cst(coords1, coords2, coords3)
    sigma = D @ Be @ a
    
    sigma_xx = sigma[0]
    sigma_yy = sigma[1]
    tau_xy = sigma[2]
    sigma_zz = nu * (sigma_xx + sigma_yy)

    center = (sigma_xx + sigma_yy) / 2
    radius = np.sqrt(((sigma_xx - sigma_yy) / 2)**2 + tau_xy**2)
    s1_in = center + radius
    s2_in = center - radius

    sigma_principal = np.sort([s1_in, s2_in, sigma_zz])
    sigma_complete = np.hstack([sigma, sigma_principal])
    return sigma_complete

a_test = np.array([0, 0, 0.003, 0.001, 0.002, 0.002]).T
coords1 = [0.00, 0.00]
coords2 = [1.00, 0.25]
coords3 = [0.50, 1.00]

sigma_val = sigma(coords1, coords2, coords3, a_test)

displayvar('\sigma', sigma_val, accuracy=4)

Es = np.zeros((num_el, 6))
for el in range(num_el):
    
    coords1 = np.array([Ex[el,0], Ey[el,0]])
    coords2 = np.array([Ex[el,1], Ey[el,1]])
    coords3 = np.array([Ex[el,2], Ey[el,2]])
    
    edofs = Edof[el,1:] - 1
    
    # Be = Be_func_cst(coords1, coords2, coords3)
    # Es[el,:] = D @ Be @ a[edofs]
    
    Es[el, :] = sigma(coords1, coords2, coords3, a[edofs])
    
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

#%%
#===================================================================================================
new_subtask('Task 1 - f) FOS')
#===================================================================================================

# https://www.engineeringtoolbox.com/concrete-properties-d_1223.html
compressive_strength = 30e6 # Pa
tensile_strength = 3.5e6 # Pa

FOS_compression = compressive_strength / np.abs(np.min(Es[:, 3]))
FOS_tension = tensile_strength / np.max(Es[:, -1])

displayvar('FOC_c', FOS_compression, accuracy=3)
displayvar('FOC_t', FOS_tension, accuracy=3)


fig4, ax4 = plt.subplots()
pc4 = PolyCollection(
    polygons,
    array= Es[:,3],
    cmap='turbo',
    edgecolors='k'
)
ax4.add_collection(pc4)
ax4.autoscale()
ax4.set_title("sigma 1")
fig2.colorbar(pc4, ax=ax4)

fig5, ax5 = plt.subplots()
pc5 = PolyCollection(
    polygons,
    array= Es[:,5],
    cmap='turbo',
    edgecolors='k'
)
ax5.add_collection(pc5)
ax5.autoscale()
ax5.set_title("sigma 3")
fig2.colorbar(pc5, ax=ax5)

#%%