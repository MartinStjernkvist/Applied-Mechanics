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

# xi, eta = sp.symbols('xi, eta')
# xe1, xe2, xe3, ye1, ye2, ye3 = sp.symbols('xe1, xe2, xe3, ye1, ye2, ye3')

# xe = sp.Matrix([xe1, xe2, xe3])
# ye = sp.Matrix([ye1, ye2, ye3])

# N1 = xi
# N2 = eta
# N3 = 1 - xi - eta

# Ne_bar = sp.Matrix([N1, N2, N3]).T
# displayvar('Nbar', Ne_bar)

# x = Ne_bar @ xe
# y = Ne_bar @ ye 

# J = sp.Matrix([x, y]).jacobian(sp.Matrix([xi, eta]))
# displayvar('J', J)

# dNbar_dxi = sp.diff(Ne_bar, xi)
# dNbar_deta = sp.diff(Ne_bar, eta)

# # dNbar_dxideta = sp.Matrix([dNbar_dxi, dNbar_deta])
# dNbar_dxideta = dNbar_dxi.col_join(dNbar_deta)
# displayvar('dNbar_dxideta', dNbar_dxideta)

# dNbar_dxdy = (J.T).inv() @ dNbar_dxideta 
# displayvar('dNbar_dxdy', dNbar_dxdy)
# print(sp.shape(dNbar_dxdy))

# Be = sp.zeros(3, 6)
# Be[0, 0::2] = dNbar_dxdy[0, :]
# Be[1, 1::2] = dNbar_dxdy[1, :]
# Be[2, 0::2] = dNbar_dxdy[1, :]
# Be[2, 1::2] = dNbar_dxdy[0, :]
# displayvar('B_e', Be)

# calc_Be = sp.lambdify((xi, eta, xe1, xe2, xe3, ye1, ye2, ye3), Be, "numpy")

xi1, xi2 = sp.symbols('xi1 xi2', real=True)
xi = sp.Matrix([xi1, xi2])

N1 = 1 - xi1 - xi2
N2 = xi1
N3 = xi2

dN1_dxi = sp.Matrix([sp.diff(N1, xi1), sp.diff(N1, xi2)])
dN2_dxi = sp.Matrix([sp.diff(N2, xi1), sp.diff(N2, xi2)])
dN3_dxi = sp.Matrix([sp.diff(N3, xi1), sp.diff(N3, xi2)])

xe1_1, xe1_2 = sp.symbols('xe1_1 xe1_2', real=True)
xe2_1, xe2_2 = sp.symbols('xe2_1 xe2_2', real=True)
xe3_1, xe3_2 = sp.symbols('xe3_1 xe3_2', real=True)
xe1 = sp.Matrix([xe1_1, xe1_2])
xe2 = sp.Matrix([xe2_1, xe2_2])
xe3 = sp.Matrix([xe3_1, xe3_2])

x = N1*xe1 + N2*xe2 + N3*xe3
Fisop = x.jacobian(xi)
Fisop_inv_T = Fisop.inv().T

dN1_dx = sp.simplify(Fisop_inv_T * dN1_dxi)
dN2_dx = sp.simplify(Fisop_inv_T * dN2_dxi)
dN3_dx = sp.simplify(Fisop_inv_T * dN3_dxi)

Be = sp.Matrix([
[dN1_dx[0], 0, dN2_dx[0], 0, dN3_dx[0], 0],
[0, dN1_dx[1], 0, dN2_dx[1], 0, dN3_dx[1]],
[dN1_dx[1], dN1_dx[0], dN2_dx[1], dN2_dx[0], dN3_dx[1], dN3_dx[0]]
])

Be_func_cst = sp.lambdify((xe1, xe2, xe3), Be, modules="numpy")

Ae = sp.simplify(0.5 * Fisop.det())

Ae_func = sp.lambdify((xe1, xe2, xe3), Ae, modules="numpy")

def cst(ex, ey, D, h=1):
    x1, x2, x3 = ex
    y1, y2, y3 = ey
    
    Be_val = Be_func_cst(coords1, coords2, coords3)
    Ae_val = Ae_func(coords1, coords2, coords3)

    h = l_L
    Ke = Be_val.T @ D @ Be_val * h * Ae_val

    nodal_force = Ae_val * l_L * b / 3

    fe = np.zeros((6, 1))
    fe[1::2, :] = nodal_force 
    
    return Ke, fe
    
coords1 = [0.00, 0.00]
coords2 = [1.00, 0.25]
coords3 = [0.50, 1.00]

ex = [coords1[0], coords2[0], coords3[0]]
ey = [coords1[1], coords2[1], coords3[1]]

Ke, fe = cst(ex, ey, D)

displayvar('K^e', Ke, accuracy=3)
displayvar('f^e', fe, accuracy=3)

#%%
#===================================================================================================
new_subtask('Task 1 - c) Constant strain element')
#===================================================================================================

# def fe_edge(coords1, coords2, t, h=1):
    
#     vec = coords2 - coords1
#     n = np.array([vec[1], -vec[0]]).T
#     n_unit = n / np.linalg.norm(n)
#     tn = np.dot(n_unit, t) * n_unit
#     Le = np.linalg.norm(vec)
#     fe = tn * h * Le / 2
#     return fe

def fe_edge(coords1, coords2, p, h=1):
    
    vec = coords2 - coords1
    n = np.array([vec[1], -vec[0]]).T
    n_unit = n / np.linalg.norm(n)
    tn = -p * n_unit
    Le = np.linalg.norm(vec)
    fe = tn * h * Le / 2
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

p = l_Z * rho_w * g

Nr, Nt = 10, 10
elemtype = 1

output = TunnelMeshGen(l_H, l_B, l_D, l_b, l_h, l_r, Nr, Nt, elemtype)
Edof, Coord, Ex, Ey, LeftSide_nodes, TopSide_nodes, RightSide_nodes, BottomSide_nodes = output


# Example. Plot polygons without colouring (e.g. undeformed and deformed mesh)
fig1, ax1 = plt.subplots()

pc1 = PolyCollection(
    polygons,
    facecolors='none',
    edgecolors='k'
)

ax1.add_collection(pc1)
ax1.autoscale()
ax1.set_title("Undeformed mesh")
fig1.colorbar(pc1, ax=ax1)



num_dofs = 2 * Coord.shape[0]
num_el = Edof.shape[0]
num_ed_right = len(RightSide_nodes) - 1
num_ed_top = len(TopSide_nodes) - 1

print(TopSide_nodes)

K = np.zeros((num_dofs, num_dofs))
f = np.zeros((num_dofs))

for el in range(num_el):
    ex = Ex[el, :]
    ey = Ey[el, :]
    
    Ke, fe = cst(ex, ey, D)
    
    dofs = Edof[el, :] - 1
    
    for row in range(6):
        for column in range(6):
            K[dofs[row], dofs[column]] += Ke[row, column]
    
    for row in range(dofs):
        f[dofs[row]] += fe[row]

# Right edge contributions
for ed in range(num_ed_right):
    n1 = RightSide_nodes[ed]
    n2 = RightSide_nodes[ed + 1]
    
    dofs = np.array([2 * n1, 2 * n1 + 1, 2 * n2, 2 * n2 + 1])

    coords1 = Coord[n1]
    coords2 = Coord[n2]
    
    fe = fe_edge(coords1, coords2, p)
    
    f[dofs] += fe

# Top edge contributions
for ed in range(num_ed_top):
    n1 = TopSide_nodes[ed]
    n2 = TopSide_nodes[ed + 1]
    
    dofs = np.array([2 * n1, 2 * n1 + 1, 2 * n2, 2 * n2 + 1])

    coords1 = Coord[n1]
    coords2 = Coord[n2]
    
    fe = fe_edge(coords1, coords2, p)
    
    f[dofs] += fe

bc_dofs = []
bc_vals = []




#%%