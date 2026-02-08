#%%
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
#===================================================================================================
new_subtask('Task 2 - CE1 ')
#===================================================================================================
#===================================================================================================


#%%
#===================================================================================================
#===================================================================================================
new_subtask('Task 2 a) - Extension of CE1')
#===================================================================================================
#===================================================================================================

# Define inputs
E_val = 20e6 # Pa
nu_val = 0.45 

# From read_matfiles-1.py
def read_toplogy_from_mat_file(filename):
    mat_file = sio.loadmat(filename)
    edof_x=mat_file['Edof'] #note that edof matlab contains one column too much
    edof=edof_x[:,1:13]
    ex=mat_file['Ex']
    ey=mat_file['Ey']
    dof_upper=mat_file['dof_upper']
    dof_lower=mat_file['dof_lower']
    # height=mat_file['height']
    ndofs=(mat_file['ndofs']).item()
    nelem=mat_file['nelem'].item()
    nnodes=mat_file['nnodes'].item()
    return ex,ey,edof,dof_upper,dof_lower,ndofs,nelem,nnodes

#example 
filename='topology_coarse_6node.mat'
Ex,Ey,Edof,dof_upper,dof_lower,ndofs,nelem,nnodes=read_toplogy_from_mat_file(filename)

from matplotlib.collections import PolyCollection

polygons = np.zeros((nelem, 3, 2))
for el in range(nelem):
    polygons[el,:,:] = [[Ex[el,0],Ey[el,0]], [Ex[el,1],Ey[el,1]], [Ex[el,2],Ey[el,2]]]
    

fig1, ax1 = plt.subplots()
pc1 = PolyCollection(
    polygons,
    facecolors='none',
    edgecolors='k'
)
ax1.add_collection(pc1)
ax1.autoscale()


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
#===================================================================================================
new_subtask('Task 2 b) - Yeoh hyperelsatic material model')
#===================================================================================================
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


printt('REFERENCE RESULTS, FOR VALIDATION:')
print('Neo-Hooke: Cauchy stress sigma11 for F11 = 1.5 is ', 2.2525e01, ' MPa')
print('Yeoh: Cauchy stress sigma11 for F11 = 1.5 is ',1.2142e02, ' MPa')


title = 'Cauchy stress component σ11 vs F11 - pure elongation'
plt.figure()
plt.plot(..., ...)
plt.title(title)
plt.xlabel('uΓ [m]')
plt.ylabel('total vertical [m]')
plt.show()
sfig(title)

title = 'Cauchy stress component σ11 vs F11 - pure contraction'
plt.figure()
plt.plot(..., ...)
plt.title(title)
plt.xlabel('uΓ [m]')
plt.ylabel('total vertical [m]')
plt.show()
sfig(title)


#%%
#===================================================================================================
#===================================================================================================
new_subtask('Task 2 c) - Own element function')
#===================================================================================================
#===================================================================================================


def deformation_gradient_2d():
    pass

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
#===================================================================================================
new_subtask('Task 2 d) - Combine routines')
#===================================================================================================
#===================================================================================================

title = 'total vertical on the upper boundary of the rubber profile vs uΓ'
plt.figure()
plt.plot(..., ...)
plt.title(title)
plt.xlabel('uΓ [m]')
plt.ylabel('total vertical [m]')
plt.show()
sfig(title)

title = 'von Mises equivalent stress - coarse mesh'
plt.figure()
plt.plot(..., ...)
plt.title(title)
plt.xlabel('uΓ [m]')
plt.ylabel('total vertical [m]')
plt.show()
sfig(title)

title = 'von Mises equivalent stress - fine mesh'
plt.figure()
plt.plot(..., ...)
plt.title(title)
plt.xlabel('uΓ [m]')
plt.ylabel('total vertical [m]')
plt.show()
sfig(title)

#%%