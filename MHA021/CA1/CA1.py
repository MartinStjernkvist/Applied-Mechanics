#%%
# %matplotlib widget

from mha021 import *

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

def new_prob(string):
    print_string = '\n' + '=' * 80 + '\n' + 'Task ' + str(string) + '\n' + '=' * 80 + '\n'
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

new_prob('1 - Planar truss')

# Define input data
L = 1.6         # Element length [m]
P = 60e3        # Force [N]
E = 200e9       # Young's modulus [Pa]
A = 1e-3        # Cross-sectional area





#%%
####################################################################################################
####################################################################################################
####################################################################################################



# Task 2 - Planar frame



####################################################################################################
####################################################################################################
####################################################################################################

new_prob('2 - Planar frame')

# Define input data




#%%
####################################################################################################
####################################################################################################
####################################################################################################



# Task 3 - 1D bar with distributed load



####################################################################################################
####################################################################################################
####################################################################################################
new_prob('3 - 1D bar with distributed load')

# Define input data
EA = 210e5      # Axial stiffness [N]
L = 2           # Element length [m]
q0 = 1000       # Distributed load at x = L [N / m]

def q(x):
    return q0 * x / L

#---------------------------------------------------------------------------------------------------
# e)
#---------------------------------------------------------------------------------------------------



#---------------------------------------------------------------------------------------------------
# g)
#---------------------------------------------------------------------------------------------------
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
new_prob('4 - Beam on elastic foundation')

# Define input data
I_y = 30.3 * 10**(-6)   # [m^4]
E = 210 * 10**9         # [Pa]
K_w = 10.05 * 10**(6)   # [N / m^2]
L = 10                  # [m]
P = 70 * 10**3          # [N]

#%%
