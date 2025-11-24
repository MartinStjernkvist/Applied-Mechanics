#%%
# %matplotlib widget

import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import math
import pandas as pd

import sys
import os
# Add parent directory to sys.path
parent_dir = os.path.abspath(os.path.join(os.getcwd(),'..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

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

def new_prob(string):
    print_string = '\n' + '=' * 80 + '\n' + 'Assignment ' + str(string) + '\n' + '=' * 80 + '\n'
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



# Starting point



####################################################################################################
####################################################################################################
####################################################################################################
new_prob('Starting point')

# These functions need to be finalized by you

def compute_Ne_Be_detJ(nodes, ξ, η):
    """
    Compute the stiffness matrix and element external force vector
    for a bilinear plane stress or plane strain element.
    
    Parameters:
        nodes : (4, 2) ndarray
            Node coordinates [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]. 
        ξ, η : float
            Local coorinates in the parent domain
    
    Returns:
        N : numpy.ndarray
            Matrix of shape functions evaluated in the point (ξ, η)  (2x8)
        B : numpy.ndarray
            B-Matrix containing derivatives of the shape functions wrt the global coorinate system evaluated in the point (ξ, η)  (3x8)
        detJ : float
            Determinant of the jacobian matrix J (2x2)    
    """
    
    # Shape functions
    Ne = ...

    # Derivatives of shape functions
    dNe = ...

    # Jacobian matrix
    J = ...

    detJ = ...
    minDetJ = 1e-16
    if detJ < minDetJ:
        raise ValueError(f"Bad element geometry: detJ = {detJ}") # may happen if the nodes are not counter-clockwize 

    # Derivatives of shape functions w.r.t global coordinates x, y
    dNedxy = ...

    # N matrix 
    N = ...

    # B-matrix
    Be = ...

    return N, Be, detJ

def bilinear_element(nodes, D, t, body_load, ngp):
    """
    Compute the stiffness matrix and element external force vector
    for a bilinear plane stress or plane strain element.
    
    Parameters:
        nodes : (4, 2) ndarray
            Node coordinates [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]. 
        t : float
            Thickness
        D : numpy.ndarray
            Constitutive matrix for 2D elasticity (plane stress or plane strain)
        body_load: array-like
            Body forces [bx, by]
        ngp : int
            The number of Gauss points: 1^2, 2^2, 3^2
    
    Returns:
        Ke : numpy.ndarray
            Element stiffness matrix (8x8)
        fe : numpy.ndarray
            Equivalent nodal forces (8x1)
    """
    b = np.asarray(body_load, dtype=float).reshape(2)
    
    Ke = np.zeros(...)
    fe = np.zeros(...)

    # Define Gauss points and weights should handle three cases: 1^2, 2^2, 3^2 points
    # see function gauss_integration_rule in mha021 for support
    weights = ...
    coords = ...
    
    for gpIndex_1, weight_ξ in enumerate(weights):
        for gpIndex_2, weight_η in enumerate(weights):
            ξ = ...
            η = ...
            
            N, Be, detJ = compute_Ne_Be_detJ(nodes, ξ, η) # use the function you wrote earlier

            # Stiffness matrix and force vector
            Ke += ...
            fe += ...

    return Ke, fe

def bilinear_element_stress_strain(nodes: np.ndarray, D: np.ndarray, ae: np.ndarray):
    """
    Compute stress and strain for a bilinear quad element.

    Parameters
    ----------
    nodes : (4, 2) ndarray
        Node coordinates [[x1,y1],[x2,y2],[x3,y3],[x4,y4]].
    D : (3, 3) ndarray
        Constitutive matrix.
    ae : (8,) ndarray
        Nodal displacement vector [u1,v1,u2,v2,u3,v3,u4,v4].

    Returns
    -------
    stress : (3,) ndarray
        Stress vector [σ_xx, σ_yy, σ_xy].
    strain : (3,) ndarray
        Strain vector [ε_xx, ε_yy, γ_xy].
    """
    ϵe = ...
    σe = ...
    return σe, ϵe

#%%
####################################################################################################
####################################################################################################
####################################################################################################



# Verification



####################################################################################################
####################################################################################################
####################################################################################################
new_prob('Verification')

#---------------------------------------------------------------------------------------------------
# 
#---------------------------------------------------------------------------------------------------
nodes = np.array([[0.1, 0.0],
                [1.0, 0.0],
                [1.2, 1.0],
                [0.0, 1.3]]) # an element defined by these four nodes

N, B, detJ = compute_Ne_Be_detJ(nodes, ξ=0.15, η=0.25) # Call your function here with the provied nodes, ξ and η  

# It should then produce the following output
N_ref = np.array([
    [0.159375, 0., 0.215625, 0., 0.359375, 0., 0.265625, 0. ],
    [0., 0.159375, 0., 0.215625, 0., 0.359375, 0., 0.265625]
])
B_ref = np.array([
    [-0.40532365,  0.        ,  0.25408348,  0.        ,  0.65537407,   0.        , -0.5041339 ,  0.        ],
    [ 0.        , -0.35087719,  0.        , -0.52631579,  0.        ,   0.46783626,  0.        ,  0.40935673],
    [-0.35087719, -0.40532365, -0.52631579,  0.25408348,  0.46783626,   0.65537407,  0.40935673, -0.5041339 ]
])
detJ_ref = 0.3099375

# automatically compare your result against the reference 
print(f" N is correct: {np.allclose(N, N_ref)}")
print(f" B is correct: {np.allclose(B, B_ref)}")
print(f" detJ is correct: {np.allclose([detJ], [detJ_ref])}")


#---------------------------------------------------------------------------------------------------
# 
#---------------------------------------------------------------------------------------------------
# Check Ke and fe for number of Gauss points = 2x2 = 4
D = np.array([
    [ 1, .1,   0],
    [.1,  1,   0],
    [ 0,  0, 0.5],
])
Ke, fe = bilinear_element(nodes, D, t=1, body_load=[1, 2], ngp=4)

Ke_ref = np.array([
    [ 0.59197373,  0.16437482, -0.3259681 , -0.07816942, -0.31935773, -0.15689005,  0.0533521 ,  0.07068465],
    [ 0.16437482,  0.5122176 ,  0.12183058, -0.06840708, -0.15689005, -0.25651223, -0.12931535, -0.18729828],
    [-0.3259681 ,  0.12183058,  0.53867555, -0.13007999, -0.00506561, -0.13091923, -0.20764184,  0.13916864],
    [-0.07816942, -0.06840708, -0.13007999,  0.54735146,  0.06908077, -0.24209229,  0.13916864, -0.23685208],
    [-0.31935773, -0.15689005, -0.00506561,  0.06908077,  0.57250116,  0.16384019, -0.24807781, -0.07603092],
    [-0.15689005, -0.25651223, -0.13091923, -0.24209229,  0.16384019,  0.49395293,  0.12396908,  0.00465159], 
    [ 0.0533521 , -0.12931535, -0.20764184,  0.13916864, -0.24807781,  0.12396908,  0.40236755, -0.13382237],
    [ 0.07068465, -0.18729828,  0.13916864, -0.23685208, -0.07603092,  0.00465159, -0.13382237,  0.41949877]])

fe_ref = np.array([0.3   , 0.6   , 0.2775, 0.555 , 0.3075, 0.615 , 0.33  , 0.66  ])
print(f" Ke is correct: {np.allclose(Ke, Ke_ref)}")
print(f" fe is correct: {np.allclose(fe, fe_ref)}")

#%%
####################################################################################################
####################################################################################################
####################################################################################################



# Verification



####################################################################################################
####################################################################################################
####################################################################################################
new_prob('Verification')