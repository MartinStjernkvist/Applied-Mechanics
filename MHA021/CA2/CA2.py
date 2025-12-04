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
    
from mha021 import *

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

def figg(fig_name):
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



# Task 1



####################################################################################################
####################################################################################################
####################################################################################################
new_task('Task 1')

# Define inputs
W = 5 # m
H = 0.4 # m
t = 0.01 # m
E = 210e9 # Pa
nu = 0.3
rho = 7800 # kg/m^3
g = 9.81 # m/s^2
b = [0, -rho * g] # N




L = W * 2 
A = t * H
I = (t * (H**3)) / 12
w = rho * A * g 
delta_analytic = -(5 * w * (L**4)) / (384 * E * I)
M_max = (w * (L**2)) / 8
c = H / 2
sigma_analytic = (M_max * c) / I

# # Analytical: simply supported beam with uniform load q = rho*g*A
# q = rho * g * (H * t)
# I = t * H**3 / 12
# # L here is the FULL length (W_full)
# delta_analytic = -(5 * q * (W * 2)**4) / (384 * E * I)

# # Max Stress (My/I): M_max = qL^2/8
# M_max = q * (W * 2)**2 / 8
# sigma_analytic = M_max * (H/2) / I


def task1(element_type='cst', nelx=50, nely=10, plot_n_print=False):

    mesh = MeshGenerator.structured_rectangle_mesh(
        width=W,
        height=H,
        nx=nelx,
        ny=nely
        # dofs_per_node = 2
        # element_type = "tri"
    )

    nodes = mesh.nodes
    elements = mesh.elements
    edge_nodes = mesh.edges
    Edof = mesh.edofs

    if plot_n_print == True:
        fig = mesh.plot('mesh')
        fig = plot_mesh(nodes, elements, edge_nodes)
        fig.show()
        # displayvar('right edge nodes', edge_nodes['right'])
        # display(Edof)
    else:
        pass
    
    edof_map = build_edof(elements, dofs_per_node=2)

    D = hooke_2d_plane_stress(E, nu)
    # displayvar('D', D)
    
    # Assembly of K and f
    ndofs = nodes.shape[0] * 2
    K = np.zeros((ndofs, ndofs))
    f = np.zeros(ndofs)
    
    for el in range(len(elements)):
        Ke, fe = cst_element(nodes[elements[el, :] - 1], D, t, b)
        dofs = Edof[el, :]
        assem(K, Ke, dofs)
        assem(f, fe, dofs)
    
    # displayvar('K', K, accuracy=3)
    # displayvar('f', f, accuracy=3)
    
    bc_dofs = []
    bc_vals = []
    
    # Symmetry condition (right edge)
    right_nodes = edge_nodes['right']
    for n in right_nodes:
        dof_x = 2 * (n - 1) + 1
        bc_dofs.append(dof_x)
        bc_vals.append(0)

    # Support condition (left bottom node)
    left_nodes = edge_nodes['left']
    min_y = np.min(nodes[left_nodes - 1, 1])
    support_node = None
    for n in left_nodes:
        if np.isclose(nodes[n - 1, 1], min_y):
            support_node = n
            break
            
    if support_node:
        # bc_dofs.append(2 * (support_node - 1) + 1) # u_x
        # bc_vals.append(0)
        bc_dofs.append(2 * (support_node - 1) + 2) # u_y
        bc_vals.append(0)
    else:
        print("support node not found")
        
    a, r = solve_eq(K, f, bc_dofs, bc_vals)
    # displayvar('a', a, accuracy=3)
    # displayvar('r', r, accuracy=3)
    
    right_dofs_y = [2 * (n - 1) + 2 for n in right_nodes]
    disp_y_right = a[np.array(right_dofs_y) - 1]
    avg_deflection = np.mean(disp_y_right)
    # displayvar('u_{right, avg}', avg_deflection, accuracy=3)
    
    ed = extract_dofs(a, Edof)
    if plot_n_print == True:
        fig = plot_deformed_mesh(nodes, elements, ed, scale=40e-3, field='utotal')
        fig.show()
    else:
        pass
    
    el_stresses = np.zeros((len(elements), 3))
    for el in range(len(elements)):
        nodes[elements[el, :] - 1]
        dofs = Edof[el, :]
        ae = a[dofs - 1]
        σe, ϵe = cst_element_stress_strain(nodes[elements[el, :] - 1], D, ae)
        el_stresses[el, :] = σe
    # displayvar('σ', el_stresses, accuracy=3)
    
    right_norm_stresses = []
    
    # for n in right_nodes:
    #     right_norm_stresses.append(el_stresses[n, :])
    right_elements = [el for el in range(len(elements)) 
                  if sum(n in right_nodes for n in elements[el, :]) >= 2]

    right_edge_stresses = el_stresses[right_elements, :]
    right_max_norm_stress = np.max(np.abs(right_edge_stresses[:, 0]))
    print(f'maximum normal stress at right edge: {right_max_norm_stress:.3e} Pa')
    
    print(f'\nComparison with analytical solution: \n Deflection: {avg_deflection:.3e} m vs {delta_analytic:.3e} m \n Stress: {right_max_norm_stress:.3e} Pa vs {sigma_analytic:.3e} Pa')

    return avg_deflection, right_max_norm_stress

task1(element_type='cst', nelx=40, nely=8, plot_n_print=True)
# task1(element_type='cst', nelx=250, nely=50, plot_n_print=False)

#%%
#---------------------------------------------------------------------------------------------------
# Convergence
#---------------------------------------------------------------------------------------------------
new_subtask('Task 1 - Convergence')

nelx_list = np.arange(50, 250, 25)
nely_list = [int(i * (H / W)) for i in nelx_list]
print(nelx_list)
print(nely_list)

avg_deflection_list = []
max_norm_stress_list = []
num_of_nodes_list = []
for i in range(len(nelx_list)):
    avg_deflection, right_max_norm_stress = task1(element_type='cst', nelx=nelx_list[i], nely=nely_list[i])
    avg_deflection_list.append(avg_deflection)
    max_norm_stress_list.append(right_max_norm_stress)
    num_of_nodes_list.append(int(nelx_list[i] * nely_list[i]))
    print(num_of_nodes_list)
    print(np.shape(avg_deflection_list))

    plt.figure()
    plt.plot(num_of_nodes_list, avg_deflection_list, 'X-')
    plt.axhline(delta_analytic, color='orange', linestyle='--', alpha=0.5)
    plt.xlabel('Number of elements')
    plt.ylabel('Average deflection (m)')
    plt.show()
    
    plt.figure()
    plt.plot(num_of_nodes_list, max_norm_stress_list, 'X-', color='red')
    plt.axhline(sigma_analytic, color='orange', linestyle='--', alpha=0.5)
    plt.xlabel('Number of elements')
    plt.ylabel('Maximum normal stress (Pa)')
    plt.show()
#%%

#%%
####################################################################################################
####################################################################################################
####################################################################################################



# Task 2



####################################################################################################
####################################################################################################
####################################################################################################
new_task('Task 2')

#---------------------------------------------------------------------------------------------------
# Starting point
#---------------------------------------------------------------------------------------------------
new_subtask('Starting point')

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
    # Order: (-1,-1), (1,-1), (1,1), (-1,1) -> Counter-clockwise
    Ne = 0.25 * np.array([
        (1 - ξ) * (1 - η),
        (1 + ξ) * (1 - η),
        (1 + ξ) * (1 + η),
        (1 - ξ) * (1 + η)
    ])
    
    # dN_dξ
    dN_dxi = 0.25 * np.array([
        -(1 - η),
         (1 - η),
         (1 + η),
        -(1 + η)
    ])
    
    # dN_dη
    dN_deta = 0.25 * np.array([
        -(1 - ξ),
        -(1 + ξ),
         (1 + ξ),
         (1 - ξ)
    ])

    # Derivatives of shape functions
    dNe = np.vstack((dN_dxi, dN_deta))

    # Jacobian matrix
    J = dNe @ nodes

    detJ = np.linalg.det(J)
    
    minDetJ = 1e-16
    if detJ < minDetJ:
        raise ValueError(f"Bad element geometry: detJ = {detJ}") # may happen if the nodes are not counter-clockwize 
    Jinv = np.linalg.inv(J)
    
    # Derivatives of shape functions w.r.t global coordinates x, y
    dNedxy = Jinv @ dNe

    # N matrix 
    N = np.zeros((2, 8))
    N[0, 0::2] = Ne
    N[1, 1::2] = Ne

    # B-matrix
    Be = np.zeros((3, 8))
    for i in range(4):
        dNdx = dNedxy[0, i]
        dNdy = dNedxy[1, i]
        
        # Column indices for u_i and v_i
        idx_u = 2 * i
        idx_v = 2 * i + 1
        
        Be[0, idx_u] = dNdx
        Be[1, idx_v] = dNdy
        Be[2, idx_u] = dNdy
        Be[2, idx_v] = dNdx

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
    
    Ke = np.zeros((8, 8))
    fe = np.zeros(8)

    # Define Gauss points and weights should handle three cases: 1^2, 2^2, 3^2 points
    # see function gauss_integration_rule in mha021 for support
    coords, weights = gauss_integration_rule(int(np.sqrt(ngp))) 
    
    for gpIndex_1, weight_ξ in enumerate(weights):
        for gpIndex_2, weight_η in enumerate(weights):
            ξ = coords[gpIndex_1]
            η = coords[gpIndex_2]
            
            N, Be, detJ = compute_Ne_Be_detJ(nodes, ξ, η) # use the function you wrote earlier
            
            weight_factor =weight_ξ *weight_η

            # Stiffness matrix and force vector
            Ke += Be.T @ D @ Be * detJ * weight_factor
            fe += (N.T @ b) * detJ * weight_factor

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
    _, Be, _ = compute_Ne_Be_detJ(nodes, 0.0, 0.0)
    ϵe = Be @ ae
    σe = D @ ϵe
    return σe, ϵe

#---------------------------------------------------------------------------------------------------
# Verification
#---------------------------------------------------------------------------------------------------
new_subtask('Verification')

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