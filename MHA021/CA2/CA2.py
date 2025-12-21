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

def sfig(fig_name):
    fig_output_file = script_dir / "figures" / fig_name
    fig_output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_output_file, dpi=dpi, bbox_inches='tight')
    print('figure name: ', fig_name)
    
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

# Analytical: simply supported beam with uniform load
L = W * 2 
A = t * H
I = (t * (H**3)) / 12
w = rho * A * g 
delta_analytic = -(5 * w * (L**4)) / (384 * E * I)
M_max = (w * (L**2)) / 8
c = H / 2
sigma_analytic = (M_max * c) / I

def task12(element_type='cst', nelx=50, nely=10, plot_n_print=False, W=5):

    if element_type == 'cst':
        mesh = MeshGenerator.structured_rectangle_mesh(
            width=W,
            height=H,
            nx=nelx,
            ny=nely
        )
    else:
        mesh = MeshGenerator.semistructured_rectangle_mesh_quads(
            width=W,
            height=H,
            nx=nelx,
            ny=nely
        )

    nodes = mesh.nodes
    elements = mesh.elements
    edge_nodes = mesh.edges
    Edof = mesh.edofs
    
    el_centers = np.mean(nodes[elements[:, :] - 1], axis=1)

    if plot_n_print == True:
        fig = mesh.plot('mesh')
        fig = plot_mesh(nodes, elements, edge_nodes)
        fig.show()
    else:
        pass
    
    edof_map = build_edof(elements, dofs_per_node=2)

    D = hooke_2d_plane_stress(E, nu)
    
    ndofs = nodes.shape[0] * 2
    K = np.zeros((ndofs, ndofs))
    f = np.zeros(ndofs)
    
    for el in range(len(elements)):
        
        if element_type == 'cst':
            Ke, fe = cst_element(nodes[elements[el, :] - 1], D, t, b)
        else:
            Ke, fe = bilinear_element(nodes[elements[el, :] - 1], D, t, b, ngp=4)
            
        dofs = Edof[el, :]
        assem(K, Ke, dofs)
        assem(f, fe, dofs)
    
    bc_dofs = []
    bc_vals = []
    
    # Symmetry condition (right edge)
    right_nodes = edge_nodes['right']
    for n in right_nodes:
        bc_dofs.append(2 * (n - 1) + 1)
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
        bc_dofs.append(2 * (support_node - 1) + 2) # u_y
        bc_vals.append(0)
    else:
        print("support node not found")
        
    a, r = solve_eq(K, f, bc_dofs, bc_vals)
    
    right_dofs_y = [2 * (n - 1) + 2 for n in right_nodes]
    uy_right = a[np.array(right_dofs_y) - 1]
    uy_avg = np.mean(uy_right)
    
    ed = extract_dofs(a, Edof)
    if plot_n_print == True:
        fig = plot_deformed_mesh(nodes, elements, ed, scale=40e-3, field='uy')
        fig.show()
    else:
        pass
    
    el_stress = np.zeros((len(elements), 3))
    el_strain = np.zeros((len(elements), 3))

    for el in range(len(elements)):
        nodes[elements[el, :] - 1]
        dofs = Edof[el, :]
        ae = a[dofs - 1]
        
        if element_type == 'cst':
            σe, ϵe = cst_element_stress_strain(nodes[elements[el, :] - 1], D, ae)
        else:
            σe, ϵe = bilinear_element_stress_strain(nodes[elements[el, :] - 1], D, ae)
            
        el_stress[el, :] = σe
        el_strain[el, :] = ϵe
        
    el_right_edge = [el for el in range(len(elements)) 
                  if sum(n in right_nodes for n in elements[el, :]) >= 2]

    right_edge_stress = el_stress[el_right_edge, :]
    sigmaxx_max = np.max(np.abs(right_edge_stress[:, 0]))
        
    print(f'\n number of DOFs: {ndofs}')
    print(f'Deflection: {uy_avg:.3e} m')
    print(f'Stress: {sigmaxx_max:.3e} Pa')
    
    # Analytical: simply supported beam with uniform load
    L = W * 2
    A = t * H
    I = (t * (H**3)) / 12
    w = rho * A * g 
    delta_analytic = -(5 * w * (L**4)) / (384 * E * I)
    M_max = (w * (L**2)) / 8
    c = H / 2
    sigma_analytic = (M_max * c) / I
    
    # Displacement and stress fractions
    f_uy = uy_avg / delta_analytic
    f_sigmaxx = sigmaxx_max / sigma_analytic
    # print(f'\nComparison with analytical solution:')
    # print(f'Deflection fraction (result / analytical): {f_uy:.2e}')
    # print(f'Stress fraction (result / analytical): {f_sigmaxx:.2e}')

    return uy_avg, sigmaxx_max, ndofs, f_uy, f_sigmaxx, nodes, elements, el_centers, el_stress, el_strain


#%%
#---------------------------------------------------------------------------------------------------
# Run analysis
#---------------------------------------------------------------------------------------------------
new_subtask('Task 1 - Run analysis')

# task12(element_type='cst', nelx=40, nely=8, plot_n_print=True)

task12(element_type='cst', nelx=212, nely=16, plot_n_print=False)

#%%
#---------------------------------------------------------------------------------------------------
# Convergence
#---------------------------------------------------------------------------------------------------
new_subtask('Task 1 - Convergence')

nelx_list = []
for i in range(5):
    nelx_list.append(int(300 / np.sqrt(2)**(4 - i)))
nely_list = [int(i * (H / W)) for i in nelx_list]

uy_avg_list = []
sigmaxx_max_list = []
ndofs_list = []
f_uy_list = []
f_sigmaxx_list = []

for i in range(len(nelx_list)):
    uy_avg, sigmaxx_max, ndofs, f_uy, f_sigmaxx, nodes, elements, el_centers, el_stress, el_strain = task12(element_type='cst', nelx=nelx_list[i], nely=nely_list[i])
    
    uy_avg_list.append(uy_avg)
    sigmaxx_max_list.append(sigmaxx_max)
    ndofs_list.append(ndofs)
    f_uy_list.append(f_uy)
    f_sigmaxx_list.append(f_sigmaxx)
    
    if i != 0:
        rel_change_uy = (f_uy_list[i] - f_uy_list[i -1 ]) / f_uy_list[i]
        rel_change_sigmaxx = (f_sigmaxx_list[i] - f_sigmaxx_list[i - 1]) / f_sigmaxx_list[i]

        print(f'relative change in displacement: {rel_change_uy*100:.2f} %')
        print(f'relative change in stress: {rel_change_sigmaxx*100:.2f} %')
        
        if rel_change_uy <= 0.02:
            print(f'Deflection convergence for NDOF = {ndofs:.2f}')
            print(f'nelx, nely: {nelx_list[i], nely_list[i]}')
        
        if rel_change_sigmaxx <= 0.02:
            print(f'Stress convergence for NDOF = {ndofs:.2f}')
            print(f'nelx, nely: {nelx_list[i], nely_list[i]}')

plt.figure()
plt.plot(ndofs_list, uy_avg_list, 'X-')
plt.axhline(delta_analytic, color='orange', linestyle='--', alpha=0.5)
plt.xlabel('Number of DOFs')
plt.ylabel('Average deflection (m)')
sfig('Average deflection vs Number of DOFs.png')
plt.show()

plt.figure()
plt.plot(ndofs_list, f_uy_list, 'o-', color='black')
plt.axhline(1, color='orange', linestyle='--', alpha=0.5)
plt.xlabel('Number of DOFs')
plt.ylabel('Displacement fraction: Numerical / Analytical')
sfig('Displacement fraction vs Number of DOFs.png')
plt.show()

plt.figure()
plt.plot(ndofs_list, sigmaxx_max_list, 'X-', color='red')
plt.axhline(sigma_analytic, color='orange', linestyle='--', alpha=0.5)
plt.xlabel('Number of DOFs')
plt.ylabel('Maximum normal stress (Pa)')
sfig('Maximum normal stress vs Number of DOFs.png')
plt.show()

plt.figure()
plt.plot(ndofs_list, f_sigmaxx_list, 'o-', color='black')
plt.axhline(1, color='orange', linestyle='--', alpha=0.5)
plt.xlabel('Number of DOFs')
plt.ylabel('Stress fraction: Numerical / Analytical')
sfig('Stress fraction vs Number of DOFs.png')
plt.show()

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
new_subtask('Task 2 - Starting point')

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

#%%
#---------------------------------------------------------------------------------------------------
# Verification
#---------------------------------------------------------------------------------------------------
new_subtask('Task 2 - Verification')

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
#---------------------------------------------------------------------------------------------------
# Run analysis
#---------------------------------------------------------------------------------------------------
new_subtask('Task 2 - Run analysis')

task12(element_type='quad', nelx=40, nely=8, plot_n_print=True)

#%%
#---------------------------------------------------------------------------------------------------
# Plot stress and strain (d)
#---------------------------------------------------------------------------------------------------
new_subtask('Task 2 - Plot stress and strain (d)')

uy_avg, sigmaxx_max, ndofs, f_uy, f_sigmaxx, nodes, elements, el_centers, el_stress, el_strain = task12(element_type='quad', nelx=40, nely=8, plot_n_print=False)

import matplotlib.tri as tri

def plot_single_contour(el_centers, field_data, field_name, file_name, cmap='jet'):
    plt.figure()
    
    triang = tri.Triangulation(el_centers[:, 0], el_centers[:, 1])
    contour = plt.tricontourf(triang, field_data, levels=30, cmap=cmap)
    plt.tricontour(triang, field_data, levels=30, colors='k', linewidths=0.5, alpha=0.3)
    
    cbar = plt.colorbar(contour)
    cbar.set_label(field_name)
    
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.title(f'{field_name}')
    plt.gca().set_aspect('equal')
    
    plt.savefig('figures/' + file_name + '.png')
    plt.show()
    
horizontal_strain = el_strain[:, 0]
plot_single_contour(el_centers, horizontal_strain, 'Horizontal strain $\epsilon_{xx}$', 'Horizontal strain', cmap='jet')

shear_strain = el_strain[:, 2]
plot_single_contour(el_centers, shear_strain, 'Shear strain $\gamma_{xy}$', 'Shear strain', cmap='jet')

vertical_strain = el_strain[:, 1]
plot_single_contour(el_centers, vertical_strain, 'Vertical strain $\epsilon_{yy}$', 'Vertical strain', cmap='jet')

normal_stress = el_stress[:, 0]
plot_single_contour(el_centers, normal_stress, 'Normal stress $\sigma_{xx}$','Normal stress', cmap='jet')

print(np.shape(horizontal_strain))
print(np.shape(vertical_strain))
print(np.shape(shear_strain))

# Plot fractions
fraction_horizontal_shear = []
for i in range(len(horizontal_strain)):
    fraction_horizontal_shear.append(shear_strain[i] / horizontal_strain[i])
plot_single_contour(el_centers, fraction_horizontal_shear, 'Fraction shear / horizontal strain','Fraction shear horizontal', cmap='jet')

fraction_horizontal_vertical = []
for i in range(len(horizontal_strain)):
    fraction_horizontal_vertical.append(vertical_strain[i] / horizontal_strain[i])
plot_single_contour(el_centers, fraction_horizontal_vertical, 'Fraction vertical / horizontal strain','Fraction vertical horizontal', cmap='jet')

#%%
#---------------------------------------------------------------------------------------------------
# Euler-Bernoulli breakdown (e)
#---------------------------------------------------------------------------------------------------
new_subtask('Euler-Bernoulli breakdown (e)')

ratios = np.arange(1, 17, 0.5)
f_uy_list = []
f_sigmaxx_list = []

for r in ratios:
    W_val = (r * 0.4) / 2
    _, _, _, f_uy, f_sigmaxx, _, _, el_centers, el_stress, el_strain = task12(element_type='quad', nelx=int(W_val * 50), nely=20, plot_n_print=False, W=W_val)
    print(f'\nratio = {r}: displacement fraction: {f_uy:.2e} stress fraction: {f_sigmaxx:.2e}')
    
    f_uy_list.append(f_uy)
    f_sigmaxx_list.append(f_sigmaxx)
    
    if f_uy >= 1.05:
        print('More than 5% difference')
    
    if r in [1, 4, 16]:
        shear_strain = el_strain[:, 1]
        plot_single_contour(el_centers, shear_strain, 'Vertical strain $\epsilon_{yy}$' + f'\nratio ={r}', 'Shear strain_' + str(r), cmap='jet')

plt.figure()
plt.plot(ratios, f_uy_list, 'o-', label='displacement', alpha=0.75)
plt.plot(ratios, f_sigmaxx_list, 'o-', label='stress', color='red', alpha=0.75)
plt.axhline(y=1, color='black', label='fraction = 1', linestyle='--', alpha=0.75)
plt.xlabel('ratio L/H')
plt.ylabel('fraction: numerical / analytical')
plt.legend()
plt.tight_layout()
plt.savefig('figures/Euler Bernoulli breakdown')
plt.show()
#%%