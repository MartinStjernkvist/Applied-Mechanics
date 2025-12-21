#%%
# %matplotlib widget

import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import math
import pandas as pd
import gmsh

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



# Task 1 - Copy from CA2, with modification



####################################################################################################
####################################################################################################
####################################################################################################
new_task('Task 1 - Copy from CA2, with modification')

def compute_Ne_Be_detJ(nodes, ξ, η):
    
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

# Define inputs
W = 5 # m
H = 0.4 # m
t = 0.01 # m
E = 210e9 # Pa
nu = 0.3
rho = 7800 # kg/m^3
g = 9.81 # m/s^2
b = [0, -rho * g] # N

def task12(E, rho, nelx=50, nely=10, plot_n_print=False, W=5):

    mesh = MeshGenerator.structured_rectangle_mesh(
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
    
    # Mass matrix
    M = np.zeros_like(K)
    
    for el in range(len(elements)):
        
        Ke, fe = cst_element(nodes[elements[el, :] - 1], D, t, b)
        
        # Element mass matrix
        Me = cst_element_M(nodes[elements[el, :] - 1], rho, t)
            
        dofs = Edof[el, :]
        assem(K, Ke, dofs)
        assem(f, fe, dofs)
        
        # Assemble mass matrix
        assem(M, Me, dofs)
    
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
    
    # Reduce matrices
    all_dofs = np.arange(1, ndofs + 1, 1, dtype=int)
    free_dofs = np.setdiff1d(all_dofs, bc_dofs)
    K_red = extract_block(K, free_dofs)
    M_red = extract_block(M, free_dofs)
    
    # Solve for eigenmodes and frequencies
    omega2, phi_red = eigh(K_red, M_red)
    f = np.sqrt(omega2)/(2*np.pi)
    
    # phi_j
    phi_j = np.zeros(ndofs)
    phi_j[free_dofs - 1] = phi_red[:, 0]
    
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
        
        σe, ϵe = cst_element_stress_strain(nodes[elements[el, :] - 1], D, ae)
        
        el_stress[el, :] = σe
        el_strain[el, :] = ϵe
        
    el_right_edge = [el for el in range(len(elements)) 
                  if sum(n in right_nodes for n in elements[el, :]) >= 2]

    right_edge_stress = el_stress[el_right_edge, :]
    sigmaxx_max = np.max(np.abs(right_edge_stress[:, 0]))
    
    return f, phi_red, ndofs, free_dofs, nodes, elements, Edof

#%%
#---------------------------------------------------------------------------------------------------
# Task 1 - a) Lowest natural frequencies
#---------------------------------------------------------------------------------------------------
new_subtask('Task 1 - a) Lowest natural frequencies')

f, phi_red, ndofs, free_dofs, nodes, elements, Edof = task12(E, rho, nelx=212, nely=16, plot_n_print=False)

indices = np.argsort(f)[:3]
f_1, f_2, f_3 = f[indices[0]], f[indices[1]], f[indices[2]]
print('Steel:')
print(f'f1: {f_1:.2f} Hz')
print(f'f2: {f_2:.2f} Hz')
print(f'f3: {f_3:.2f} Hz')

print(f'omega1: {f_1 * 2 * np.pi:.2f} rad/s')
print(f'omega2: {f_2 * 2 * np.pi:.2f} rad/s')
print(f'omega3: {f_3 * 2 * np.pi:.2f} rad/s')

#%%
#---------------------------------------------------------------------------------------------------
# Task 1 - a) Plot modes
#---------------------------------------------------------------------------------------------------
new_subtask('Task 1 - a) Plot modes')

for i in range(len(indices)):
    phi_j = np.zeros((ndofs))

    mode = indices[i]

    phi_j[free_dofs - 1] = phi_red[:, mode]
    ed = extract_dofs(phi_j, Edof)

    fig = plot_deformed_mesh(nodes, elements, ed, scale=1, field='uy')
    fig.show()
    print(f'Mode phi_{i+1}')

#%%
#---------------------------------------------------------------------------------------------------
# Task 1 - a) Convergence validation
#---------------------------------------------------------------------------------------------------
new_subtask('Task 1 - a) Convergence validation')

# Reduce number of DOFs by half
nelx = int(212 * (1 / np.sqrt(2)))
nely = int(16* (1 / np.sqrt(2)))

f, _, _, _, _, _, _ = task12(E, rho, nelx, nely, plot_n_print=False)
f_sorted = np.sort(f)
f_1_prev, f_2_prev, f_3_prev = f_sorted[0], f_sorted[1], f_sorted[2]

f1_change = (f_1 - f_1_prev) / f_1_prev * 100
f2_change = (f_2 - f_2_prev) / f_2_prev * 100
f3_change = (f_3 - f_3_prev) / f_3_prev * 100

print('Change in reducing DOFs by half:')
print(f'Change in f1: {f1_change:.2f} %')
print(f'Change in f2: {f2_change:.2f} %')
print(f'Change in f3: {f3_change:.2f} %')
# All below 2% -> converged results

#%%
#---------------------------------------------------------------------------------------------------
# Task 1 - b) LLM verification
#---------------------------------------------------------------------------------------------------
new_subtask('Task 1 - b) LLM verification')

L = 2 * W       # Total Length

# Analytical Solution (Euler-Bernoulli)
n_modes = np.array([1, 2, 3, 4, 5])
f_analytical = (np.pi * n_modes**2 * H / (2 * L**2)) * np.sqrt(E / (12 * rho))

print(f"Analytical Frequencies (Hz):", f_analytical)


#%%
#---------------------------------------------------------------------------------------------------
# Task 1 - c) Different materials
#---------------------------------------------------------------------------------------------------
new_subtask('Task 1 - c) Different materials')

E_aluminium = 69 * 10**(9)
rho_aluminium = 2.7 * 10**(3)
E, rho = E_aluminium, rho_aluminium

f, _, _, _, _, _, _ = task12(E, rho, nelx=212, nely=16, plot_n_print=False)
f_sorted = np.sort(f)
f_1, f_2, f_3 = f_sorted[0], f_sorted[1], f_sorted[2]
print('Aluminium:')
print(f'f1: {f_1:.2f} Hz')
print(f'f2: {f_2:.2f} Hz')
print(f'f3: {f_3:.2f} Hz')

E_copper = 117 * 10**(9)
rho_copper = 8.96 * 10**(3)
E, rho = E_copper, rho_copper

f, _, _, _, _, _, _ = task12(E, rho, nelx=212, nely=16, plot_n_print=False)
f_sorted = np.sort(f)
f_1, f_2, f_3 = f_sorted[0], f_sorted[1], f_sorted[2]
print('\nCopper:')
print(f'f1: {f_1:.2f} Hz')
print(f'f2: {f_2:.2f} Hz')
print(f'f3: {f_3:.2f} Hz')

#%%
####################################################################################################
####################################################################################################
####################################################################################################



# Task 2 - Starting point



####################################################################################################
####################################################################################################
####################################################################################################
new_task('Task 2 - Starting point')

#%%
#---------------------------------------------------------------------------------------------------
# Starting point - Mesh code
#---------------------------------------------------------------------------------------------------
new_subtask('Starting point - Mesh code')

def generate_floor_mesh(width=10.0, height=5.0, radius=1.0, vertical_offset=0.0, mesh_size=0.5):
    """
    Generate a 2D triangular mesh of a rectangle with a circular cutout on the left edge.

    Parameters:
        width (float): Rectangle width.
        height (float): Rectangle height.
        radius (float): Radius of the circular cutout.
        vertical_offset (float): Vertical offset of the circle's center from the rectangle's center.
        mesh_size (float): Target mesh element size.

    Returns:
        nodes (np.ndarray): Node coordinates (N x 2).
        elements (np.ndarray): Triangle connectivity (M x 3) with 1-based node indices.
        node_groups (dict): Dictionary of ordered node indices (1-based) for each boundary ('left', 'right', 'top', 'bottom', 'circle').
    """
    gmsh.initialize()
    gmsh.model.add("Rect_with_left_cutout")

    # Geometry: Rectangle and circular disk on left edge
    rect = gmsh.model.occ.addRectangle(0, 0, 0, width, height)
    cx = 0.0
    cy = vertical_offset
    circle = gmsh.model.occ.addDisk(cx, cy, 0, radius, radius)

    # Subtract circle from rectangle to create hole
    out = gmsh.model.occ.cut([(2, rect)], [(2, circle)], removeObject=True, removeTool=True)
    surface_tag = out[0][0][1]  # Tag of resulting surface
    gmsh.model.occ.synchronize()

    # Set mesh size on all points
    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), mesh_size)

    # Identify boundary curves of the surface
    boundary_curves = gmsh.model.getBoundary([(2, surface_tag)], oriented=False)
    edge_tags = [tag for dim, tag in boundary_curves if dim == 1]

    left_edges = []
    right_edges = []
    top_edges = []
    bottom_edges = []
    circle_edges = []

    tol = 1e-8
    for edge in edge_tags:
        # Get coordinates of the endpoints of the edge
        end_pts = gmsh.model.getBoundary([(1, edge)], oriented=False)
        pt_tags = [tag for dim, tag in end_pts if dim == 0]
        if len(pt_tags) != 2:
            continue  # skip if not a normal line segment
        # Coordinates of endpoints
        x1_min, y1_min, _, x1_max, y1_max, _ = gmsh.model.getBoundingBox(0, pt_tags[0])
        x2_min, y2_min, _, x2_max, y2_max, _ = gmsh.model.getBoundingBox(0, pt_tags[1])
        x1, y1 = 0.5 * (x1_min + x1_max), 0.5 * (y1_min + y1_max)
        x2, y2 = 0.5 * (x2_min + x2_max), 0.5 * (y2_min + y2_max)
        # Classify edge by orientation and position
        if abs(x1 - x2) < tol:
            # Vertical edge
            if abs(x1) < tol:  # near x = 0 (left side)
                com_x, com_y, _ = gmsh.model.occ.getCenterOfMass(1, edge)
                if abs(com_x) < tol:
                    left_edges.append(edge)   # straight left segment
                else:
                    circle_edges.append(edge) # arc segment on left
            elif abs(x1 - width) < tol:
                right_edges.append(edge)
        elif abs(y1 - y2) < tol:
            # Horizontal edge
            if abs(y1) < tol:
                com_x, com_y, _ = gmsh.model.occ.getCenterOfMass(1, edge)
                if abs(com_y) < tol:
                    bottom_edges.append(edge)  # straight bottom segment
                else:
                    circle_edges.append(edge)  # arc segment on bottom (if any)
            elif abs(y1 - height) < tol:
                top_edges.append(edge)
        else:
            # Non-axis-aligned edge (likely a circular arc)
            circle_edges.append(edge)

    # Create physical groups for boundary edges
    left_phys   = gmsh.model.addPhysicalGroup(1, left_edges)   if left_edges   else None
    right_phys  = gmsh.model.addPhysicalGroup(1, right_edges)  if right_edges  else None
    top_phys    = gmsh.model.addPhysicalGroup(1, top_edges)    if top_edges    else None
    bottom_phys = gmsh.model.addPhysicalGroup(1, bottom_edges) if bottom_edges else None
    circle_phys = gmsh.model.addPhysicalGroup(1, circle_edges) if circle_edges else None

    gmsh.model.mesh.generate(2)  # Generate 2D mesh (triangles)

    # Node coordinates
    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    node_tags = np.array(node_tags, dtype=int)
    coords = np.array(node_coords).reshape(-1, 3)[:, :2]  # Nx2 array of x, y coordinates
    tag_to_index = {tag: idx for idx, tag in enumerate(node_tags)}

    # Triangle connectivity (3 nodes per triangle, 1-based indexing)
    elements_list = []
    elem_types, _, elem_node_tags = gmsh.model.mesh.getElements(2)
    for etype, nodes in zip(elem_types, elem_node_tags):
        if etype == 2:  # 3-node triangle
            nodes = np.array(nodes, dtype=int)
            for i in range(0, len(nodes), 3):
                n1 = tag_to_index[nodes[i]]
                n2 = tag_to_index[nodes[i+1]]
                n3 = tag_to_index[nodes[i+2]]
                elements_list.append([n1, n2, n3])
    elements = np.array(elements_list, dtype=int) + 1

    # Helper to get sorted 1-based node indices for a physical group
    def get_sorted_nodes(phys_group, sort_by='x'):
        if phys_group is None:
            return np.array([], dtype=int)
        tags, coords_array = gmsh.model.mesh.getNodesForPhysicalGroup(1, phys_group)
        tags = np.array(tags, dtype=int)
        if tags.size == 0:
            return np.array([], dtype=int)
        # Get node indices and sort by coordinate
        idx = [tag_to_index[t] for t in tags]
        if sort_by == 'x':
            idx.sort(key=lambda i: coords[i, 0])
        elif sort_by == 'y':
            idx.sort(key=lambda i: coords[i, 1])
        return np.array([i + 1 for i in idx], dtype=int)

    # Ordered node groups (1-based indices for each boundary)
    node_groups = {
        'left':   get_sorted_nodes(left_phys,   sort_by='y'),
        'right':  get_sorted_nodes(right_phys,  sort_by='y'),
        'top':    get_sorted_nodes(top_phys,    sort_by='x'),
        'bottom': get_sorted_nodes(bottom_phys, sort_by='x'),
        'circle': get_sorted_nodes(circle_phys, sort_by='y')
    }

    gmsh.finalize()
    # return coords, elements, node_groups
    return Mesh(coords, elements, node_groups, dofs_per_node=1)

#%%
#---------------------------------------------------------------------------------------------------
# Starting point - Mesh generation & visualization
#---------------------------------------------------------------------------------------------------
new_subtask('Starting point - Mesh generation & visualization')

# Following are given in unit m
R = 10 * 10**(-3)
L_t = 35 * 10**(-3)
L_b = 95 * 10**(-3)
L_h = 100 * 10**(-3)

width = L_h
height = L_t + L_b + 2 * R
radius = R
vertical_offset = L_b
mesh_size=0.003

mesh = generate_floor_mesh(width, height, radius, vertical_offset, mesh_size)

fig = plot_mesh(mesh.nodes, mesh.elements, mesh.edges)
fig.show()

#%%
####################################################################################################
####################################################################################################
####################################################################################################



# Task 2 - Continuation



####################################################################################################
####################################################################################################
####################################################################################################
new_task('Task 2 - Continuation')

#%%
#---------------------------------------------------------------------------------------------------
# Task 2 - d) Solve system of equations
#---------------------------------------------------------------------------------------------------
new_subtask('Task 2 - d) Solve system of equations')

# You will additionally need to use the folowing functions
# plot_scalar_field
# plot_vector_field
# flow2t_Ke_fe
# flow2t_qe
# convection_Ke_fe

alpha_w = 1000 # W / (m^2 C)
T_air = 20 # C
alpha_air = 5 # W / (m^2 C)
T_b = 10 # C
k = 0.75 # W / (m C)
t = 1 # m

T_w = 30 # C

# Constitutive matrix
D = k * np.array([
    [1, 0],
    [0, 1]
])

# System matrices
num_nodes = mesh.nodes.shape[0]
num_el = mesh.elements.shape[0]
num_dofs = num_nodes

# Initiate stiffness and load matrices
K = np.zeros((num_dofs, num_dofs))
f = np.zeros((num_dofs, 1))

for el in range(num_el):
    el_nodes = mesh.nodes[mesh.elements[el] - 1]
    Ke, fe = flow2t_Ke_fe(el_nodes, D=D, t=t, Q=0)
    el_dofs = mesh.edofs[el, :]
    assem(K, Ke, el_dofs)
    assem(f, fe, el_dofs)

# Convection: water
conv_nodes_w = mesh.edges['circle']
num_edges_w = len(conv_nodes_w) - 1

for edge in range(num_edges_w):
    edge_nodes = conv_nodes_w[edge:edge + 2] - 1
    nodes = mesh.nodes[edge_nodes, :]
    # displayvar("nodes", nodes)
    Kec, fec = convection_Ke_fe(nodes, alpha=alpha_w, t=t, Tamb=T_w)
    dofs = edge_nodes + 1
    # displayvar("edge dofs", dofs)
    assem(K, Kec, dofs)
    assem(f, fec, dofs)


# Convection: air
conv_nodes_air = mesh.edges['top']
num_edges_air = len(conv_nodes_air) - 1

for edge in range(num_edges_air):
    edge_nodes = conv_nodes_air[edge:edge + 2] - 1
    nodes = mesh.nodes[edge_nodes, :]
    # displayvar("nodes", nodes)
    Kec, fec = convection_Ke_fe(nodes, alpha=alpha_air, t=t, Tamb=T_air)
    dofs = edge_nodes + 1
    # displayvar("edge dofs", dofs)
    assem(K, Kec, dofs)
    assem(f, fec, dofs)

# Essential boundary conditions
bottom_dofs = mesh.edges['bottom']

bc_dofs = bottom_dofs
bc_vals = np.ones_like(bc_dofs) * T_b

# Solve system
a, r = solve_eq(K, f, bc_dofs, bc_vals)

# Plot temperature field
Ed = extract_dofs(a, mesh.edofs)
# print('Ed shape and values:\n', np.shape(Ed), Ed)
fig = plot_scalar_field(mesh.nodes, mesh.elements, Ed, title=fr'Tempterature')
fig.show()

T_sum = 0
# Manual check for convergence of mean temperature of convection boundary for air
for edge in range(num_edges_air):
    
    node_i = conv_nodes_air[edge] - 1
    node_j = conv_nodes_air[edge + 1] - 1
    
    Te_i = a[node_i]
    Te_j = a[node_j]
    
    T_sum += (Te_i + Te_j) / 2
    
T_mean = T_sum / num_edges_air
print(f'\nNumber of DOFs: {num_dofs}')
print(f'Mean temperature: {T_mean:.3f}')

# Manual comparison between values when changing mesh_size
T_mean_before = 22.560 # 4mm mesh_size, 1199 DOFs
T_mean_after = 22.561 # 3mm mesh_size, 2168 DOFs
change = np.abs((T_mean_after - T_mean_before) / T_mean_after)
print(fr'\nChange: {change * 100:.4f} %')
# Roughly doubled the number of DOFs
# The change in mean temperature is way below arbitrary convergence criteria of 1% -> converged

#%%
#---------------------------------------------------------------------------------------------------
# Task 2 - e) Heat flux vectors + visualization
#---------------------------------------------------------------------------------------------------
new_subtask('Task 2 - e) Compute heat flux vectors + visualization')

q = np.zeros((num_el, 2))

for el in range(num_el):
    el_nodes = mesh.nodes[mesh.elements[el] - 1]
    qe = flow2t_qe(el_nodes, D, Ed[el, :])
    q[el, :] = qe

plot_vector_field(mesh.nodes, mesh.elements, q, title='Heat flux')

#%%
#---------------------------------------------------------------------------------------------------
# Task 2 - f) Convective heat inflow
#---------------------------------------------------------------------------------------------------
new_subtask('Task 2 - f) Convective heat inflow')

Q = 0

for edge in range(num_edges_w):
    
    node_i = conv_nodes_w[edge] - 1
    node_j = conv_nodes_w[edge + 1] - 1
    
    Te_i = a[node_i]
    Te_j = a[node_j]
    # print(Te_i, Te_j)
    
    Q += alpha_w * mesh_size * t * ((Te_i + Te_j) / 2 - T_w)

print(f'\nTotal heat flux: {Q}')

#%%
#---------------------------------------------------------------------------------------------------
# Task 2 - g) Required water temperature
#---------------------------------------------------------------------------------------------------
new_subtask('Task 2 - g) Required water temperature')

def FEA(T_w_var):
    # Initiate stiffness and load matrices
    K = np.zeros((num_dofs, num_dofs))
    f = np.zeros((num_dofs, 1))

    for el in range(num_el):
        el_nodes = mesh.nodes[mesh.elements[el] - 1]
        Ke, fe = flow2t_Ke_fe(el_nodes, D=D, t=t, Q=0)
        el_dofs = mesh.edofs[el, :]
        assem(K, Ke, el_dofs)
        assem(f, fe, el_dofs)

    for edge in range(num_edges_w):
        edge_nodes = conv_nodes_w[edge:edge + 2] - 1
        nodes = mesh.nodes[edge_nodes, :]
        Kec, fec = convection_Ke_fe(nodes, alpha=alpha_w, t=t, Tamb=T_w_var)
        dofs = edge_nodes + 1
        assem(K, Kec, dofs)
        assem(f, fec, dofs)

    for edge in range(num_edges_air):
        edge_nodes = conv_nodes_air[edge:edge + 2] - 1
        nodes = mesh.nodes[edge_nodes, :]
        Kec, fec = convection_Ke_fe(nodes, alpha=alpha_air, t=t, Tamb=T_air)
        dofs = edge_nodes + 1
        assem(K, Kec, dofs)
        assem(f, fec, dofs)

    # Essential boundary conditions
    bottom_dofs = mesh.edges['bottom']

    bc_dofs = bottom_dofs
    bc_vals = np.ones_like(bc_dofs) * T_b

    # Solve system
    a, r = solve_eq(K, f, bc_dofs, bc_vals)
    
    return a

# Idea: start with T_w_var = 30, and large T_step values
# Once a general value for T_w is found, optimize the T_w_var start value and step values

# Rough steps
T_sum = 0
T_mean = 0
T_w_var = 30 # starting value, optimized

while T_mean <=30:
    
    print(f'T_w = {T_w_var}')
    a = FEA(T_w_var)
    
    for edge in range(num_edges_air):
    
        node_i = conv_nodes_air[edge] - 1
        node_j = conv_nodes_air[edge + 1] - 1
        
        Te_i = a[node_i]
        Te_j = a[node_j]
        # print(Te_i, Te_j)
        
        T_sum += (Te_i + Te_j) / 2
        
    T_mean = T_sum / num_edges_air
    print(f'\nMean temperature: {T_mean:.3f}')
    
    T_step = 0.5
    T_w_var += T_step # temperature step, optimized
    T_sum = 0

# Refined steps
T_sum = 0
T_mean = 0
T_w_var = 46.45 # starting value, optimized

while T_mean <=30:
    
    print(f'T_w = {T_w_var}')
    a = FEA(T_w_var)
    
    for edge in range(num_edges_air):
    
        node_i = conv_nodes_air[edge] - 1
        node_j = conv_nodes_air[edge + 1] - 1
        
        Te_i = a[node_i]
        Te_j = a[node_j]
        # print(Te_i, Te_j)
        
        T_sum += (Te_i + Te_j) / 2
        
    T_mean = T_sum / num_edges_air
    print(f'\nMean temperature: {T_mean:.3f}')
    
    T_step = 0.001
    T_w_var += T_step # temperature step, optimized
    T_sum = 0
#%%