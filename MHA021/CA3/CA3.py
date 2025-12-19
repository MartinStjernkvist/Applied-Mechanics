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



# Task 1



####################################################################################################
####################################################################################################
####################################################################################################
new_task('Task 1')

#%%
#---------------------------------------------------------------------------------------------------
# (a)
#---------------------------------------------------------------------------------------------------
new_subtask('(b)')


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
fig = plot_scalar_field(mesh.nodes, mesh.elements, Ed, title=fr'Tempterature ($^\circ$C)')
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