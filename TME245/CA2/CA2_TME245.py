#%%
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import scipy.sparse.linalg as spla
from scipy.linalg import eigh
import scipy.io as sio
from pathlib import Path
from matplotlib.collections import PolyCollection
from scipy.sparse import lil_matrix
import scipy.sparse.linalg as spla

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

#%%  
#===================================================================================================
####################################################################################################
####################################################################################################
####################################################################################################

new_task('Task 1 - Linear elastic plate analysis using MATLAB or Python')

####################################################################################################
####################################################################################################
####################################################################################################
#===================================================================================================
        
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# Define inputs
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
g = 9.81

# Geometry
h_snow = 0.25 # m
angle_roof = np.deg2rad(15) # rad
W_roof = 2 # m
L_roof = 3 # m
h_plate = 0.001

# Material properties aluminium
Emod_al = 80e9 # Pa
nu_al = 0.2
sigma_yield_al = 200e6 # Pa
alpha_al = 20e-6 # 1/K  

rho_snow = 500 # kmg/m^3
q0 = rho_snow * g * h_snow
q_bar  = q0 * np.cos(angle_roof)**2
fx_bar = q0 * np.cos(angle_roof) * np.sin(angle_roof)

print(f'\nPlate thickness: {h_plate*1e3:.1f} mm')

GP = 1.0 / np.sqrt(3.)
xi_GP = np.array([[-GP,-GP], [GP,-GP], [GP,GP], [-GP,GP]])

H_v  = np.ones(4)
xi_v = np.array([
        [-1 / np.sqrt(3), -1 / np.sqrt(3), 1 / np.sqrt(3), 1 / np.sqrt(3)],
        [-1 / np.sqrt(3), 1 / np.sqrt(3), -1 / np.sqrt(3), 1 / np.sqrt(3)]
        ])
        
#===================================================================================================
####################################################################################################
new_subtask('Task 1 d) - Code implementation')
####################################################################################################
#===================================================================================================

from kirchoff_funcs import bast_kirchoff_func
from N_kirchoff_func import N_kirchoff_func

def detFisop_4node_func(in1, in2, in3, in4, in5):
    
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

    xin_flat = np.atleast_1d(np.squeeze(xin))
    xi1v, xi2v = xin_flat[0], xin_flat[1]
    
    dN_dxi = 0.25 * np.array([
        [-(1 - xi2v), (1 - xi2v), (1 + xi2v), -(1 + xi2v)],
        [-(1 - xi1v), -(1 + xi1v), (1 + xi1v), (1 -xi1v)]
    ])
    
    J = dN_dxi @ xe_nodes
    detJ = float(np.linalg.det(J))
    dN_dx = np.linalg.inv(J) @ dN_dxi

    B_u = np.zeros((3, 8))
    for k in range(4):
        B_u[0, 2 * k] = dN_dx[0, k]
        B_u[1, 2 * k+1] = dN_dx[1, k]
        B_u[2, 2 * k] = dN_dx[1, k]
        B_u[2, 2 * k + 1] = dN_dx[0, k]
        
    return B_u, detJ


def Nu_inplane(xin):
    
    xin_flat = np.atleast_1d(np.squeeze(xin))
    xi1v, xi2v = xin_flat[0], xin_flat[1]
    
    Nv = [0.25 * (1 - xi1v) * (1 - xi2v),
          0.25 * (1 + xi1v) * (1 - xi2v),
          0.25 * (1 + xi1v) * (1 + xi2v),
          0.25 * (1 - xi1v) * (1 + xi2v)]
    
    N_u = np.zeros((2, 8))
    for k, Nk in enumerate(Nv):
        N_u[0, 2 * k] = Nk
        N_u[1, 2 * k + 1] = Nk
        
    return N_u


def hooke_plane_stress(E, nu):
    return (E / (1 - nu**2)) * np.array([
        [1, nu, 0],
        [nu, 1, 0],
        [0, 0, (1 - nu) / 2]
    ])


def assem(edof, K, Ke, f, fe):
    idx = edof - 1 # Convert to 0-based indexing
    for i in range(len(idx)):
        f[idx[i], 0] += fe[i, 0]
        for j in range(len(idx)):
            K[idx[i], idx[j]] += Ke[i, j]
    return K, f

# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# Plate element function
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

def kirchoff_plate_element(ex, ey, h, Dbar, body_val):
    
    K_uu = np.zeros((8, 8))
    K_ww = np.zeros((12, 12))
    f_u  = np.zeros((8, 1))
    f_w  = np.zeros((12, 1))
    
    n1 = np.array([[ex[0]], [ey[0]]])
    n2 = np.array([[ex[1]], [ey[1]]])
    n3 = np.array([[ex[2]], [ey[2]]])
    n4 = np.array([[ex[3]], [ey[3]]])
    
    xe_nodes = np.column_stack((ex, ey))
    
    for gp in range(4):
        Hgp = 1
        
        xi_vec = xi_v[:, gp]
        xin = xi_vec.reshape(2, 1)
        
        # In-plane membrane
        B_u, detJ = Bu_and_detJ(xi_vec, xe_nodes)
        N_u = Nu_inplane(xi_vec)
        
        fy_bar = q0 * np.cos(angle_roof) * np.sin(angle_roof)
        f_body = np.array([[0.0], [-fy_bar]])

        K_uu += B_u.T @ (h * D) @ B_u * detJ * Hgp
        f_u  += N_u.T @ f_body * detJ * Hgp
        
        # Bending
        detFisop = detFisop_4node_func(xin, n1, n2, n3, n4)[0]
        N_w = N_kirchoff_func(xin, n1, n2, n3, n4)
        dNdx, Bastn, _ = bast_kirchoff_func(xin, n1, n2, n3, n4)
        
        f_w += (N_w.T * body_val * detFisop * Hgp)
        K_ww += (Bastn.T @ Dbar @ Bastn * detFisop * Hgp)
        
    return K_uu, K_ww, f_u, f_w
#%%
#===================================================================================================
####################################################################################################
new_subtask('Task 1 e) - Small FE-program for plate analysis')
####################################################################################################
#===================================================================================================

# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# Mesh parameters
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
xmin = 0
xmax = 0.5
ymin = 0
ymax = 0.75
nelx = 20
nely = 30

"""
xmin = 0;   xmax = 0.5;
ymin = 0;   ymax = 0.75;

nelx = 20;
nely = 30;

[mesh, coord, Edof_ip, Edof_oop] = rectMesh(xmin, xmax, ymin, ymax, nelx, nely);

save('testfil','-v7');
"""
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# Read matlab file
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
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
node_idx = (mesh.T - 1).astype(int)
Ex = Coord[node_idx, 0]
Ey = Coord[node_idx, 1]

# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# Plot undeformed mesh
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
polygons = np.dstack((Ex, Ey))

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
ax1.set_title('Undeformed mesh')
ax1.set_xlabel('x (m)')
ax1.set_ylabel('y (m)')
plt.grid(True, alpha=0.3)
plt.show()

# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# Initialize sparse global matrices
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
nel = mesh.shape[1]
nnodes = Coord.shape[0]
dofs_per_node = 5 # u_x, u_y, w, theta_y, theta_x
ndofs = dofs_per_node * nnodes

K_global = lil_matrix((ndofs, ndofs)) 
f_global = np.zeros(ndofs)
a = np.zeros(ndofs)

# Consitutive matrix
D = hooke_plane_stress(Emod_al, nu_al)
Dbar = (h_plate**3 / 12) * D

# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# Boundary conditions - FIXED
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
tol = 1e-10
x_all = Coord[:, 0]
y_all = Coord[:, 1]

gliding_nodes = np.where(np.abs(y_all - ymin) < tol)[0]
clamped_nodes = np.where(
    (np.abs(x_all - xmin) < tol) |
    (np.abs(x_all - xmax) < tol) |
    (np.abs(y_all - ymax) < tol)
    )[0]

prescribed = set()

# Clamped: all 5 DOFs
for n in clamped_nodes:
    for d in range(5):
        prescribed.add(5 * n + d)

# Gliding
for n in gliding_nodes:
    prescribed.add(5 * n + 2) # w
    prescribed.add(5 * n + 3) # theta_y
    prescribed.add(5 * n + 4) # theta_x

dof_C = np.array(sorted(prescribed), dtype=int)
dof_F = np.setdiff1d(np.arange(ndofs), dof_C)
a_C = np.zeros(len(dof_C))

body = q_bar

# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# Setup and solve FE equations
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
for el in range(nel):
    
    K_uu, K_ww, f_u, f_w = kirchoff_plate_element(Ex[el, :], Ey[el, :], h_plate, Dbar, body)
    
    nodes = node_idx[el, :] 
    
    # Map membrane DOFs
    d_ip = np.empty(8, dtype=int)
    d_ip[0::2] = 5 * nodes + 0  # ux
    d_ip[1::2] = 5 * nodes + 1  # uy
    
    # Map bending DOFs
    d_oop = np.empty(12, dtype=int)
    d_oop[0::3] = 5 * nodes + 2  # w
    d_oop[1::3] = 5 * nodes + 3  # theta_y
    d_oop[2::3] = 5 * nodes + 4  # theta_x
    
    # Assembly for membrane
    for i in range(8):
        f_global[d_ip[i]] += f_u[i, 0]
        for j in range(8):
            K_global[d_ip[i], d_ip[j]] += K_uu[i, j]
            
    # Assembly for bending
    for i in range(12):
        f_global[d_oop[i]] += f_w[i, 0]
        for j in range(12):
            K_global[d_oop[i], d_oop[j]] += K_ww[i, j]

# Convert to CSR format
K = K_global.tocsr()

# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# Block partitioning and solve for free displacements
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
K_FF = K[np.ix_(dof_F, dof_F)]
K_FC = K[np.ix_(dof_F, dof_C)]
K_CF = K[np.ix_(dof_C, dof_F)]
K_CC = K[np.ix_(dof_C, dof_C)]

f_F = f_global[dof_F] - K_FC @ a_C

a_F = spla.spsolve(K_FF, f_F)
if a_F.ndim == 0:
    a_F = np.array([a_F])
a_F = np.atleast_1d(a_F).ravel()

# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# Calculate reaction forces
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

# Eq (7.146)
r_C = K_CF @ a_F + K_CC @ a_C - f_global[dof_C]

a[dof_F] = a_F
a[dof_C] = a_C

# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# Extract displacement fields
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
w_nodes  = a[2::5]
u_x_nodes = a[0::5]
u_y_nodes = a[1::5]
theta_y_nodes = a[3::5]
theta_x_nodes = a[4::5]

w_max = np.abs(w_nodes).max()

print(f'w_max = {w_max * 1e3:.3f} mm')
if w_max * 1e3 <= 25.0:
    print('SATISFIED')
else:
    print('VIOLATED')

#%%
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# Displacement contour plots
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
def element_average(nodal_field):
    return nodal_field[node_idx].mean(axis=1)

polygons = Coord[node_idx]

fig, ax = plt.subplots(figsize=(8, 6))
fig.suptitle(f'Displacement')
el_vals = element_average(w_nodes * 1e3)
pc = PolyCollection(polygons, array=el_vals, cmap='coolwarm', edgecolors='none')
ax.add_collection(pc)
ax.autoscale()
ax.set_aspect('equal')
ax.set_title('w [mm]')
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
plt.colorbar(pc, ax=ax)
sfig('task1_displacements_w.png')
plt.show()

fig, ax = plt.subplots(figsize=(8, 6))
fig.suptitle(f'Displacement')
el_vals = element_average(u_x_nodes * 1e3)
pc = PolyCollection(polygons, array=el_vals, cmap='coolwarm', edgecolors='none')
ax.add_collection(pc)
ax.autoscale()
ax.set_aspect('equal')
ax.set_title('ux [mm]')
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
plt.colorbar(pc, ax=ax)
sfig('task1_displacements_ux.png')
plt.show()

fig, ax = plt.subplots(figsize=(8, 6))
fig.suptitle(f'Displacement')
el_vals = element_average(u_y_nodes * 1e3)
pc = PolyCollection(polygons, array=el_vals, cmap='coolwarm', edgecolors='none')
ax.add_collection(pc)
ax.autoscale()
ax.set_aspect('equal')
ax.set_title('uy [mm]')
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
plt.colorbar(pc, ax=ax)
sfig('task1_displacements_uy.png')
plt.show()

# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# Stress computation at integration points
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
def von_mises_plane_stress(sigma):
    s11, s22, s12 = sigma
    sigma_vm = np.sqrt(s11**2 - s11 * s22 + s22**2 + 3.0 * s12**2)
    return sigma_vm

def element_stress_at_z(ex, ey, a_u_el, a_w_el, z_coord):
        
    xe_nodes = np.column_stack((ex, ey))
    n1=np.array([[ex[0]], [ey[0]]])
    n2=np.array([[ex[1]], [ey[1]]])
    n3=np.array([[ex[2]], [ey[2]]])
    n4=np.array([[ex[3]], [ey[3]]])

    sigma_vM = np.zeros(4)
    for i in range(4):
        xi1v, xi2v = xi_GP[i]
        xin = np.array([[xi1v],[xi2v]])

        # Membrane strain
        B_u, _ = Bu_and_detJ(xin, xe_nodes)
        eps0   = B_u @ a_u_el

        # Curvature
        _, Bast, _ = bast_kirchoff_func(xin, n1, n2, n3, n4)
        kappa = Bast @ a_w_el

        # Total strain at depth z
        eps_z  = eps0 - z_coord * kappa

        # Stress and von Mises (remark 7.16)
        sigma = D @ eps_z
        sigma_vM[i] = von_mises_plane_stress(sigma)

    return sigma_vM


z_levels = [-h_plate / 2, 0, h_plate / 2]
vM_contour = {}

for z in z_levels:
    el_avg_vM = np.zeros(nel)
    for el in range(nel):
        ex = Coord[node_idx[el,:], 0]
        ey = Coord[node_idx[el,:], 1]

        nodes_el = node_idx[el, :]
        d_ip = np.empty(8, dtype=int)
        d_oop = np.empty(12, dtype=int)
        
        d_ip[0::2] = 5 * nodes_el + 0 # ux
        d_ip[1::2] = 5 * nodes_el + 1 # uy
        d_oop[0::3] = 5 * nodes_el + 2 # w
        d_oop[1::3] = 5 * nodes_el + 3 # theta_y
        d_oop[2::3] = 5 * nodes_el + 4 # theta_x

        a_u_el = a[d_ip]
        a_w_el = a[d_oop]

        svm = element_stress_at_z(ex, ey, a_u_el, a_w_el, z)
        el_avg_vM[el] = svm.mean()

    vM_contour[z] = el_avg_vM

#%%
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# Von Mises contour plots
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
fig, ax = plt.subplots(figsize=(8, 6))
fig.suptitle(f'Von Mises effective stress [MPa]')
el_vals = vM_contour[z_levels[0]] / 1e6
pc = PolyCollection(polygons, array=el_vals, cmap='coolwarm', edgecolors='none')
ax.add_collection(pc)
ax.autoscale()
ax.set_aspect('equal')
ax.set_title('z = -h/2 (bottom)')
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
plt.colorbar(pc, ax=ax)
sfig('task1_stress_vonMises_bottom.png')
plt.show()

fig, ax = plt.subplots(figsize=(8, 6))
fig.suptitle(f'Von Mises effective stress [MPa]')
el_vals = vM_contour[z_levels[1]] / 1e6
pc = PolyCollection(polygons, array=el_vals, cmap='coolwarm', edgecolors='none')
ax.add_collection(pc)
ax.autoscale()
ax.set_aspect('equal')
ax.set_title('z = 0 (middle)')
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
plt.colorbar(pc, ax=ax)
sfig('task1_stress_vonMises_middle.png')
plt.show()

fig, ax = plt.subplots(figsize=(8, 6))
fig.suptitle(f'Von Mises effective stress [MPa]')
el_vals = vM_contour[z_levels[2]] / 1e6
pc = PolyCollection(polygons, array=el_vals, cmap='coolwarm', edgecolors='none')
ax.add_collection(pc)
ax.autoscale()
ax.set_aspect('equal')
ax.set_title('z = h/2 (top)')
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
plt.colorbar(pc, ax=ax)
sfig('task1_stress_vonMises_top.png')
plt.show()

# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# Load vector verification
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
plate_area = xmax * ymax

w_dof_mask  = np.arange(2, ndofs, 5)
uy_dof_mask = np.arange(1, ndofs, 5)

sum_fw  = f_global[w_dof_mask].sum()
sum_fuy = f_global[uy_dof_mask].sum()

print(f'Sum nodal w-forces = {sum_fw:.4f} N (expect {q_bar * plate_area:.3f})')
print(f'\nSum nodal ux-forces = {sum_fuy:.3f} N (expect {fx_bar * plate_area:.3f}')

#%%  
#===================================================================================================
####################################################################################################
####################################################################################################
####################################################################################################

new_task('Task 2 - Buckling analysis (temperature elevation)')

####################################################################################################
####################################################################################################
####################################################################################################
#===================================================================================================

# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# Define inputs
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
delta_T  = 30.0 # degrees

# Gauss points used in G^(R)
GP3 = np.sqrt(3.0 / 5.0)
xi_GP3x3 = np.array([
    [-GP3, -GP3], [0.0, -GP3], [GP3, -GP3],
    [-GP3, 0.0], [0.0, 0.0], [GP3, 0.0 ],
    [-GP3, GP3], [0.0, GP3], [GP3, GP3],
])
W3 = np.array([5.0/9, 8.0/9, 5.0/9])
W3X3 = np.outer(W3, W3).ravel()

#===================================================================================================
####################################################################################################
new_subtask('Task 2 b) - Thermal in-plane load vector (element contribution)')
####################################################################################################
#===================================================================================================

def f_thermal_element(ex, ey, h, E, nu, alpha, dT):
    D_mat  = (E / (1.0 - nu**2)) * np.array([
        [1, nu, 0],
        [nu, 1, 0],
        [0, 0, (1 - nu) / 2]
    ])
    
    eps_th = alpha * dT * np.array([1, 1, 0])
    sigma_th = D_mat @ eps_th

    xe_nodes = np.column_stack((ex, ey))

    f_th = np.zeros((8, 1))
    for gp in range(4):
        xin = xi_v[:, gp]
        B_u, detJ = Bu_and_detJ(xin, xe_nodes)
        f_th += (B_u.T @ sigma_th).reshape(8, 1) * h * detJ * H_v[gp]

    return f_th

def kirchhoff_buckling_element(ex, ey, h, Dbar, N_sec):
    
    n1 = np.array([[ex[0]], [ey[0]]])
    n2 = np.array([[ex[1]], [ey[1]]])
    n3 = np.array([[ex[2]], [ey[2]]])
    n4 = np.array([[ex[3]], [ey[3]]])

    K_K_ww = np.zeros((12, 12))
    for gp in range(4):
        xin = xi_v[:, gp].reshape(2, 1)
        
        detFisop = detFisop_4node_func(xin, n1, n2, n3, n4)[0]
        _, Bastn, _ = bast_kirchoff_func(xin, n1, n2, n3, n4)
        K_K_ww += Bastn.T @ Dbar @ Bastn * detFisop * H_v[gp]

    G_e_R = np.zeros((12, 12))
    for gp in range(9):
        xin   = xi_GP3x3[gp].reshape(2, 1)
        W_gp  = W3X3[gp]
        
        detFisop = detFisop_4node_func(xin, n1, n2, n3, n4)[0]
        dNdx, _, _ = bast_kirchoff_func(xin, n1, n2, n3, n4)
        G_e_R += dNdx.T @ N_sec @ dNdx * detFisop * W_gp

    return K_K_ww, G_e_R


def inplane_stress_at_point(xin, xe_nodes, a_u_el, D_mat, alpha, dT):
    B_u, _ = Bu_and_detJ(xin, xe_nodes)
    eps = B_u @ a_u_el
    eps_th = alpha * dT * np.array([1, 1, 0])
    sigma = D_mat @ (eps - eps_th)
    return sigma

#===================================================================================================
####################################################################################################
new_subtask('Task 2 c) - Linearised pre-buckling FE program')
####################################################################################################
#===================================================================================================

# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# In-plane thermal problem
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
K_uu_global = lil_matrix((ndofs, ndofs))
f_u_thermal = np.zeros(ndofs)

D_mat = hooke_plane_stress(Emod_al, nu_al)

for el in range(nel):
    ex = Coord[node_idx[el, :], 0]
    ey = Coord[node_idx[el, :], 1]

    # Element in-plane stiffness
    K_uu_e, _, _, _ = kirchoff_plate_element(ex, ey, h_plate, Dbar, body_val=0)

    # Element thermal load
    f_th_e = f_thermal_element(ex, ey, h_plate, Emod_al, nu_al, alpha_al, delta_T)

    # DOF mapping (in-plane only)
    nodes = node_idx[el, :]
    d_ip = np.empty(8, dtype=int)
    d_ip[0::2] = 5 * nodes + 0
    d_ip[1::2] = 5 * nodes + 1

    # Assemble
    for i in range(8):
        f_u_thermal[d_ip[i]] += f_th_e[i, 0]
        for j in range(8):
            K_uu_global[d_ip[i], d_ip[j]] += K_uu_e[i, j]

K_uu_csr = K_uu_global.tocsr()

# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# In-plane boundary conditions for thermal problem
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
prescribed_ip = set()

for n in clamped_nodes:
    prescribed_ip.add(5 * n + 0) # u_x
    prescribed_ip.add(5 * n + 1) # u_y

for n in range(nnodes):
    prescribed_ip.add(5 * n + 2) # w
    prescribed_ip.add(5 * n + 3) # theta_y
    prescribed_ip.add(5 * n + 4) # theta_x

dof_C_ip = np.array(sorted(prescribed_ip), dtype=int)
dof_F_ip = np.setdiff1d(np.arange(ndofs), dof_C_ip)

K_FF_ip = K_uu_csr[np.ix_(dof_F_ip, dof_F_ip)]
f_F_ip  = f_u_thermal[dof_F_ip]

a_u_F = spla.spsolve(K_FF_ip, f_F_ip)

a_u = np.zeros(ndofs)
a_u[dof_F_ip] = a_u_F

u_x_th = a_u[0::5]
u_y_th = a_u[1::5]

print(f'Max ux = {np.abs(u_x_th).max() * 1e3:.4f} mm')
print(f'Max uy = {np.abs(u_y_th).max() * 1e3:.4f} mm')

# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# Compute N_sec for each element
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
N_sec_all = []

xin_mid = np.array([[0.0], [0.0]])

for el in range(nel):
    ex = Coord[node_idx[el, :], 0]
    ey = Coord[node_idx[el, :], 1]
    xe_nodes = np.column_stack((ex, ey))

    nodes  = node_idx[el, :]
    d_ip   = np.empty(8, dtype=int)
    d_ip[0::2] = 5 * nodes + 0
    d_ip[1::2] = 5 * nodes + 1
    a_u_el = a_u[d_ip]

    # In-plane stress at midpoint (including thermal strain)
    sigma = inplane_stress_at_point(xin_mid.ravel(), xe_nodes,
                                    a_u_el, D_mat, alpha_al, delta_T)
    sig_xx, sig_yy, sig_xy = sigma

    N_sec = h_plate * np.array([
        [sig_xx, sig_xy],
        [sig_xy, sig_yy]
    ])
    N_sec_all.append(N_sec)

# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# Assemble K_K_ww and G^(R)
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
K_Kww_global = lil_matrix((ndofs, ndofs))
G_R_global   = lil_matrix((ndofs, ndofs))

for el in range(nel):
    ex = Coord[node_idx[el, :], 0]
    ey = Coord[node_idx[el, :], 1]

    K_K_ww_e, G_e_R = kirchhoff_buckling_element(ex, ey, h_plate, Dbar, N_sec_all[el])

    nodes  = node_idx[el, :]
    d_oop  = np.empty(12, dtype=int)
    d_oop[0::3] = 5 * nodes + 2 # w
    d_oop[1::3] = 5 * nodes + 3 # theta_x
    d_oop[2::3] = 5 * nodes + 4 # theta_y

    for i in range(12):
        for j in range(12):
            K_Kww_global[d_oop[i], d_oop[j]] += K_K_ww_e[i, j]
            G_R_global  [d_oop[i], d_oop[j]] += G_e_R   [i, j]

K_Kww_csr = K_Kww_global.tocsr()
G_R_csr   = G_R_global.tocsr()

# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# Eigenvalue problem
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
prescribed_oop = set()
for n in clamped_nodes:
    for d in [2, 3, 4]:
        prescribed_oop.add(5 * n + d)
for n in gliding_nodes:
    prescribed_oop.add(5 * n + 2) # w
    prescribed_oop.add(5 * n + 3) # theta_x
    prescribed_oop.add(5 * n + 4) # theta_y

dof_C_oop = np.array(sorted(prescribed_oop), dtype=int)
all_oop_dofs = np.sort(np.concatenate([
    np.arange(2, ndofs, 5), # w
    np.arange(3, ndofs, 5), # theta_y
    np.arange(4, ndofs, 5), # theta_x
]))
dof_F_oop = np.setdiff1d(all_oop_dofs, dof_C_oop)


K_FF_bck = K_Kww_csr[np.ix_(dof_F_oop, dof_F_oop)].toarray()
G_FF_bck = G_R_csr[np.ix_(dof_F_oop, dof_F_oop)].toarray()

# Solve with eigh (symmetric) asking for the smallest eigenvalue of K w.r.t. -G.
neg_G = -G_FF_bck

# subset_by_index to get only the first few eigenvalues
n_eig = min(10, K_FF_bck.shape[0])
eigvals, eigvecs = eigh(K_FF_bck, neg_G, subset_by_index=[0, n_eig - 1])

# Filter for positive eigenvalues
pos_mask = eigvals > 0
if pos_mask.any():
    lam_min = eigvals[pos_mask][0]
    z_min   = eigvecs[:, pos_mask][:, 0]
else:
    pass

print(f'\nSmallest positive eigenvalue (SF) = {lam_min:.4f}')
    
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# Plot buckling mode shape
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
z_full = np.zeros(ndofs)
z_full[dof_F_oop] = z_min

# Extract only out-of-plane DOFs per node for plotting
w_mode = z_full[2::5]

# Normalise for visualisation
w_mode /= np.abs(w_mode).max() if np.abs(w_mode).max() > 0 else 1.0

polygons_plot = Coord[node_idx]

fig, ax = plt.subplots(figsize=(8, 6))
el_w = w_mode[node_idx].mean(axis=1)
pc = PolyCollection(polygons_plot, array=el_w, cmap='coolwarm', edgecolors='none')
ax.add_collection(pc)
ax.autoscale()
ax.set_aspect('equal')
ax.set_title(f'Buckling mode 1 (eigen value = {lam_min:.3f})')
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
plt.colorbar(pc, ax=ax, label='Normalised w (buckling mode)')
sfig('task2_buckling_mode.png')
plt.show()

fig, ax = plt.subplots(figsize=(8, 6))
fig.suptitle(f'In-plane thermal displacements')
field = u_x_th * 1e3
el_vals = field[node_idx].mean(axis=1)
pc = PolyCollection(polygons_plot, array=el_vals, cmap='coolwarm', edgecolors='none')
ax.add_collection(pc)
ax.autoscale()
ax.set_aspect('equal')
ax.set_title('ux [mm]')
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
plt.colorbar(pc, ax=ax)
sfig('task2_thermal_displacements_uxth.png')
plt.show()

fig, ax = plt.subplots(figsize=(8, 6))
fig.suptitle(f'In-plane thermal displacements')
field = u_y_th * 1e3
el_vals = field[node_idx].mean(axis=1)
pc = PolyCollection(polygons_plot, array=el_vals, cmap='coolwarm', edgecolors='none')
ax.add_collection(pc)
ax.autoscale()
ax.set_aspect('equal')
ax.set_title('uy [mm]')
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
plt.colorbar(pc, ax=ax)
sfig('task2_thermal_displacements_uxth.png')
plt.show()
#%%