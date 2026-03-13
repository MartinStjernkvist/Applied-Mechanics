#%%
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import scipy.sparse.linalg as spla
import scipy.io as sio

from scipy.linalg import eigh
from scipy.sparse import lil_matrix, issparse
from pathlib import Path
from matplotlib.collections import PolyCollection

# ------------------------------------------------------------------
# Utility print functions
# ------------------------------------------------------------------
def new_task(string):
    sep = '@' * 80
    print(f'\n{sep}\n{sep}\n{string}\n{sep}\n{sep}\n')

def new_subtask(string):
    sep = '=' * 80
    print(f'\n{sep}\n{string}\n{sep}\n')

def printt(string):
    sep = '*' * 80
    print(f'\n{sep}\n{string}\n{sep}\n')

# ------------------------------------------------------------------
# Plot settings
# ------------------------------------------------------------------
SMALL_SIZE, MEDIUM_SIZE, BIGGER_SIZE, dpi = 10, 12, 14, 500

plt.rc('font',   size=SMALL_SIZE)
plt.rc('axes',   titlesize=BIGGER_SIZE, labelsize=MEDIUM_SIZE)
plt.rc('xtick',  labelsize=SMALL_SIZE)
plt.rc('ytick',  labelsize=SMALL_SIZE)
plt.rc('legend', fontsize=SMALL_SIZE)
plt.rc('figure', titlesize=BIGGER_SIZE, figsize=(8, 6))

script_dir = Path(__file__).parent

def sfig(fig_name):
    out = script_dir / "figures_v2" / fig_name
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=dpi, bbox_inches='tight')
    print('figure name: ', fig_name)

# ------------------------------------------------------------------
# Generic field contour plot
# ------------------------------------------------------------------
def plot_field(field, fname, title, cbar_label, suptitle=''):
    fig, ax = plt.subplots()
    if suptitle:
        fig.suptitle(suptitle)
    # If field is nodal (nnodes,) average to elements; if already element-sized use directly
    el_vals = field if len(field) == len(node_idx) else field[node_idx].mean(axis=1)
    pc = PolyCollection(polygons, array=el_vals, cmap='coolwarm', edgecolors='none')
    ax.add_collection(pc)
    ax.autoscale()
    ax.set_aspect('equal')
    ax.set_title(title)
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    plt.colorbar(pc, ax=ax, label=cbar_label)
    sfig(fname)
    plt.show()

# ------------------------------------------------------------------
# DOF index helpers  (avoids repeated d_ip / d_oop blocks)
# ------------------------------------------------------------------
def get_dof_ip(nodes):
    d = np.empty(8, dtype=int)
    d[0::2] = 5 * nodes
    d[1::2] = 5 * nodes + 1
    return d

def get_dof_oop(nodes):
    d = np.empty(12, dtype=int)
    d[0::3] = 5 * nodes + 2
    d[1::3] = 5 * nodes + 3
    d[2::3] = 5 * nodes + 4
    return d

# ------------------------------------------------------------------
# Node-column helper (avoids repeated n1,n2,n3,n4 extraction)
# ------------------------------------------------------------------
def node_cols(ex, ey):
    return [np.array([[ex[i]], [ey[i]]]) for i in range(4)]

#%%
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
new_task('Task 1 - Linear elastic plate analysis using MATLAB or Python')
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# ------------------------------------------------------------------
# Inputs
# ------------------------------------------------------------------
g           = 9.81
h_snow      = 0.25          # m
angle_roof  = np.deg2rad(15)
W_roof      = 2             # m
L_roof      = 3             # m
rho_snow    = 500           # kg/m^3

Emod_al         = 80e9     # Pa
nu_al           = 0.2
sigma_yield_al  = 200e6    # Pa

# Gauss point data (2×2 rule)
GP   = 1 / np.sqrt(3)
xi_GP = np.array([[-GP, -GP], 
                  [GP,  -GP], 
                  [GP,   GP], 
                  [-GP,  GP]])
H_v  = np.ones(4)
xi_v = np.array([[-GP, -GP, GP,  GP],
                 [-GP,  GP, -GP, GP]])

h_plate = 8e-3  # m

#===================================================================================================
new_subtask('Task 1 d) - Code implementation')
#===================================================================================================

from kirchoff_funcs import bast_kirchoff_func
from N_kirchoff_func import N_kirchoff_func


def detFisop_4node_func(in1, in2, in3, in4, in5):
    inputs = [in1, in2, in3, in4, in5]
    in1, in2, in3, in4, in5 = [np.atleast_2d(x).reshape(2, -1) for x in inputs]

    xi1, xi2   = in1[0, :], in1[1, :]
    xe11, xe12 = in2[0, :], in2[1, :]
    xe21, xe22 = in3[0, :], in3[1, :]
    xe31, xe32 = in4[0, :], in4[1, :]
    xe41, xe42 = in5[0, :], in5[1, :]

    dN1_dxi1 = -0.25 * (1 - xi2);  dN2_dxi1 =  0.25 * (1 - xi2)
    dN3_dxi1 =  0.25 * (1 + xi2);  dN4_dxi1 = -0.25 * (1 + xi2)
    dN1_dxi2 = -0.25 * (1 - xi1);  dN2_dxi2 = -0.25 * (1 + xi1)
    dN3_dxi2 =  0.25 * (1 + xi1);  dN4_dxi2 =  0.25 * (1 - xi1)

    dx_dxi1 = dN1_dxi1*xe11 + dN2_dxi1*xe21 + dN3_dxi1*xe31 + dN4_dxi1*xe41
    dy_dxi1 = dN1_dxi1*xe12 + dN2_dxi1*xe22 + dN3_dxi1*xe32 + dN4_dxi1*xe42
    dx_dxi2 = dN1_dxi2*xe11 + dN2_dxi2*xe21 + dN3_dxi2*xe31 + dN4_dxi2*xe41
    dy_dxi2 = dN1_dxi2*xe12 + dN2_dxi2*xe22 + dN3_dxi2*xe32 + dN4_dxi2*xe42

    return dx_dxi1 * dy_dxi2 - dx_dxi2 * dy_dxi1


def Bu_and_detJ(xin, xe_nodes):
    xi1v, xi2v = np.atleast_1d(np.squeeze(xin))[:2]

    dN_dxi = 0.25 * np.array([
        [-(1 - xi2v),  (1 - xi2v),  (1 + xi2v), -(1 + xi2v)],
        [-(1 - xi1v), -(1 + xi1v),  (1 + xi1v),  (1 - xi1v)]
    ])

    J     = dN_dxi @ xe_nodes
    detJ  = float(np.linalg.det(J))
    dN_dx = np.linalg.inv(J) @ dN_dxi

    B_u = np.zeros((3, 8))
    B_u[0, 0::2] = dN_dx[0]
    B_u[1, 1::2] = dN_dx[1]
    B_u[2, 0::2] = dN_dx[1]
    B_u[2, 1::2] = dN_dx[0]

    return B_u, detJ


def Nu_inplane(xin):
    xi1v, xi2v = np.atleast_1d(np.squeeze(xin))[:2]

    Nv = np.array([
        0.25 * (1 - xi1v) * (1 - xi2v),
        0.25 * (1 + xi1v) * (1 - xi2v),
        0.25 * (1 + xi1v) * (1 + xi2v),
        0.25 * (1 - xi1v) * (1 + xi2v),
    ])

    N_u = np.zeros((2, 8))
    N_u[0, 0::2] = Nv
    N_u[1, 1::2] = Nv
    return N_u


def hooke_plane_stress(E, nu):
    return (E / (1 - nu**2)) * np.array([
        [1,  nu,          0],
        [nu,  1,          0],
        [0,   0, (1-nu)/2  ]
    ])


def kirchoff_plate_element(ex, ey, h, Dbar, q, f):
    K_uu = np.zeros((8, 8));   K_ww = np.zeros((12, 12))
    f_u  = np.zeros((8, 1));   f_w  = np.zeros((12, 1))

    xe_nodes = np.column_stack((ex, ey))
    n1, n2, n3, n4 = node_cols(ex, ey)

    for gp in range(4):
        xi_vec = xi_v[:, gp]
        xin    = xi_vec.reshape(2, 1)

        B_u, detJ = Bu_and_detJ(xi_vec, xe_nodes)
        N_u       = Nu_inplane(xi_vec)
        
        K_uu += B_u.T @ (h * D) @ B_u * detJ * H_v[gp]
        f_u  += N_u.T @ f * detJ * H_v[gp]

        detFisop    = detFisop_4node_func(xin, n1, n2, n3, n4)[0]
        N_w         = N_kirchoff_func(xin, n1, n2, n3, n4)
        _, Bastn, _ = bast_kirchoff_func(xin, n1, n2, n3, n4)
        
        f_w  += N_w.T * q * detFisop * H_v[gp]
        K_ww += Bastn.T @ Dbar @ Bastn * detFisop * H_v[gp]

    return K_uu, K_ww, f_u, f_w

#%%
#===================================================================================================
new_subtask('Task 1 e) - FE-program for plate analysis')
#===================================================================================================

# ------------------------------------------------------------------
# Mesh parameters
# ------------------------------------------------------------------
xmin, xmax = 0, 0.5
ymin, ymax = 0, 0.5 / np.cos(angle_roof)
nelx, nely = 20, 20
print(f'ymax = {ymax}')

# ------------------------------------------------------------------
# Read .mat file
# ------------------------------------------------------------------
data    = sio.loadmat('undeformedv4.mat', squeeze_me=True, struct_as_record=False)
mesh    = data['mesh']
Coord   = data['coord']
Edof_ip  = data['Edof_ip']
Edof_oop = data['Edof_oop']

# Convert sparse if needed and cast to int
for name in ['Edof_ip', 'Edof_oop']:
    arr = locals()[name]
    if issparse(arr): arr = arr.toarray()
    locals()[name]
Edof_ip  = Edof_ip.toarray().astype(int)  if issparse(Edof_ip)  else Edof_ip.astype(int)
Edof_oop = Edof_oop.toarray().astype(int) if issparse(Edof_oop) else Edof_oop.astype(int)

node_idx = (mesh.T - 1).astype(int)
Ex = Coord[node_idx, 0]
Ey = Coord[node_idx, 1]

# ------------------------------------------------------------------
# Plot undeformed mesh
# ------------------------------------------------------------------
polygons = np.dstack((Ex, Ey))

fig1, ax1 = plt.subplots()
ax1.add_collection(PolyCollection(polygons, facecolors='none', edgecolors='k', linewidths=1.2))
ax1.autoscale()
ax1.set_aspect('equal')
ax1.set_title('Undeformed mesh')
ax1.set_xlabel('x (m)')
ax1.set_ylabel('y (m)')
sfig('undeformed.png')
plt.show()

#%%
# ------------------------------------------------------------------
# Initialise matrices
# ------------------------------------------------------------------
nel          = mesh.shape[1]
nnodes       = Coord.shape[0]
ndofs        = 5 * nnodes

K_global = lil_matrix((ndofs, ndofs))
f_global = np.zeros(ndofs)
a        = np.zeros(ndofs)

D    = hooke_plane_stress(Emod_al, nu_al)
Dbar = (h_plate**3 / 12) * D

# ------------------------------------------------------------------
# Boundary conditions
# ------------------------------------------------------------------
tol = 1e-10
x_all, y_all = Coord[:, 0], Coord[:, 1]

gliding_nodes = np.where(np.abs(y_all - ymin) < tol)[0]
clamped_nodes = np.where(
    (np.abs(x_all - xmin) < tol) |
    (np.abs(x_all - xmax) < tol) |
    (np.abs(y_all - ymax) < tol))[0]

prescribed = set()
for n in clamped_nodes:
    for d in range(5):
        prescribed.add(5 * n + d)
for n in gliding_nodes:
    prescribed.update([5*n+2, 5*n+3, 5*n+4])

dof_C = np.array(sorted(prescribed), dtype=int)
dof_F = np.setdiff1d(np.arange(ndofs), dof_C)
a_C   = np.zeros(len(dof_C))

# ------------------------------------------------------------------
# Loads
# ------------------------------------------------------------------
q0     = rho_snow * g * h_snow
q_body = q0 * np.cos(angle_roof)
fy_bar = q0 * np.sin(angle_roof)
f_body = np.array([[0], [-fy_bar]])

# ------------------------------------------------------------------
# Assemble global system
# ------------------------------------------------------------------
for el in range(nel):
    ex, ey = Coord[node_idx[el, :], 0], Coord[node_idx[el, :], 1]
    K_uu, K_ww, f_u, f_w = kirchoff_plate_element(ex, ey, h_plate, Dbar, q=q_body, f=f_body)

    nodes = node_idx[el, :]
    d_ip  = get_dof_ip(nodes)
    d_oop = get_dof_oop(nodes)

    f_global[d_ip]                  += f_u[:, 0]
    f_global[d_oop]                 += f_w[:, 0]
    K_global[np.ix_(d_ip,  d_ip)]  += K_uu
    K_global[np.ix_(d_oop, d_oop)] += K_ww

K = K_global.tocsr()

# ------------------------------------------------------------------
# Solve
# ------------------------------------------------------------------
K_FF = K[np.ix_(dof_F, dof_F)]
f_F  = f_global[dof_F] - K[np.ix_(dof_F, dof_C)] @ a_C

a_F = np.atleast_1d(spla.spsolve(K_FF, f_F)).ravel()
a[dof_F] = a_F
a[dof_C] = a_C

u_x_nodes    = a[0::5]
u_y_nodes    = a[1::5]
w_nodes      = a[2::5]
theta_y_nodes = a[3::5]
theta_x_nodes = a[4::5]

w_max = np.abs(w_nodes).max()
print(f'w_max = {w_max * 1e3:.3f} mm')
if w_max * 1e3 <= 25:
    print('SATISFIED')

#%%
# ------------------------------------------------------------------
# Displacement contour plots
# ------------------------------------------------------------------
polygons = Coord[node_idx]

plot_field(w_nodes   * 1e3, 'task1_displacements_w.png',  'z-direction', 'w [mm]',   'Displacement')
plot_field(u_x_nodes * 1e3, 'task1_displacements_ux.png', 'x-direction', 'ux [mm]',  'Displacement')
plot_field(u_y_nodes * 1e3, 'task1_displacements_uy.png', 'y-direction', 'uy [mm]',  'Displacement')

# ------------------------------------------------------------------
# Stress at integration points
# ------------------------------------------------------------------
def von_mises_plane_stress(sigma):
    s11, s22, s12 = sigma
    return np.sqrt(s11**2 - s11*s22 + s22**2 + 3*s12**2)


def element_stress_at_z(ex, ey, a_u_el, a_w_el, z_coord):
    xe_nodes = np.column_stack((ex, ey))
    n1, n2, n3, n4 = node_cols(ex, ey)

    sigma_vM = np.zeros(4)
    for i in range(4):
        xin = xi_GP[i].reshape(2, 1)
        B_u, _   = Bu_and_detJ(xin, xe_nodes)
        _, Bast, _ = bast_kirchoff_func(xin, n1, n2, n3, n4)
        eps_z    = B_u @ a_u_el - z_coord * (Bast @ a_w_el)
        sigma_vM[i] = von_mises_plane_stress(D @ eps_z)

    return sigma_vM


z_levels   = [-h_plate / 2, 0, h_plate / 2]
vM_contour = {}

for z in z_levels:
    el_avg_vM = np.zeros(nel)
    for el in range(nel):
        nodes_el = node_idx[el, :]
        d_ip  = get_dof_ip(nodes_el)
        d_oop = get_dof_oop(nodes_el)
        ex, ey = Coord[nodes_el, 0], Coord[nodes_el, 1]
        el_avg_vM[el] = element_stress_at_z(ex, ey, a[d_ip], a[d_oop], z).mean()
    vM_contour[z] = el_avg_vM

#%%
# ------------------------------------------------------------------
# Von Mises contour plots
# ------------------------------------------------------------------
for z, tag, title in [
    (z_levels[0], 'task1_stress_vonMises_bottom.png', 'z = -h/2 (bottom)'),
    (z_levels[1], 'task1_stress_vonMises_middle.png', 'z = 0 (middle)'),
    (z_levels[2], 'task1_stress_vonMises_top.png',    'z = h/2 (top)'),
]:
    plot_field(vM_contour[z] / 1e6, tag, title, 'Von Mises stress [MPa]', 'Von Mises stress')

# ------------------------------------------------------------------
# Load vector verification
# ------------------------------------------------------------------
plate_area = xmax * ymax
print(f'Sum nodal w-forces  = {f_global[2::5].sum():.3f} N (expect {q_body * plate_area:.3f})')
print(f'Sum nodal uy-forces = {f_global[1::5].sum():.3f} N (expect {-fy_bar * plate_area:.3f})')

#%%
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
new_task('Task 2 - Buckling analysis (temperature elevation)')
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

delta_T  = 30       # degrees
alpha_al = 20e-6    # 1/K

# 3×3 Gauss rule for G^(R)
GP3      = np.sqrt(3 / 5)
xi_GP3x3 = np.array([
    [-GP3, -GP3], [0, -GP3], [GP3, -GP3],
    [-GP3,  0  ], [0,  0  ], [GP3,  0  ],
    [-GP3,  GP3], [0,  GP3], [GP3,  GP3],
])
W3    = np.array([5/9, 8/9, 5/9])
W3X3  = np.outer(W3, W3).ravel()

#===================================================================================================
new_subtask('Task 2 a) - Kirchoff plate bending element for linearised pre-buckling')
#===================================================================================================

def kirchhoff_buckling_element(ex, ey, Dbar, N_sec):
    n1, n2, n3, n4 = node_cols(ex, ey)

    K_K_ww = np.zeros((12, 12))
    for gp in range(4):
        xin      = xi_v[:, gp].reshape(2, 1)
        detFisop = detFisop_4node_func(xin, n1, n2, n3, n4)[0]
        _, Bastn, _ = bast_kirchoff_func(xin, n1, n2, n3, n4)
        K_K_ww  += Bastn.T @ Dbar @ Bastn * detFisop * H_v[gp]

    G_e_R = np.zeros((12, 12))
    for gp in range(9):
        xin      = xi_GP3x3[gp].reshape(2, 1)
        detFisop = detFisop_4node_func(xin, n1, n2, n3, n4)[0]
        dNdx, _, _ = bast_kirchoff_func(xin, n1, n2, n3, n4)
        G_e_R   += dNdx.T @ N_sec @ dNdx * detFisop * W3X3[gp]

    return K_K_ww, G_e_R

#===================================================================================================
new_subtask('Task 2 b) - Thermal in-plane load vector (element contribution)')
#===================================================================================================

def f_thermal_element(ex, ey, h, D, alpha, dT):
    xe_nodes = np.column_stack((ex, ey))
    eps_th   = alpha * dT * np.array([1, 1, 0])
    sigma_th = D @ eps_th
    f_th = np.zeros((8, 1))
    for gp in range(4):
        B_u, detJ = Bu_and_detJ(xi_v[:, gp], xe_nodes)
        f_th += (B_u.T @ sigma_th).reshape(8, 1) * h * detJ * H_v[gp]
    return f_th


def inplane_stress_at_point(xin, xe_nodes, a_u_el, D_mat, alpha, dT):
    B_u, _ = Bu_and_detJ(xin, xe_nodes)
    eps_th = alpha * dT * np.array([1, 1, 0])
    return D_mat @ (B_u @ a_u_el - eps_th)

#===================================================================================================
new_subtask('Task 2 c) - FE program for linearised pre-buckling')
#===================================================================================================

# ------------------------------------------------------------------
# Assemble in-plane stiffness and thermal load
# ------------------------------------------------------------------
K_uu_global = lil_matrix((ndofs, ndofs))
f_u_thermal = np.zeros(ndofs)

for el in range(nel):
    ex, ey = Coord[node_idx[el, :], 0], Coord[node_idx[el, :], 1]
    K_uu_e, _, _, _ = kirchoff_plate_element(ex, ey, h_plate, Dbar, q=0, f=np.zeros((2,1)))
    f_th_e          = f_thermal_element(ex, ey, h_plate, D, alpha_al, delta_T)

    d_ip = get_dof_ip(node_idx[el, :])
    f_u_thermal[d_ip]               += f_th_e[:, 0]
    K_uu_global[np.ix_(d_ip, d_ip)] += K_uu_e

K_uu_csr = K_uu_global.tocsr()

# ------------------------------------------------------------------
# Boundary conditions for thermal problem
# ------------------------------------------------------------------
prescribed_ip = set()
for n in clamped_nodes:
    prescribed_ip.update([5*n, 5*n+1])
for n in range(nnodes):
    prescribed_ip.update([5*n+2, 5*n+3, 5*n+4])

dof_C_ip = np.array(sorted(prescribed_ip), dtype=int)
dof_F_ip = np.setdiff1d(np.arange(ndofs), dof_C_ip)

a_u   = np.zeros(ndofs)
a_u[dof_F_ip] = spla.spsolve(
    K_uu_csr[np.ix_(dof_F_ip, dof_F_ip)], f_u_thermal[dof_F_ip])

u_x_th, u_y_th = a_u[0::5], a_u[1::5]
print(f'Max ux = {np.abs(u_x_th).max() * 1e3:.4f} mm')
print(f'Max uy = {np.abs(u_y_th).max() * 1e3:.4f} mm')

# ------------------------------------------------------------------
# Compute N_sec for each element
# ------------------------------------------------------------------
xin_mid  = np.array([0.0, 0.0])
N_sec_all = []

for el in range(nel):
    nodes_el = node_idx[el, :]
    ex, ey   = Coord[nodes_el, 0], Coord[nodes_el, 1]
    d_ip     = get_dof_ip(nodes_el)
    sigma    = inplane_stress_at_point(xin_mid, np.column_stack((ex, ey)),
                                       a_u[d_ip], D, alpha_al, delta_T)
    sig_xx, sig_yy, sig_xy = sigma
    N_sec_all.append(h_plate * np.array([[sig_xx, sig_xy], [sig_xy, sig_yy]]))

# ------------------------------------------------------------------
# Assemble K_ww and G^(R)
# ------------------------------------------------------------------
K_Kww_global = lil_matrix((ndofs, ndofs))
G_R_global   = lil_matrix((ndofs, ndofs))

for el in range(nel):
    ex, ey = Coord[node_idx[el, :], 0], Coord[node_idx[el, :], 1]
    K_K_ww_e, G_e_R = kirchhoff_buckling_element(ex, ey, Dbar, N_sec_all[el])

    d_oop = get_dof_oop(node_idx[el, :])
    K_Kww_global[np.ix_(d_oop, d_oop)] += K_K_ww_e
    G_R_global[np.ix_(d_oop,   d_oop)] += G_e_R

K_Kww_csr = K_Kww_global.tocsr()
G_R_csr   = G_R_global.tocsr()

# ------------------------------------------------------------------
# Solve eigenvalue problem
# ------------------------------------------------------------------
prescribed_oop = set()
for n in clamped_nodes:
    prescribed_oop.update([5*n+2, 5*n+3, 5*n+4])
for n in gliding_nodes:
    prescribed_oop.update([5*n+2, 5*n+3, 5*n+4])

dof_C_oop = np.array(sorted(prescribed_oop), dtype=int)
all_oop   = np.sort(np.concatenate([np.arange(2, ndofs, 5),
                                     np.arange(3, ndofs, 5),
                                     np.arange(4, ndofs, 5)]))
dof_F_oop = np.setdiff1d(all_oop, dof_C_oop)

K_FF_bck = K_Kww_csr[np.ix_(dof_F_oop, dof_F_oop)].toarray()
G_FF_bck = G_R_csr  [np.ix_(dof_F_oop, dof_F_oop)].toarray()

n_eig = min(10, K_FF_bck.shape[0])
eigvals, eigvecs = eigh(K_FF_bck, -G_FF_bck, subset_by_index=[0, n_eig - 1])

pos_mask = eigvals > 0
if pos_mask.any():
    lam_min = eigvals[pos_mask][0]
    z_min   = eigvecs[:, pos_mask][:, 0]

printt(f'\nSmallest positive eigenvalue (SF) = {lam_min:.4f}')

# ------------------------------------------------------------------
# Plot buckling mode shape
# ------------------------------------------------------------------
z_full = np.zeros(ndofs)
z_full[dof_F_oop] = z_min
w_mode = z_full[2::5]
w_mode /= np.abs(w_mode).max() if np.abs(w_mode).max() > 0 else 1

polygons_plot = Coord[node_idx]
fig, ax = plt.subplots()
pc = PolyCollection(polygons_plot, array=w_mode[node_idx].mean(axis=1),
                    cmap='coolwarm', edgecolors='none')
ax.add_collection(pc)
ax.autoscale()
ax.set_aspect('equal')
ax.set_title(f'Buckling mode 1 (eigen value = {lam_min:.3f})')
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
plt.colorbar(pc, ax=ax, label='Normalised w [mm]')
sfig('task2_buckling_mode.png')
plt.show()

# ------------------------------------------------------------------
# Thermal displacement plots  (reuse generic plot_field)
# ------------------------------------------------------------------
plot_field(u_x_th * 1e3, 'task2_thermal_displacements_uxth.png',
           'x-direction', 'ux [mm]', 'In-plane thermal displacement')
plot_field(u_y_th * 1e3, 'task2_thermal_displacements_uyth.png',
           'y-direction', 'uy [mm]', 'In-plane thermal displacement')
#%%