#%%
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import pandas as pd
import sys
import os
import scipy.sparse.linalg as spla
import scipy.sparse as sps
import scipy.io as sio

from typing import Optional
from IPython.display import display, Math
from scipy.sparse import coo_matrix, csr_matrix
from pathlib import Path
from matplotlib.collections import PolyCollection

# Add parent directory to sys.path
parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# ------------------------------------------------------------------
# Utility print functions
# ------------------------------------------------------------------
def new_task(string):
    sep = '=' * 80
    print(f'\n{sep}\n{sep}\n{string}\n{sep}\n{sep}\n')

def new_subtask(string):
    sep = '=' * 80
    print(f'\n{sep}\n{string}\n{sep}\n')

def printt(string):
    sep = '=' * 40
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
plt.rc('figure', titlesize=BIGGER_SIZE, figsize=(8, 4))

script_dir = Path(__file__).parent

def sfig(fig_name):
    out = script_dir / "figures_v2" / fig_name
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=dpi, bbox_inches='tight')
    print('figure name: ', fig_name)

def displayvar(name: str, var, post: Optional[str] = None, accuracy: Optional[int] = None) -> None:
    if isinstance(var, np.ndarray):
        var = sp.Matrix(var)
    if accuracy is None:
        display(Math(f"{name} = {sp.latex(var)}"))
    else:
        display(Math(f"{name} \\approx {sp.latex(sp.sympify(var).evalf(accuracy))}"))

# ------------------------------------------------------------------
# Plot helper — force vs displacement
# ------------------------------------------------------------------
def plot_fd(disp, force, title, marker='-o', label=None, xlabel='Displacement u_Gamma [mm]'):
    plt.figure()
    plt.plot(disp, force, marker, **({"label": label} if label else {}))
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('Reaction Force [N]')
    plt.grid(True)
    if label:
        plt.legend()
    sfig(title)
    plt.show()

#===================================================================================================
####################################################################################################
new_subtask('Task 1 - Force vs displacement plot ')
####################################################################################################
#===================================================================================================

df = pd.read_excel('TENSION (1).xlsx', sheet_name='Sheet1')
time_vals  = df.iloc[:, 0]
tens_vals  = df.iloc[:, 1]
print('time values:\n', time_vals)
print('force values tension:\n', tens_vals)

df = pd.read_excel('COMPRESSION.xlsx', sheet_name='Sheet1')
print("Columns found:", df.columns)
comp_vals = df.iloc[:, 1]
print('force values compression:\n', comp_vals)

tens_u_vals = [i * 20  for i in time_vals]
comp_u_vals = [i * -15 for i in time_vals]

for u_vals, f_vals, tag in [
    (tens_u_vals, tens_vals, 'tension'),
    (comp_u_vals, comp_vals, 'compression'),
]:
    title = f'Reaction force vs displacement - {tag} (Abaqus)'
    plt.figure()
    plt.plot(u_vals, f_vals)
    plt.xlabel('uΓ [mm]')
    plt.ylabel('vertical reaction force [N]')
    sfig(title)
    plt.show()

#===================================================================================================
####################################################################################################
new_subtask('Task 2 - CE1 ')
####################################################################################################
#===================================================================================================

# ------------------------------------------------------------------
# Be, Ae
# ------------------------------------------------------------------
xi1, xi2 = sp.symbols('xi1 xi2', real=True)
xi = sp.Matrix([xi1, xi2])

xe1_1, xe1_2 = sp.symbols('xe1_1 xe1_2', real=True)
xe2_1, xe2_2 = sp.symbols('xe2_1 xe2_2', real=True)
xe3_1, xe3_2 = sp.symbols('xe3_1 xe3_2', real=True)

xe1 = sp.Matrix([xe1_1, xe1_2])
xe2 = sp.Matrix([xe2_1, xe2_2])
xe3 = sp.Matrix([xe3_1, xe3_2])

N1, N2, N3 = 1 - xi1 - xi2, xi1, xi2

dN1_dxi = sp.Matrix([sp.diff(N1, xi1), sp.diff(N1, xi2)])
dN2_dxi = sp.Matrix([sp.diff(N2, xi1), sp.diff(N2, xi2)])
dN3_dxi = sp.Matrix([sp.diff(N3, xi1), sp.diff(N3, xi2)])

x = N1*xe1 + N2*xe2 + N3*xe3
J = x.jacobian(xi)
J_inv_T = J.inv().T

dN1_dx = sp.simplify(J_inv_T * dN1_dxi)
dN2_dx = sp.simplify(J_inv_T * dN2_dxi)
dN3_dx = sp.simplify(J_inv_T * dN3_dxi)

Be = sp.Matrix([
    [dN1_dx[0], 0,          dN2_dx[0], 0,          dN3_dx[0], 0         ],
    [0,         dN1_dx[1],  0,         dN2_dx[1],  0,         dN3_dx[1] ],
    [dN1_dx[1], dN1_dx[0],  dN2_dx[1], dN2_dx[0],  dN3_dx[1], dN3_dx[0] ]
])

Ae = sp.simplify(0.5 * J.det())

Be_func_cst = sp.lambdify((xe1, xe2, xe3), Be, modules="numpy")
Ae_func     = sp.lambdify((xe1, xe2, xe3), Ae, modules="numpy")

# ------------------------------------------------------------------
# Precompute sparse COO pattern
# ------------------------------------------------------------------
def precompute_pattern(Edof):
    Edof0 = Edof[:, 1:].astype(np.int64) - 1
    ndofe = Edof0.shape[1]
    ii = np.repeat(np.arange(ndofe), ndofe)
    jj = np.tile(np.arange(ndofe), ndofe)
    rows = Edof0[:, ii].ravel()
    cols = Edof0[:, jj].ravel()
    return rows, cols

# ------------------------------------------------------------------
# Precompute Be and Ae for all elements
# ------------------------------------------------------------------
def create_Be_Ae_matrix(nel, Ex, Ey):
    ngp = 1
    Be_matrix = np.zeros((nel, ngp, 3, 6))
    Ae_matrix = np.zeros((nel, ngp))
    for el in range(nel):
        x1 = np.array([Ex[el, 0], Ey[el, 0]])
        x2 = np.array([Ex[el, 1], Ey[el, 1]])
        x3 = np.array([Ex[el, 2], Ey[el, 2]])
        Be_matrix[el, 0] = Be_func_cst(x1, x2, x3)
        Ae_matrix[el, 0] = Ae_func(x1, x2, x3)
    return Be_matrix, Ae_matrix

# ------------------------------------------------------------------
# Assemble K and f_int using precomputed COO pattern
# ------------------------------------------------------------------
def assemble_K_fint_coo(a, Edof, rows, cols, ndof, nel, Ex, Ey, D, body, thickness, my_element):
    Be_matrix, Ae_matrix = create_Be_Ae_matrix(nel, Ex, Ey)
    f_ext   = np.zeros(ndof, dtype=float)
    ndofe   = Edof.shape[1] - 1
    nnz_per_el = ndofe * ndofe
    data    = np.empty(nel * nnz_per_el, dtype=float)
    p = 0

    for el in range(nel):
        edof = Edof[el, 1:].astype(np.int64) - 1
        ae   = a[edof]
        fe, Ke, *_ = my_element(ae, el, Be_matrix, Ae_matrix, D, body, thickness)
        f_ext[edof] += fe
        data[p:p + nnz_per_el] = Ke.ravel()
        p += nnz_per_el

    K = coo_matrix((data, (rows, cols)), shape=(ndof, ndof)).tocsr()
    K.sum_duplicates()
    return K, f_ext

# ------------------------------------------------------------------
# Constant strain element  (ngp=1, loop removed)
# ------------------------------------------------------------------
def cst_element(ae, el, Be_matrix, Ae_matrix, D, body, h):
    Ae = Ae_matrix[el, 0]
    Be = Be_matrix[el, 0]
    fe = np.tile([body[el], body[el]], 3) * Ae / 3
    Ke = Be.T @ D @ Be * Ae * h
    return fe, Ke

#%%
#===================================================================================================
####################################################################################################
new_subtask('Task 2 a) - Extension of CE1')
####################################################################################################
#===================================================================================================

h_val  = 100   # mm
E_val  = 20    # MPa
nu_val = 0.45

factor = E_val / ((1 + nu_val) * (1 - 2 * nu_val))
D = factor * np.array([
    [1 - nu_val,  nu_val,              0              ],
    [nu_val,      1 - nu_val,          0              ],
    [0,           0,           (1 - 2*nu_val) / 2     ]
])

filename = 'topology_coarse_3node.mat'

# ------------------------------------------------------------------
# Read .mat topology file
# ------------------------------------------------------------------
def read_toplogy_from_mat_file(filename):
    mat = sio.loadmat(filename)
    load = lambda k: mat[k].ravel().astype(int)
    Ex, Ey   = mat['Ex'], mat['Ey']
    Edof     = mat['Edof'].astype(int)
    ndofs    = mat['ndofs'].item()
    nelem    = mat['nelem'].item()
    nnodes   = mat['nnodes'].item()
    return (Ex, Ey, Edof,
            load('dof_upper'), load('dof_lower'),
            ndofs, nelem, nnodes,
            load('dof_corner'))

Ex, Ey, Edof, dof_upper, dof_lower, ndofs, nelem, nnodes, dof_corner = \
    read_toplogy_from_mat_file(filename)

dof_upper -= 1
dof_lower -= 1

# Vectorised polygon build
polygons = np.stack([Ex[:, :3], Ey[:, :3]], axis=2)

fig1, ax1 = plt.subplots()
ax1.add_collection(PolyCollection(polygons, facecolors='none', edgecolors='k'))
ax1.autoscale()

# ------------------------------------------------------------------
# Simulation parameters
# ------------------------------------------------------------------
rows, cols = precompute_pattern(Edof)
n_steps    = 10
tol        = 1e-6
max_iter   = 10
body       = np.zeros(nelem)

def solve_task_2a(target_displacement, plot_tag):
    force_history = []
    u_vals = []
    a = np.zeros(ndofs)

    for step in range(1, n_steps + 1):
        t = step / n_steps
        current_u_gamma = t * target_displacement

        dof_lower_y    = dof_lower[dof_lower % 2 == 1]
        dof_C          = np.concatenate([dof_lower_y, dof_corner])
        prescribed_dofs = dof_upper[dof_upper % 2 == 1]

        bc_dofs = np.concatenate([dof_C, prescribed_dofs])
        bc_vals = np.zeros(len(bc_dofs))
        bc_vals[len(dof_C):] = current_u_gamma

        dof_F = np.setdiff1d(np.arange(ndofs), bc_dofs)
        a[bc_dofs] = bc_vals

        for i in range(max_iter):
            K, f_ext = assemble_K_fint_coo(
                a, Edof, rows, cols, ndofs, nelem,
                Ex, Ey, D, body, h_val, my_element=cst_element)

            f_int = K @ a
            r     = f_int - f_ext
            r_F   = r[dof_F]
            res_norm = np.linalg.norm(r_F)

            if res_norm < tol:
                print(f'Converged in {i} iterations, \nresidual: {res_norm:.2e}')
                break

            a[dof_F] += spla.spsolve(K[dof_F, :][:, dof_F], -r_F)

            if i == max_iter - 1:
                print('Max iterations reached without convergence.')

        r_tot  = (K @ a) - f_ext
        Ry_sum = np.sum(r_tot[dof_upper])
        u_vals.append(current_u_gamma)
        force_history.append(Ry_sum)

    # ------------------------------------------------------------------
    # Deformed mesh
    # ------------------------------------------------------------------
    def_polygons = np.zeros((nelem, 3, 2))
    for el in range(nelem):
        edofs = Edof[el, 1:] - 1
        def_polygons[el] = [
            [Ex[el, 0] + a[edofs[0]], Ey[el, 0] + a[edofs[1]]],
            [Ex[el, 1] + a[edofs[2]], Ey[el, 1] + a[edofs[3]]],
            [Ex[el, 2] + a[edofs[4]], Ey[el, 2] + a[edofs[5]]]
        ]

    fig2, ax2 = plt.subplots()
    ax2.add_collection(PolyCollection(def_polygons, facecolors='none', edgecolors='r'))
    ax2.autoscale()
    ax2.set_title("Deformed mesh")

    # ------------------------------------------------------------------
    # Stress plot
    # ------------------------------------------------------------------
    Es = np.zeros((nelem, 3))
    for el in range(nelem):
        Be = Be_func_cst(
            [Ex[el, 0], Ey[el, 0]],
            [Ex[el, 1], Ey[el, 1]],
            [Ex[el, 2], Ey[el, 2]])
        Es[el] = D @ Be @ a[Edof[el, 1:] - 1]

    fig3, ax3 = plt.subplots()
    pc3 = PolyCollection(polygons, array=Es[:, 0], cmap='turbo', edgecolors='k')
    ax3.add_collection(pc3)
    ax3.autoscale()
    ax3.set_title("sigma xx")
    fig2.colorbar(pc3, ax=ax3)

    # ------------------------------------------------------------------
    # Force-displacement plot
    # ------------------------------------------------------------------
    title = 'vertical reaction force vs uΓ - ' + plot_tag
    plt.figure()
    plt.plot(u_vals, force_history, 'X-')
    plt.title(title)
    plt.xlabel('uΓ [m]')
    plt.ylabel('vertical reaction force [N]')
    sfig(title)
    plt.show()

    return u_vals, force_history

printt('LC1 - tension')
u_vals_tension, force_history_tension = solve_task_2a(target_displacement=20,  plot_tag='tension')
printt('LC2 - compression')
u_vals_compression, force_history_compression = solve_task_2a(target_displacement=-15, plot_tag='compression')

#%%
#===================================================================================================
####################################################################################################
new_subtask('Task 2 b) - Yeoh hyperelastic material model')
####################################################################################################
#===================================================================================================

F_11_vals = np.linspace(0.5, 1.5, 100)
G_val     = E_val / (2 * (1 + nu_val))
lam_val   = E_val * nu_val / ((1 + nu_val) * (1 - 2 * nu_val))

c10, c20, c30 = G_val / 2, -G_val / 10, G_val / 30
D1, D2, D3    = 0.02, 0.01, 0.01

def yeoh_functions():
    Fv = sp.Matrix(sp.symbols('Fv0:4', real=True))
    F  = sp.Matrix([[Fv[0], Fv[2], 0], [Fv[1], Fv[3], 0], [0, 0, 1]])
    C  = F.T * F
    J  = F.det()
    I1 = J**(-sp.Rational(2, 3)) * sp.trace(C) - 3

    U0 = (c10*I1 + c20*I1**2 + c30*I1**3
          + (1/D1)*(J-1)**2 + (1/D2)*(J-1)**4 + (1/D3)*(J-1)**6)

    Pv     = sp.diff(U0, Fv)
    dPvdFv = Pv.jacobian(Fv)
    P      = sp.Matrix([[Pv[0], Pv[2], 0], [Pv[1], Pv[3], 0], [0, 0, 0]])
    S      = F.inv() * P
    Sv_out = sp.Matrix([S[0,0], S[1,1], S[0,1], S[1,0]])

    return (sp.lambdify([Fv], Pv,     modules="numpy", cse=True),
            sp.lambdify([Fv], dPvdFv, modules="numpy", cse=True),
            sp.lambdify([Fv], Sv_out, modules="numpy", cse=True))

def neohooke_functions():
    Fv   = sp.Matrix(sp.symbols('Fv0:4', real=True))
    F    = sp.Matrix([[Fv[0], Fv[2], 0], [Fv[1], Fv[3], 0], [0, 0, 1]])
    C    = F.T * F
    J    = F.det()
    invC = sp.simplify(C.inv())
    S    = G_val * (sp.eye(3) - invC) + lam_val * sp.log(J) * invC
    P    = F * S
    Pv   = sp.Matrix([P[0,0], P[1,1], P[0,1], P[1,0]])
    Sv_out = sp.Matrix([S[0,0], S[1,1], S[0,1], S[1,0]])

    return (sp.lambdify([Fv], Pv,             modules="numpy", cse=True),
            sp.lambdify([Fv], Pv.jacobian(Fv), modules="numpy", cse=True),
            sp.lambdify([Fv], Sv_out,          modules="numpy", cse=True))

P_Yeoh_func, dPvdFv_Yeoh_func, S_Yeoh_func = yeoh_functions()
P_Neo_func,  dPvdFv_Neo_func,  S_Neo_func  = neohooke_functions()

sigma11_yeoh, sigma11_neo = [], []

for f11 in F_11_vals:
    F_vec_input = [f11, 0.0, 0.0, 1.0]
    J = f11

    s11_yeoh = S_Yeoh_func(F_vec_input)[0][0]
    sigma11_yeoh.append((1 / J) * f11 * s11_yeoh * f11)

    s11_neo = S_Neo_func(F_vec_input)[0][0]
    sigma11_neo.append((1 / J) * f11 * s11_neo * f11)

title = 'Cauchy stress component sigma11 vs F11'
plt.figure()
plt.plot(F_11_vals, sigma11_yeoh, 'o-')
plt.title(title)
plt.xlabel('F_11')
plt.ylabel('sigma11 [MPa]')
sfig(title)
plt.show()

printt('Validate results:')
print(f'Yeoh sigma11:     {sigma11_yeoh[-1]:.4e} MPa (ref: 1.2142e02)')
print(f'Neo-Hooke sigma11:{sigma11_neo[-1]:.4e} MPa (ref: 2.2525e01)')

#%%
#===================================================================================================
####################################################################################################
new_subtask('Task 2 c) - Own element function')
####################################################################################################
#===================================================================================================

def generate_fast_tri6():
    xi, eta = sp.symbols('xi eta')
    x_nodes = [sp.symbols(f'x{i}') for i in range(6)]
    y_nodes = [sp.symbols(f'y{i}') for i in range(6)]

    L  = 1.0 - xi - eta
    Ns = [L*(2*L-1), xi*(2*xi-1), eta*(2*eta-1),
          4*xi*L, 4*xi*eta, 4*eta*L]

    x_map = sum(N*x for N, x in zip(Ns, x_nodes))
    y_map = sum(N*y for N, y in zip(Ns, y_nodes))

    J     = sp.Matrix([[sp.diff(x_map, xi), sp.diff(x_map, eta)],
                       [sp.diff(y_map, xi), sp.diff(y_map, eta)]])
    detJ  = J.det()
    invJ  = J.inv()

    dN_dxi_list = [sp.Matrix([sp.diff(N, xi), sp.diff(N, eta)]) for N in Ns]
    dN_dX_list  = [invJ.T @ dN for dN in dN_dxi_list]

    B = sp.zeros(4, 12)
    for i, dN_dX in enumerate(dN_dX_list):
        B[0, 2*i]   = dN_dX[0]
        B[1, 2*i+1] = dN_dX[0]
        B[2, 2*i]   = dN_dX[1]
        B[3, 2*i+1] = dN_dX[1]

    return sp.lambdify([xi, eta] + x_nodes + y_nodes, [B, detJ], 'numpy')


def precompute_element(X_ref, fast_func, thickness=1.0):
    gps    = [(1./6, 1./6, 1./6), (2./3, 1./6, 1./6), (1./6, 2./3, 1./6)]
    coords = list(X_ref[:, 0]) + list(X_ref[:, 1])
    return [(B, detJ * w * thickness)
            for xi, eta, w in gps
            for B, detJ in [fast_func(xi, eta, *coords)]]


def el6_yeoh(u, element_data):
    Ke, fe = np.zeros((12, 12)), np.zeros(12)
    F_all, P_all = [], []

    for B, dv in element_data:
        F_vec = B @ u
        F_vec[0] += 1.0
        F_vec[3] += 1.0
        P_vec  = P_Yeoh_func(F_vec).flatten()
        dPvdFv = dPvdFv_Yeoh_func(F_vec)
        fe += (B.T @ P_vec) * dv
        Ke += (B.T @ dPvdFv @ B) * dv
        F_all.append(F_vec)
        P_all.append(P_vec)

    return Ke, fe, np.concatenate(F_all), np.concatenate(P_all)

# Validation data
X_ref = np.array([[0.0,0.0],[3.0,0.0],[0.0,2.0],[1.5,0.0],[1.5,1.0],[0.0,1.0]])
x_curr = np.array([[6.0,0.7],[7.0,2.3],[4.5,1.8],[6.4,1.2],[5.6,2.0],[5.2,1.1]])
u = (x_curr - X_ref).flatten()

tri6_func    = generate_fast_tri6()
element_data = precompute_element(X_ref, tri6_func, thickness=0.001)
Ke_val, fe_val, F_res, P_res = el6_yeoh(u, element_data)

printt('Validate functions')
print('deformation_gradient_2d:');   displayvar('F', F_res.T, accuracy=3)
print('\nPiola_Kirchoff_2d:');        displayvar('P', P_res.T, accuracy=3)
print('\nfe_int:');                   displayvar('f', fe_val,  accuracy=3)
print('\nKe_int top (8x8):');         displayvar('K', Ke_val[0:8, 0:8], accuracy=3)

#%%
#===================================================================================================
####################################################################################################
new_subtask('Task 2 d) - Combine routines')
####################################################################################################
#===================================================================================================

def solve_task_2d(filename, n_steps=50, tol=1e-6, u_final=20, h=100, max_iter=25):
    Ex, Ey, Edof, dof_upper, dof_lower, ndof, nel, nnodes, dof_corner = \
        read_toplogy_from_mat_file(filename)

    Edof_indices             = Edof[:, 1:].astype(np.int64) - 1
    rows_pattern, cols_pattern = precompute_pattern(Edof)
    nnz_per_el               = 12 * 12
    data                     = np.empty(nel * nnz_per_el, dtype=float)

    tri6_func = generate_fast_tri6()
    print('Pre-computing element matrices')

    xi_c, eta_c = 1.0/3.0, 1.0/3.0
    element_store, centroid_store = [], []
    for el in range(nel):
        X_ref  = np.vstack([Ex[el, :], Ey[el, :]]).T
        coords = list(X_ref[:, 0]) + list(X_ref[:, 1])
        element_store.append(precompute_element(X_ref, tri6_func, h))
        B_cent, _ = tri6_func(xi_c, eta_c, *coords)
        centroid_store.append(B_cent)

    a             = np.zeros(ndof)
    u_steps       = np.linspace(0, u_final, n_steps + 1)
    force_history = [0.0]
    disp_history  = [0.0]

    printt('Starting solver')
    for step, u_app in enumerate(u_steps):
        if step == 0:
            continue

        top_y_dofs = [d - 1 for d in dof_upper if d % 2 == 0]
        bot_y_dofs = [d - 1 for d in dof_lower if d % 2 == 0]
        dof_corner_0 = dof_corner  # already 1-indexed from .mat

        bc_dofs = np.array(bot_y_dofs + list(dof_corner_0) + top_y_dofs, dtype=int)
        bc_vals = np.array([0.0] * (len(bot_y_dofs) + len(dof_corner_0))
                           + [u_app] * len(top_y_dofs))

        a[bc_dofs] = bc_vals

        free_dofs = np.where(~np.isin(np.arange(ndof), bc_dofs))[0]

        converged = False
        for it in range(max_iter):
            f_int = np.zeros(ndof)

            for el in range(nel):
                edof_idx = Edof_indices[el]
                Ke, fe, _, _ = el6_yeoh(a[edof_idx], element_store[el])
                f_int[edof_idx] += fe
                idx = el * nnz_per_el
                data[idx:idx + nnz_per_el] = Ke.ravel()

            K_global = sps.coo_matrix(
                (data, (rows_pattern, cols_pattern)), shape=(ndof, ndof)).tocsr()

            r        = -f_int[free_dofs]
            res_norm = np.linalg.norm(r)

            if res_norm < tol:
                converged = True
                print(f"Step {step}: Converged (Iter {it}, Res {res_norm:.2e})")
                break

            K_free = K_global[free_dofs, :][:, free_dofs]
            try:
                a[free_dofs] += spla.spsolve(K_free, r)
            except Exception:
                print(f"\nError: solver failed")
                return a, disp_history, force_history

        if not converged:
            print(f'\nWarning: step {step} did not converge.')
            break

        force_history.append(np.sum(f_int[top_y_dofs]))
        disp_history.append(u_app)

        # Stress plot at final step
        if step == n_steps:
            polygons_def, stress_vals = [], []
            for el in range(nel):
                edof_idx = Edof_indices[el]
                u_el     = a[edof_idx]
                ex, ey   = Ex[el, :], Ey[el, :]
                x_nodes  = ex + u_el[0::2]
                y_nodes  = ey + u_el[1::2]
                polygons_def.append(np.column_stack([x_nodes[[0,1,2]], y_nodes[[0,1,2]]]))

                B_cent = centroid_store[el]
                F_vec  = B_cent @ u_el
                F_vec[0] += 1.0
                F_vec[3] += 1.0

                F_mat  = np.array([[F_vec[0], F_vec[2]], [F_vec[1], F_vec[3]]])
                J      = F_mat[0,0]*F_mat[1,1] - F_mat[0,1]*F_mat[1,0]
                P_vals = P_Yeoh_func(F_vec).flatten()
                P_ten  = np.array([[P_vals[0], P_vals[2]], [P_vals[1], P_vals[3]]])
                Sigma  = (1.0/J) * P_ten @ F_mat.T
                s11, s22, s12 = Sigma[0,0], Sigma[1,1], Sigma[0,1]
                stress_vals.append(np.sqrt(s11**2 + s22**2 - s11*s22 + 3*s12**2))

            title = f'Von Mises Stress\nDisplacement: {u_final} mm'
            fig, ax = plt.subplots(figsize=(10, 8))
            pc = PolyCollection(polygons_def, array=np.array(stress_vals),
                                cmap='turbo', edgecolors='black', linewidths=0.2)
            ax.add_collection(pc)
            ax.autoscale()
            ax.set_aspect('equal')
            ax.set_title(title)
            fig.colorbar(pc, ax=ax, label='Von Mises Stress [MPa]')
            plt.show()

    return a, disp_history, force_history

#%%
printt('Run solver - coarse mesh - tension')
a_coarse_tension, disp_history_coarse_tension, force_history_coarse_tension = \
    solve_task_2d('topology_coarse_6node.mat', n_steps=50,  tol=1e-5, u_final=20,  h=100)

plot_fd(disp_history_coarse_tension, force_history_coarse_tension,
        'Coarse mesh - Total vertical reaction force vs displacement - tension')

#%%
printt('Run solver - coarse mesh - compression')
a_coarse_compression, disp_history_coarse_compression, force_history_coarse_compression = \
    solve_task_2d('topology_coarse_6node.mat', n_steps=16,  tol=1e-5, u_final=-13, h=100)

plot_fd(disp_history_coarse_compression, force_history_coarse_compression,
        'Coarse mesh - Total vertical reaction force vs displacement - compression')

#%%
printt('Run solver - medium mesh - tension')
a_medium, disp_history_medium, force_history_medium = \
    solve_task_2d('topology_medium_6node.mat', n_steps=150, tol=1e-5, u_final=20,  h=100)

plot_fd(disp_history_medium, force_history_medium,
        'Medium mesh - Total vertical reaction force vs displacement - tension')

#%%
printt('Comparison with task 2a)')

for title, d1, f1, l1, d2, f2, l2 in [
    ('Comparison between 2a) and 2d) - tension',
     u_vals_tension,              force_history_tension,              '2a)',
     disp_history_coarse_tension, force_history_coarse_tension,       '2d)'),
    ('Comparison between 2a) and 2d) - compression',
     u_vals_compression,              force_history_compression,          '2a)',
     disp_history_coarse_compression, force_history_coarse_compression,   '2d)'),
    ('Comparison between coarse and medium mesh - tension',
     disp_history_coarse_compression, force_history_coarse_compression, 'coarse',
     disp_history_medium,             force_history_medium,             'medium'),
]:
    plt.figure()
    plt.plot(d1, f1, 'o-', label=l1)
    plt.plot(d2, f2, '-o', alpha=0.7, label=l2)
    plt.title(title)
    plt.xlabel('Displacement u_Gamma [mm]')
    plt.ylabel('Reaction Force [N]')
    plt.grid(True)
    plt.legend()
    sfig(title)
    plt.show()
#%%