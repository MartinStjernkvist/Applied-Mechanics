import numpy as np
import scipy.sparse as scip
import scipy.sparse.linalg as spla
from scipy.sparse import coo_matrix
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import matplotlib.tri as tri
import sympy as sp

#%%
# -------------------------------------------------
# Isoparametric coordinates
# -------------------------------------------------
xi1, xi2 = sp.symbols('xi1 xi2', real=True)
xi = sp.Matrix([xi1, xi2])

# Shape functions
N1 = 1 - xi1 - xi2
N2 = xi1
N3 = xi2

# Gradients w.r.t isoparametric coordinates
dN1_dxi = sp.Matrix([sp.diff(N1, xi1), sp.diff(N1, xi2)])
dN2_dxi = sp.Matrix([sp.diff(N2, xi1), sp.diff(N2, xi2)])
dN3_dxi = sp.Matrix([sp.diff(N3, xi1), sp.diff(N3, xi2)])

# -------------------------------------------------
# Node positions (vectors)
# -------------------------------------------------
xe1_1, xe1_2 = sp.symbols('xe1_1 xe1_2', real=True)
xe2_1, xe2_2 = sp.symbols('xe2_1 xe2_2', real=True)
xe3_1, xe3_2 = sp.symbols('xe3_1 xe3_2', real=True)

xe1 = sp.Matrix([xe1_1, xe1_2])
xe2 = sp.Matrix([xe2_1, xe2_2])
xe3 = sp.Matrix([xe3_1, xe3_2])

# -------------------------------------------------
# Mapping and Jacobian
# -------------------------------------------------
x = N1*xe1 + N2*xe2 + N3*xe3
Fisop = x.jacobian(xi)

Fisop_inv_T = Fisop.inv().T

# Spatial derivatives
dN1_dx = sp.simplify(Fisop_inv_T * dN1_dxi)
dN2_dx = sp.simplify(Fisop_inv_T * dN2_dxi)
dN3_dx = sp.simplify(Fisop_inv_T * dN3_dxi)

# -------------------------------------------------
# B-matrix
# -------------------------------------------------
Be = sp.Matrix([
    [dN1_dx[0], 0,           dN2_dx[0], 0,           dN3_dx[0], 0],
    [0,           dN1_dx[1], 0,           dN2_dx[1], 0,           dN3_dx[1]],
    [dN1_dx[1], dN1_dx[0], dN2_dx[1], dN2_dx[0], dN3_dx[1], dN3_dx[0]]
])

# -------------------------------------------------
# Callable function: Be_func(xe1, xe2, xe3)
# -------------------------------------------------
Be_func_cst = sp.lambdify(
    (xe1, xe2, xe3),
    Be,
    modules="numpy"
)

# Element area expression
Ae = sp.simplify(0.5 * Fisop.det())

# Lambdified function: Ae_func(xe1, xe2, xe3)
Ae_func = sp.lambdify(
    (xe1, xe2, xe3),
    Ae,
    modules="numpy"
)
#%%
#%%
# -------------------------------------------------
# Geometry and mesh
# -------------------------------------------------
L = 200.0     # mm
H = 20.0      # mm
h = 10.0      # thickness [mm]

nel = 6
nnodes = 8
nen = 3

# -------------------------------------------------
# Edof, coordinates and dofs
# -------------------------------------------------
Edof = np.array([
    [1,  1,  2,  3,  4,  7,  8],
    [2,  3,  4,  5,  6,  7,  8],
    [3,  5,  6, 11, 12,  7,  8],
    [4,  5,  6,  9, 10, 11, 12],
    [5,  9, 10, 15, 16, 11, 12],
    [6,  9, 10, 13, 14, 15, 16]
], dtype=int)

Coord = np.array([
    [0,     H],
    [0,     0],
    [L/3,   0],
    [L/3,   H],
    [2*L/3, 0],
    [2*L/3, H],
    [L,     0],
    [L,     H]
])

Dof = np.array([
    [1,  2],
    [3,  4],
    [5,  6],
    [7,  8],
    [9, 10],
    [11,12],
    [13,14],
    [15,16]
])

nel = Edof.shape[0]
Ex = np.zeros((nel, 3)) # 3 nodes per element
Ey = np.zeros((nel, 3))
polygons = np.zeros((nel, 3, 2))

# the code below assumes Edof has elemets sortered

for el in range(nel):
    node_ids = (Edof[el, [1, 3, 5]] - 1) // 2
    Ex[el,:] = Coord[node_ids,0]
    Ey[el,:] = Coord[node_ids,1]
    polygons[el,:,:] = [[Ex[el,0],Ey[el,0]], [Ex[el,1],Ey[el,1]], [Ex[el,2],Ey[el,2]]]
    

# Example. Plot polygons without colouring (e.g. undeformed and deformed mesh)
fig1, ax1 = plt.subplots()

pc1 = PolyCollection(
    polygons,
    facecolors='none',
    edgecolors='k'
)

ax1.add_collection(pc1)
ax1.autoscale()
ax1.set_title("Undeformed mesh")


#%%
#precompute sparse pattern once to save computation time during assembly
def precompute_pattern(Edof):
    Edof0 = Edof[:, 1:].astype(np.int64) - 1
    nel, ndofe = Edof0.shape
    nnz_per_el = ndofe * ndofe
    nnz_total  = nel * nnz_per_el

    ii = np.repeat(np.arange(ndofe), ndofe)
    jj = np.tile(np.arange(ndofe), ndofe)

    rows = np.empty(nnz_total, dtype=np.int64)
    cols = np.empty(nnz_total, dtype=np.int64)

    p = 0
    for el in range(nel):
        edof = Edof0[el]
        rows[p:p+nnz_per_el] = edof[ii]
        cols[p:p+nnz_per_el] = edof[jj]
        p += nnz_per_el

    return rows, cols
rows, cols= precompute_pattern(Edof)
#%% To save computation time, we precompute Be and Ae matrices for all elements and gauss points
def create_Be_Ae_matrix(nel, Ex, Ey):
    
    ngp=1 #number of Gauss points
    Be_matrix = np.zeros((nel,ngp, 3, 6))
    Ae_matrix = np.zeros((nel, ngp))

    for el in range(nel):
        x1 = np.array([Ex[el,0], Ey[el,0]])
        x2 = np.array([Ex[el,1], Ey[el,1]])
        x3 = np.array([Ex[el,2], Ey[el,2]])
        

        for gp in range(ngp):
            Be = Be_func_cst(x1, x2, x3)
            Be_matrix[el, gp, :, :] = Be
            Ae = Ae_func(x1, x2, x3)
            Ae_matrix[el, gp] = Ae

    return Be_matrix, Ae_matrix

Be_matrix, Ae_matrix = create_Be_Ae_matrix(nel, Ex, Ey)
#%% This assembling routine uses precomputed COO pattern and precomputed Be and Ae matrices
def assemble_K_fint_coo(a, Edof, rows, cols, ndof, nel, Ex, Ey, D, body, thickness,my_element):
    # ------------------------------------------------------------------
    # Initialize global internal force vector
    # ------------------------------------------------------------------
    f_ext = np.zeros(ndof, dtype=float)

    # Number of DOFs per element (e.g. 12 for tri6 with 2 DOF/node)
    ndofe = Edof.shape[1]-1

    # Each element contributes a dense (ndofe x ndofe) stiffness block
    nnz_per_el = ndofe * ndofe
    nnz_total = nel * nnz_per_el
    # ------------------------------------------------------------------
    # Preallocate COO triplet arrays (row, col, value)
    # ------------------------------------------------------------------
    data = np.empty(nnz_total, dtype=float)
    # Pointer into the preallocated triplet arrays
    p = 0

    # ------------------------------------------------------------------
    # Element loop
    # ------------------------------------------------------------------
    for el in range(nel):
        # Element DOF indices
        edof = Edof[el, 1:].astype(np.int64) -1
        # Compute element internal force and stiffness matrix
        ae = a[edof]
        fe, Ke, *_ = my_element(
            ae, el, Be_matrix, Ae_matrix, D, body, thickness) #Ex, Ey, D, body,  thickness
        
        # Assemble internal force contributions
        f_ext[edof] += fe

        # ------------------------------------------------------------------
        # Assemble stiffness using COO triplets
        data[p:p + nnz_per_el] = Ke.ravel()
        p += nnz_per_el

    # ------------------------------------------------------------------
    # Build global stiffness matrix and convert to CSR for efficient slicing/solves
    # ------------------------------------------------------------------
    K = coo_matrix((data, (rows, cols)), shape=(ndof, ndof)).tocsr()

    # Sum duplicate entries (multiple elements contribute to same (i,j) location)
    K.sum_duplicates()

    return K, f_ext


#%%

def cst_element(ae, el, Be_matrix, Ae_matrix, D, body, h): # Ex, Ey, D, body, h):
    ngp=1; fe=np.zeros(6); Ke=np.zeros((6,6))
    #x1 = np.array([Ex[el,0], Ey[el,0]])
    #x2 = np.array([Ex[el,1], Ey[el,1]])
    #x3 = np.array([Ex[el,2], Ey[el,2]])

    for gp in range(ngp):
        Ae = Ae_matrix[el, gp]
        Be = Be_matrix[el, gp, :, :]  # Be_func_cst(x1, x2, x3)
        fe = np.tile(body[el], 3) * Ae / 3
        Ke = Be.T @ D @ Be * Ae * h       

    return fe, Ke

#%%

# -------------------------------------------------
# Global matrices
# -------------------------------------------------
ndofs = 2 * nnodes

f_ext = np.zeros(ndofs)
a = np.zeros(ndofs)

# -------------------------------------------------
# Material data
# -------------------------------------------------
Emod = 200e3     # MPa
nu = 0.3

# plane stress

D = (Emod / (1 - nu**2)) * np.array([
            [1,   nu,       0],
            [nu,   1,       0],
            [0,   0,  (1-nu)/2]
        ])

# -------------------------------------------------
# Boundary conditions
# -------------------------------------------------
dof_C = np.array([1, 2, 3, 4, 14, 16]) - 1
dof_F = np.setdiff1d(np.arange(ndofs), dof_C)

a_C = np.array([0, 0, 0, 0, H/10, H/10])

# -------------------------------------------------
# Body forces
# -------------------------------------------------
body = np.zeros((nel, 2))

#%%
# -------------------------------------------------
# Assembly 
# -------------------------------------------------
K, f_ext =assemble_K_fint_coo(a, Edof, rows, cols, ndofs, nel, Ex, Ey, D, body, h, my_element=cst_element)

#%%
# -------------------------------------------------
# Solve system
# -------------------------------------------------
a_F = spla.spsolve(
    K[np.ix_(dof_F, dof_F)],
    f_ext[dof_F] - K[np.ix_(dof_F, dof_C)] @ a_C
)

f_extC = (
    K[np.ix_(dof_C, dof_F)] @ a_F +
    K[np.ix_(dof_C, dof_C)] @ a_C -
    f_ext[dof_C]
)

a[dof_F] = a_F
a[dof_C] = a_C


# -------------------------------------------------
# Plot deformed mesh
# -------------------------------------------------
def_polygons = np.zeros((nel, 3, 2))

for el in range(nel):
    edofs = Edof[el,1:] - 1
    def_polygons[el,:,:] = [[Ex[el,0]+a[edofs[0]],Ey[el,0]+a[edofs[1]]],\
                            [Ex[el,1]+a[edofs[2]],Ey[el,1]+a[edofs[3]]],\
                            [Ex[el,2]+a[edofs[4]],Ey[el,2]+a[edofs[5]]]\
                           ]

# Example. Plot polygons without colouring (e.g. undeformed and deformed mesh)
fig2, ax2 = plt.subplots()

pc2 = PolyCollection(
    def_polygons,
    facecolors='none',
    edgecolors='r'
)

ax2.add_collection(pc2)
ax2.autoscale()
ax2.set_title("Deformed mesh")


# -------------------------------------------------
# Stress computation
# -------------------------------------------------
Es = np.zeros((nel, 3))

for el in range(nel):
    x1 = np.array([Ex[el,0], Ey[el,0]])
    x2 = np.array([Ex[el,1], Ey[el,1]])
    x3 = np.array([Ex[el,2], Ey[el,2]])

    Be = Be_func_cst(x1, x2, x3)
    edofs = Edof[el,1:] - 1

    Es[el,:] = D @ Be @ a[edofs]

# -------------------------------------------------
# Stress plot (sigma_xx)
# -------------------------------------------------

fig3, ax3 = plt.subplots()

pc3 = PolyCollection(
    polygons,
    array=Es[:,0], # values used for coloring 
    cmap='turbo', 
    edgecolors='k')

ax3.add_collection(pc3)
ax3.autoscale()
ax3.set_title("sigma xx")
fig2.colorbar(pc3, ax=ax3)
#%% For help how to read data from mat files
import scipy.io as sio
#mat_file=sio.loadmat('Matlab_solution/topology_coarse_3node.mat')
mat_file=sio.loadmat('Matlab_solution/topology_fine_3node.mat')
Edof=mat_file['Edof']; Edof = Edof.astype(int)
nel=Edof.shape[0]
Dof=mat_file['Dof']; Dof = Dof.astype(int)  
Coord=mat_file['Coord']
nnodes=Coord.shape[0]
ndofs=nnodes*2
Ex=mat_file['Ex']
Ey=mat_file['Ey']
dof_lower=mat_file['dof_lower'].ravel(); dof_lower=dof_lower.astype(int)
dof_upper=mat_file['dof_upper'].ravel(); dof_upper=dof_upper.astype(int) 
dof_right=mat_file['dof_right'].ravel();  dof_right=dof_right.astype(int)
dof_left=mat_file['dof_left'].ravel(); dof_left=dof_left.astype(int)
dof_corner=mat_file['dof_corner'].ravel(); dof_corner=dof_corner.astype(int)

#%%