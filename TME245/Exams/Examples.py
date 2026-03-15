#%%
import numpy as np
import sympy as sp
#%%
#===================================================================================================
# 1
# Write a small code to solve two coupled nonlinear equations using the Newton 
# algorithm (see P1 in the file Exam_problems_with_solutions.pdf) 
#===================================================================================================

xv = np.array([1, 1])
tol = 1e-6
q = 1

while q >= tol:
    x = xv[0]
    y = xv[1]
    
    f1 = 2*x**2 + 7*x*y + 2*y**2-3
    f2 = 5*x**2 + 2*x*y + 2
    
    g = np.array([f1, f2])
    
    df1dx = 4*x + 7*y
    df1dy = 7*x + 4*y
    df2dx = 10*x + 2*y
    df2dy = 2*x
    
    J = np.array([[df1dx, df1dy],
                  [df2dx, df2dy]])

    xv = xv - np.linalg.inv(J) @ g
    
    q = np.linalg.norm(g)

print(xv)

#%%
#===================================================================================================
# 2
# Implement a function that is computing the element shape functions and shape 
# function (first order) derivatives for a 3-noded, 4-noded and 6-noded isoparametric 
# element
#===================================================================================================

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

Be_func_cst = sp.lambdify((xe1, xe2, xe3), Be, modules="numpy")

#%%
#===================================================================================================
# 7
# Given a global plate bending stiffness matrix and a plate geometric stiffness, 
# compute the critical load factor 𝜆𝜆 that yields plate buckling. Also compute and 
# plot the associated buckling mode.
#===================================================================================================

from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigs

eigenvalues, eigenvectors = eigs(K_sparse, M = G_sparse, k=3, which='SM')

positive_idx = np.where(eigenvalues > 0)[0]
idx_min = positive_idx[np.argmin(eigenvalues[positive_idx])]
lambda_cr = eigenvalues[idx_min]
phi_cr = eigenvectors[:, idx_min]

#%%
# ================================================================
# WHAT YOU NEED TO KNOW / REMEMBER
# ================================================================

GP  = 1/np.sqrt(3)
xi_v = np.array([[-GP, -GP, GP,  GP],   # xi1 at each of the 4 GPs
                 [-GP,  GP, -GP, GP]])   # xi2 at each of the 4 GPs
H_v = np.ones(4)                         # weights (all 1 for 2x2)

# ================================================================
# TASK 3 — Stiffness matrix K_ww and load vector f_w
# ================================================================

def kirchoff_plate_element(ex, ey, q, Dbar):

    K_ww = np.zeros((12, 12))
    f_w  = np.zeros((12, 1))
    
    n1,n2,n3,n4 = [np.array([[ex[i]],[ey[i]]]) for i in range(4)]

    for gp in range(4):
        xin        = xi_v[:, gp].reshape(2,1)
        
        detJ       = detFisop_4node_func(xin, n1,n2,n3,n4)   # GIVEN
        _, Bast, _ = bast_kirchoff_func(xin, n1,n2,n3,n4)    # GIVEN → Bast (3x12)
        N_w        = N_kirchoff_func(xin, n1,n2,n3,n4)       # GIVEN → N_w (1x12)
        
        w = H_v[gp] * detJ

        K_ww += Bast.T @ Dbar @ Bast * w   # ← THE KEY LINE
        f_w  += N_w.T * q * w              # ← THE KEY LINE

    return K_ww, f_w.ravel()

# ================================================================
# TASK 4 — Cauchy stress at element midpoint (or any GP)
# ================================================================

def cauchy_stress_midpoint(ex, ey, a_u_el, a_w_el, z, D):

    xin         = np.array([[0],[0]])               # midpoint
    
    xe_nodes    = np.column_stack((ex, ey))
    
    n1,n2,n3,n4 = [np.array([[ex[i]],[ey[i]]]) for i in range(4)]

    B_u, _     = Bu_and_detJ(xin, xe_nodes)            # GIVEN
    _, Bast, _ = bast_kirchoff_func(xin, n1,n2,n3,n4)  # GIVEN

    eps   = B_u @ a_u_el - z * (Bast @ a_w_el)         # ← THE KEY LINE
    sigma = D @ eps                                    # ← THE KEY LINE

# ================================================================
# TASK 6 — Section forces and moments
# ================================================================

def section_forces(xin, ex, ey, a_w_el, given_dN_func, Dbar):

    n1,n2,n3,n4 = [np.array([[ex[i]],[ey[i]]]) for i in range(4)]
    
    _, Bast, _ = given_dN_func(xin, n1,n2,n3,n4)   # GIVEN

    kappa = Bast @ a_w_el   # curvatures [κ_xx, κ_yy, 2κ_xy]
    M     = Dbar @ kappa    # moments    [Mx,   My,   Mxy]    (Nm/m)
    
#%%