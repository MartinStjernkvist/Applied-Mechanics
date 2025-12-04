#%%
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

# ==============================================================================
# PART 1: MISSING HELPER FUNCTIONS (CST & MESH)
# The provided mha021.py did not contain these, so they are implemented here 
# to solve Task 1.
# ==============================================================================

def get_D_plane_stress(E, nu):
    """Constitutive matrix for Plane Stress."""
    factor = E / (1 - nu**2)
    return factor * np.array([
        [1,  nu, 0],
        [nu, 1,  0],
        [0,  0, (1 - nu) / 2]
    ])

def cst_element(nodes, D, t, body_load):
    """
    Constant Strain Triangle (CST) element routine.
    nodes: (3,2) array
    """
    x = nodes[:, 0]
    y = nodes[:, 1]
    
    # Area calculation
    A = 0.5 * ((x[1]*y[2] - x[2]*y[1]) + (x[2]*y[0] - x[0]*y[2]) + (x[0]*y[1] - x[1]*y[0]))
    
    if A <= 0:
        raise ValueError("Element area is zero or negative. Check node ordering.")

    # B Matrix
    beta = np.array([y[1]-y[2], y[2]-y[0], y[0]-y[1]])
    gamma = np.array([x[2]-x[1], x[0]-x[2], x[1]-x[0]])
    
    B = np.zeros((3, 6))
    for i in range(3):
        B[0, 2*i]   = beta[i]
        B[1, 2*i+1] = gamma[i]
        B[2, 2*i]   = gamma[i]
        B[2, 2*i+1] = beta[i]
    
    B /= (2 * A)
    
    # Stiffness Matrix
    Ke = (B.T @ D @ B) * A * t
    
    # Force Vector (Lumped body load)
    bx, by = body_load
    fe = np.array([bx, by, bx, by, bx, by]) * (A * t / 3.0)
    
    return Ke, fe

def cst_element_stress_strain(nodes, D, ae):
    """Computes stress/strain for CST element."""
    x = nodes[:, 0]
    y = nodes[:, 1]
    A = 0.5 * ((x[1]*y[2] - x[2]*y[1]) + (x[2]*y[0] - x[0]*y[2]) + (x[0]*y[1] - x[1]*y[0]))
    
    beta = np.array([y[1]-y[2], y[2]-y[0], y[0]-y[1]])
    gamma = np.array([x[2]-x[1], x[0]-x[2], x[1]-x[0]])
    
    B = np.zeros((3, 6))
    for i in range(3):
        B[0, 2*i]   = beta[i]
        B[1, 2*i+1] = gamma[i]
        B[2, 2*i]   = gamma[i]
        B[2, 2*i+1] = beta[i]
    B /= (2 * A)
    
    epsilon = B @ ae
    sigma = D @ epsilon
    return sigma, epsilon

class SimpleMeshGenerator:
    """Generates basic structured meshes for the assignment."""
    
    @staticmethod
    def structured_rectangle_cst(L, H, nelx, nely):
        """Generates a CST mesh (2 triangles per quad)."""
        nnx = nelx + 1
        nny = nely + 1
        x = np.linspace(0, L, nnx)
        y = np.linspace(-H/2, H/2, nny)
        X, Y = np.meshgrid(x, y)
        coords = np.column_stack([X.ravel(), Y.ravel()])
        
        edofs = []
        # Standard numbering
        node_map = np.arange(nnx * nny).reshape(nny, nnx)
        
        for j in range(nely):
            for i in range(nelx):
                n1 = node_map[j, i]
                n2 = node_map[j, i+1]
                n3 = node_map[j+1, i+1]
                n4 = node_map[j+1, i]
                
                # DOF indices (1-based)
                d1 = [2*n1+1, 2*n1+2]
                d2 = [2*n2+1, 2*n2+2]
                d3 = [2*n3+1, 2*n3+2]
                d4 = [2*n4+1, 2*n4+2]
                
                # Split quad into 2 triangles
                edofs.append(d1 + d2 + d3) # Tri 1
                edofs.append(d1 + d3 + d4) # Tri 2
                
        return coords, np.array(edofs)

    @staticmethod
    def structured_rectangle_quad(L, H, nelx, nely):
        """Generates a Quad mesh."""
        nnx = nelx + 1
        nny = nely + 1
        x = np.linspace(0, L, nnx)
        y = np.linspace(-H/2, H/2, nny)
        X, Y = np.meshgrid(x, y)
        coords = np.column_stack([X.ravel(), Y.ravel()])
        
        edofs = []
        node_map = np.arange(nnx * nny).reshape(nny, nnx)
        
        for j in range(nely):
            for i in range(nelx):
                n1 = node_map[j, i]
                n2 = node_map[j, i+1]
                n3 = node_map[j+1, i+1]
                n4 = node_map[j+1, i]
                
                # DOF indices (1-based) for Quad
                # Counter-Clockwise: BottomLeft, BottomRight, TopRight, TopLeft
                dofs = [2*n1+1, 2*n1+2, 
                        2*n2+1, 2*n2+2, 
                        2*n3+1, 2*n3+2, 
                        2*n4+1, 2*n4+2]
                edofs.append(dofs)
                
        return coords, np.array(edofs)

# ==============================================================================
# PART 2: TASK 2 IMPLEMENTATION (BILINEAR QUAD)
# ==============================================================================

def compute_Ne_Be_detJ(nodes, ξ, η):
    """
    Compute the shape functions, B-matrix and determinant of Jacobian
    for a bilinear plane stress element.
    """
    # 1. Shape functions (1x4)
    # Order: (-1,-1), (1,-1), (1,1), (-1,1)
    # N1 = 0.25*(1-ξ)*(1-η)
    # N2 = 0.25*(1+ξ)*(1-η)
    # N3 = 0.25*(1+ξ)*(1+η)
    # N4 = 0.25*(1-ξ)*(1+η)
    
    Ne = 0.25 * np.array([
        (1 - ξ) * (1 - η),
        (1 + ξ) * (1 - η),
        (1 + ξ) * (1 + η),
        (1 - ξ) * (1 + η)
    ])
    
    # Shape function matrix N (2x8)
    N = np.zeros((2, 8))
    N[0, 0::2] = Ne
    N[1, 1::2] = Ne

    # 2. Derivatives of shape functions w.r.t local coords (2x4)
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
    
    dNe = np.vstack((dN_dxi, dN_deta))

    # 3. Jacobian matrix J = dNe * nodes (2x2)
    # nodes is (4,2)
    J = dNe @ nodes

    # 4. Determinant
    detJ = np.linalg.det(J)
    
    minDetJ = 1e-16
    if detJ < minDetJ:
        raise ValueError(f"Bad element geometry: detJ = {detJ}") 

    # 5. Derivatives w.r.t global coords (dN/dx, dN/dy)
    # invJ * dNe_local
    Jinv = np.linalg.inv(J)
    dNedxy = Jinv @ dNe  # (2x4)
    
    # 6. B-matrix (3x8)
    # [ dN1/dx   0      dN2/dx   0    ... ]
    # [   0    dN1/dy     0    dN2/dy ... ]
    # [ dN1/dy dN1/dx   dN2/dy dN2/dx ... ]
    
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
    Compute stiffness matrix and force vector for bilinear quad.
    """
    b = np.asarray(body_load, dtype=float).reshape(2)
    
    Ke = np.zeros((8, 8))
    fe = np.zeros(8)

    # Define Gauss points and weights
    if ngp == 1:
        points = [0.0]
        weights = [2.0] # Weight 2 per dim -> total area 4
    elif ngp == 4: # 2x2
        val = 1.0 / np.sqrt(3)
        points = [-val, val]
        weights = [1.0, 1.0]
    elif ngp == 9: # 3x3
        val = np.sqrt(0.6)
        points = [-val, 0.0, val]
        weights = [5.0/9.0, 8.0/9.0, 5.0/9.0]
    else:
        # Default fallback to 2x2 if integer passed isn't mapped
        val = 1.0 / np.sqrt(3)
        points = [-val, val]
        weights = [1.0, 1.0]

    for i, w_xi in enumerate(weights):
        for j, w_eta in enumerate(weights):
            ξ = points[i]
            η = points[j]
            w = w_xi * w_eta
            
            N, Be, detJ = compute_Ne_Be_detJ(nodes, ξ, η)

            # Stiffness matrix
            # B^T * D * B * detJ * thick * weight
            Ke += Be.T @ D @ Be * detJ * t * w
            
            # Force vector
            # N^T * b * detJ * thick * weight
            # N is (2,8), b is (2,) -> N.T @ b is (8,)
            fe += (N.T @ b) * detJ * t * w

    return Ke, fe

def bilinear_element_stress_strain(nodes, D, ae):
    """
    Compute stress and strain at the center of the element (ξ=0, η=0).
    """
    # Evaluate at center
    _, Be, _ = compute_Ne_Be_detJ(nodes, 0.0, 0.0)
    
    epsilon = Be @ ae
    sigma = D @ epsilon
    
    return sigma, epsilon


# ==============================================================================
# MAIN EXECUTION SCRIPT
# ==============================================================================

def solve_problem(mesh_type='cst'):
    print(f"\n--- Solving for Mesh Type: {mesh_type.upper()} ---")
    
    # 1. Physical Parameters
    # ----------------------
    W_total = 5.0
    L_half = W_total / 2.0  # Modeling left half (2.5m)
    H = 0.4
    thickness = 0.01
    
    E = 210e9     # Pa
    nu = 0.3
    rho = 7800    # kg/m^3
    g = 9.81
    
    # Loads
    bx = 0.0
    by = -rho * g
    body_load = [bx, by]
    
    # Constitutive Matrix (Plane Stress)
    D = get_D_plane_stress(E, nu)

    # 2. Mesh Generation
    # ------------------
    # Using a reasonably fine mesh for convergence
    nelx = 50
    nely = 8
    
    if mesh_type == 'cst':
        coords, edofs = SimpleMeshGenerator.structured_rectangle_cst(L_half, H, nelx, nely)
        n_dofs_per_node = 2
        dofs_per_elem = 6
    else: # quad
        coords, edofs = SimpleMeshGenerator.structured_rectangle_quad(L_half, H, nelx, nely)
        n_dofs_per_node = 2
        dofs_per_elem = 8
        
    nnodes = coords.shape[0]
    ndofs = nnodes * n_dofs_per_node
    
    print(f"Mesh: {nelx}x{nely} elements")
    print(f"Nodes: {nnodes}, DOFs: {ndofs}")

    # 3. Assembly
    # -----------
    K = np.zeros((ndofs, ndofs))
    f = np.zeros((ndofs, 1))
    
    for i in range(len(edofs)):
        elem_dofs = edofs[i] # 1-based indices
        
        # Get node coordinates for this element
        # Convert dofs to node indices to fetch coords
        # Note: elem_dofs are [2*n1+1, 2*n1+2, ...]. Node index = (dof-1)//2
        node_indices = [(d-1)//2 for d in elem_dofs[0::2]] 
        el_nodes = coords[node_indices]
        
        if mesh_type == 'cst':
            Ke, fe = cst_element(el_nodes, D, thickness, body_load)
        else:
            Ke, fe = bilinear_element(el_nodes, D, thickness, body_load, ngp=4)
            
        assem(K, Ke, elem_dofs)
        assem(f, fe, elem_dofs)

    # 4. Boundary Conditions
    # ----------------------
    bc_dofs = []
    bc_vals = []
    
    # Identify nodes
    tol = 1e-6
    
    # A. Left Support (x=0)
    # Simply supported usually implies u_y = 0 at the support point.
    # For a deep beam/continuum, usually fixed at the bottom corner (y = -H/2).
    # Since we are modeling half, we don't fix u_x here (that would overconstrain with symmetry).
    # However, to prevent rigid body translation in X if not using symmetry? 
    # But we ARE using symmetry.
    
    # Find bottom-left node: x approx 0, y approx -H/2
    node_left_bottom = -1
    for i, (x, y) in enumerate(coords):
        if abs(x) < tol and abs(y - (-H/2)) < tol:
            node_left_bottom = i
            break
            
    if node_left_bottom != -1:
        # Fix uy only (DOF 2)
        bc_dofs.append(2 * node_left_bottom + 2)
        bc_vals.append(0.0)
    else:
        print("Warning: Left support node not found!")

    # B. Right Symmetry (x = L_half)
    # Symmetry condition: u_x = 0 along the entire edge
    nodes_right = []
    for i, (x, y) in enumerate(coords):
        if abs(x - L_half) < tol:
            # Fix ux (DOF 1)
            bc_dofs.append(2 * i + 1)
            bc_vals.append(0.0)

    # 5. Solve
    # --------
    a, r = solve_eq(K, f, bc_dofs, bc_vals)

    # 6. Post-Processing
    # ------------------
    
    # A. Deflection at Right Edge (Center of full beam)
    # We want average vertical deflection at x = L_half
    right_nodes_indices = [i for i, (x, y) in enumerate(coords) if abs(x - L_half) < tol]
    
    disp_y_sum = 0
    for idx in right_nodes_indices:
        disp_y_sum += a[2*idx + 1] # uy is 2nd dof (0-based index)
        
    avg_deflection = disp_y_sum / len(right_nodes_indices)
    
    # B. Max Stress (Sigma_xx) at Center (Right edge, top/bottom fiber or center?)
    # Problem asks for "maximum normal stress at the same horizontal position" (right edge).
    # Usually max stress is at y = +/- H/2.
    
    max_sigma_xx = 0.0
    
    # Loop elements to find stress
    # Note: For CST stress is constant. For Quad it is at center.
    # To find max stress at x=L, we look for elements close to x=L.
    
    for i in range(len(edofs)):
        elem_dofs = edofs[i]
        ed = extract_dofs(a, np.array([elem_dofs])) # returns (1, ndofs)
        
        node_indices = [(d-1)//2 for d in elem_dofs[0::2]] 
        el_nodes = coords[node_indices]
        
        # Check if element is on the right boundary
        center_x = np.mean(el_nodes[:,0])
        
        if center_x > (L_half - (L_half/nelx)*1.5): # Roughly the last column of elements
            if mesh_type == 'cst':
                sigma, _ = cst_element_stress_strain(el_nodes, D, ed[0])
            else:
                sigma, _ = bilinear_element_stress_strain(el_nodes, D, ed[0])
            
            # Sigma vector is [s_xx, s_yy, s_xy]
            s_xx = abs(sigma[0])
            if s_xx > max_sigma_xx:
                max_sigma_xx = s_xx

    print(f"Results for {mesh_type.upper()}:")
    print(f"  Avg Deflection at center (x={L_half}): {avg_deflection:.6e} m")
    print(f"  Max Normal Stress (approx):         {max_sigma_xx:.6e} Pa")

    # Analytic check (Beam Theory)
    # 5*q*L^4 / (384*EI) for simply supported? 
    # Here: Self weight q = rho*g*A_sect = rho*g*H*t (Load per length)
    # Max deflection simply supported: 5 w L^4 / 384 EI
    # I = t*H^3 / 12
    # w (load/m) = rho * g * H * t
    
    I = thickness * H**3 / 12
    w = rho * g * H * thickness
    L_full = W_total
    
    delta_analytic = -(5 * w * L_full**4) / (384 * E * I)
    sigma_analytic = (w * L_full**2 / 8) * (H/2) / I # M*y/I
    
    print(f"Analytic Reference:")
    print(f"  Deflection: {delta_analytic:.6e} m")
    print(f"  Max Stress: {sigma_analytic:.6e} Pa")
    
    return avg_deflection, max_sigma_xx


if __name__ == "__main__":
    
    # Verify the Quad implementation using the data from the starting_point notebook
    print("--- Verifying Quad Implementation ---")
    nodes_ver = np.array([[0.1, 0.0], [1.0, 0.0], [1.2, 1.0], [0.0, 1.3]])
    N_ver, B_ver, detJ_ver = compute_Ne_Be_detJ(nodes_ver, 0.15, 0.25)
    print(f"DetJ calculated: {detJ_ver:.7f} (Ref: 0.3099375)")
    
    # Run Task 1
    solve_problem(mesh_type='cst')
    
    # Run Task 2
    solve_problem(mesh_type='quad')
#%%