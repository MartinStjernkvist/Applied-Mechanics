import numpy as np
import matplotlib.pyplot as plt
from mha021 import * # Import provided utilities

# ==============================================================================
# PART 1: TASK 2 IMPLEMENTATION (Bilinear Quad Element)
# ==============================================================================

def compute_Ne_Be_detJ(nodes, ξ, η):
    """
    Compute shape functions (N), B-matrix, and Jacobian determinant for a 
    bilinear isoparametric element at local coordinates (ξ, η).
    
    Ref: Task 2(a) 
    """
    # 1. Shape functions (1x4)
    # Order: (-1,-1), (1,-1), (1,1), (-1,1) -> Counter-clockwise
    Ne = 0.25 * np.array([
        (1 - ξ) * (1 - η),
        (1 + ξ) * (1 - η),
        (1 + ξ) * (1 + η),
        (1 - ξ) * (1 + η)
    ])
    
    # Construct N matrix (2x8)
    N = np.zeros((2, 8))
    N[0, 0::2] = Ne
    N[1, 1::2] = Ne

    # 2. Derivatives w.r.t local coords (2x4)
    # dN_dξ
    dN_dxi = 0.25 * np.array([-(1-η), (1-η), (1+η), -(1+η)])
    # dN_dη
    dN_deta = 0.25 * np.array([-(1-ξ), -(1+ξ), (1+ξ), (1-ξ)])
    
    dNe_local = np.vstack((dN_dxi, dN_deta))

    # 3. Jacobian matrix J = dNe_local @ nodes (2x2)
    J = dNe_local @ nodes
    detJ = np.linalg.det(J)
    
    if detJ <= 1e-16:
        raise ValueError(f"Bad element geometry (detJ={detJ}). Check node ordering.")

    # 4. Derivatives w.r.t global coords: inv(J) @ dNe_local
    dNe_global = np.linalg.solve(J, dNe_local)  # (2x4) [[dN/dx], [dN/dy]]
    
    # 5. Construct B-matrix (3x8)
    # [ dN/dx       0   ]
    # [   0       dN/dy ]
    # [ dN/dy     dN/dx ]
    Be = np.zeros((3, 8))
    Be[0, 0::2] = dNe_global[0, :]  # dN/dx -> strain_xx
    Be[1, 1::2] = dNe_global[1, :]  # dN/dy -> strain_yy
    Be[2, 0::2] = dNe_global[1, :]  # dN/dy -> shear
    Be[2, 1::2] = dNe_global[0, :]  # dN/dx -> shear

    return N, Be, detJ

def bilinear_element(nodes, D, t, body_load, ngp):
    """
    Compute stiffness matrix Ke and force vector fe for a bilinear quad 
    using Gauss integration.
    
    Ref: Task 2(a) 
    """
    nodes = np.asarray(nodes)
    b_vec = np.asarray(body_load).reshape(2) # [bx, by]
    
    Ke = np.zeros((8, 8))
    fe = np.zeros(8)
    
    # Get Gauss points and weights from mha021
    coords, weights = gauss_integration_rule(int(np.sqrt(ngp))) 

    # Loop over Gauss points (tensor product)
    for i, ξ in enumerate(coords):
        w_xi = weights[i]
        for j, η in enumerate(coords):
            w_eta = weights[j]
            weight_factor = w_xi * w_eta * t
            
            # Get element matrices at this integration point
            N, Be, detJ = compute_Ne_Be_detJ(nodes, ξ, η)
            
            # Integrate Stiffness: B.T @ D @ B * detJ * weight
            Ke += Be.T @ D @ Be * detJ * weight_factor
            
            # Integrate Body Force: N.T @ b * detJ * weight
            # N.T is (8,2), b is (2,)
            fe += (N.T @ b_vec) * detJ * weight_factor
            
    return Ke, fe

def bilinear_element_stress_strain(nodes, D, ae):
    """
    Compute stress and strain at the element center (ξ=0, η=0).
    
    Ref: Task 2(b) [cite: 122, 124]
    """
    # Evaluate at centroid
    _, Be, _ = compute_Ne_Be_detJ(nodes, 0.0, 0.0)
    
    strain = Be @ ae      # [eps_xx, eps_yy, gamma_xy]
    stress = D @ strain   # [sig_xx, sig_yy, sig_xy]
    
    return stress, strain

# ==============================================================================
# PART 2: ANALYSIS & SOLVER (TASK 1 & 2)
# ==============================================================================

def run_analysis(element_type='cst', nelx=60, nely=10):
    """
    Sets up and solves the beam problem for Task 1 (CST) or Task 2 (Quad).
    Includes mesh generation, BC application (symmetry), and verification.
    """
    print(f"\n--- Running Analysis: {element_type.upper()} Elements ({nelx}x{nely}) ---")

    # 1. Problem Parameters [cite: 47, 48]
    W_full = 5.0
    L_model = W_full / 2.0  # Analyzing left half
    H = 0.4
    thickness = 0.01
    E = 210e9
    nu = 0.3
    rho = 7800
    g = 9.81
    body_load = [0.0, -rho * g]
    
    # Plane Stress D-Matrix [cite: 47]
    D = hooke_2d_plane_stress(E, nu)

    # 2. Mesh Generation
    # Task 1 uses CST [cite: 44, 70], Task 2 uses Semi-structured Quad [cite: 101, 132]
    if element_type == 'cst':
        # structured_rectangle_mesh returns a Mesh object in updated mha021 logic?
        # Checking mha021 provided: returns dict for structured, Mesh obj for semi.
        # We handle the discrepancy below.
        mesh_data = MeshGenerator.structured_rectangle_mesh(
            L_model, H, nelx, nely, element_type='tri', origin=(0, -H/2)
        )
        nodes = mesh_data.nodes
        elements = mesh_data.elements
        # Edges are in the mesh_data.edges dict
        edges = mesh_data.edges
        dofs_per_elem = 6
    else:
        # Task 2: Semi-structured quad mesh
        mesh = MeshGenerator.semistructured_rectangle_mesh_quads(
            L_model, H, nx=nelx, ny=nely, origin=(0, -H/2)
        )
        nodes = mesh.nodes
        elements = mesh.elements
        edges = mesh.edges
        dofs_per_elem = 8

    ndofs = nodes.shape[0] * 2
    K = np.zeros((ndofs, ndofs))
    f = np.zeros(ndofs)

    # 3. Assembly [cite: 65]
    print("Assembling stiffness matrix...")
    
    # Pre-compute edofs map
    edof_map = build_edof(elements, dofs_per_node=2) # (Nel, dofs_per_elem)

    for el_idx in range(elements.shape[0]):
        el_nodes_idx = elements[el_idx] - 1  # 0-based
        el_coor = nodes[el_nodes_idx]
        
        if element_type == 'cst':
            Ke, fe = cst_element(el_coor, D, thickness, body_load)
        else:
            Ke, fe = bilinear_element(el_coor, D, thickness, body_load, ngp=4)
        
        # Assemble using mha021.assem (requires 1-based indices)
        el_dofs = edof_map[el_idx] # 1-based DOFs
        assem(K, Ke, el_dofs)
        assem(f, fe, el_dofs)

    # 4. Boundary Conditions [cite: 62, 63]
    # Left Half Model:
    # 1. Symmetry at Right Edge (x = 2.5): u_x = 0
    # 2. Support at Left Edge (x = 0): "Change in support" -> Pin at (0, -H/2) implies u_x=u_y=0.
    
    bc_dofs = []
    bc_vals = []
    
    # A. Symmetry Condition (Right edge)
    # edges['right'] contains node indices (1-based)
    right_nodes = edges['right']
    for node in right_nodes:
        dof_x = 2 * (node - 1) + 1 # 1-based DOF x
        bc_dofs.append(dof_x)
        bc_vals.append(0.0)

    # B. Support Condition (Left Bottom Node)
    # Find node at (0, -H/2). In structured mesh, this is usually node 1 or in edges['left']
    left_nodes = edges['left']
    # Identify the bottom-most node on the left edge
    min_y = np.min(nodes[left_nodes - 1, 1])
    support_node = None
    for node in left_nodes:
        if np.isclose(nodes[node-1, 1], min_y):
            support_node = node
            break
            
    if support_node:
        # Fix u_x and u_y (Pin)
        bc_dofs.append(2*(support_node-1) + 1) # u_x
        bc_dofs.append(2*(support_node-1) + 2) # u_y
        bc_vals.append(0.0)
        bc_vals.append(0.0)
    else:
        print("Warning: Support node not found.")

    # 5. Solve [cite: 65]
    a, r = solve_eq(K, f, bc_dofs, bc_vals)

    # 6. Post-Processing & Verification [cite: 69, 73]
    
    # A. Average Deflection at Right Edge (Symmetry Plane)
    # Extract u_y (DOF 2) for all nodes on the right edge
    right_dofs_y = [2*(n-1) + 2 for n in right_nodes] # 0-based index in a is (dof-1)
    disp_y_right = a[np.array(right_dofs_y) - 1] # -1 for Python indexing
    avg_deflection = np.mean(disp_y_right)
    
    # B. Max Normal Stress (Sigma_xx) at Right Edge (Symmetry Plane)
    # We look for elements connected to the right boundary
    max_sigma_xx = 0.0
    
    # Iterate elements to calculate stress
    for el_idx in range(elements.shape[0]):
        el_nodes_idx = elements[el_idx] - 1
        el_coor = nodes[el_nodes_idx]
        
        # Check if element is at the right boundary (check centroid)
        centroid_x = np.mean(el_coor[:, 0])
        if centroid_x > (L_model - 0.1): # Close to right edge
            
            el_dofs = edof_map[el_idx]
            ae = a[el_dofs - 1]
            
            if element_type == 'cst':
                sigma, _ = cst_element_stress_strain(el_coor, D, ae)
            else:
                sigma, _ = bilinear_element_stress_strain(el_coor, D, ae)
            
            # sigma = [sig_xx, sig_yy, sig_xy]
            if abs(sigma[0]) > max_sigma_xx:
                max_sigma_xx = abs(sigma[0])

    # C. Analytical Solution (Beam Theory) 
    # Simply supported beam with uniform load q = rho*g*A
    # Max deflection (center): 5 * q * L^4 / (384 * E * I)
    q = rho * g * (H * thickness)
    I = thickness * H**3 / 12
    # L here is the FULL length (W_full)
    delta_analytic = -(5 * q * W_full**4) / (384 * E * I)
    
    # Max Stress (My/I): M_max = qL^2/8
    M_max = q * W_full**2 / 8
    sigma_analytic = M_max * (H/2) / I

    # Report
    print(f"  Results:")
    print(f"  FEM Avg Deflection (x=L/2): {avg_deflection:.5e} m")
    print(f"  Beam Theory Deflection:     {delta_analytic:.5e} m")
    print(f"  Error:                      {abs((avg_deflection - delta_analytic)/delta_analytic)*100:.2f} %")
    print("-" * 30)
    print(f"  FEM Max Stress (x=L/2):     {max_sigma_xx:.5e} Pa")
    print(f"  Beam Theory Stress:         {sigma_analytic:.5e} Pa")
    print(f"  Error:                      {abs((max_sigma_xx - sigma_analytic)/sigma_analytic)*100:.2f} %")
    
    # Plotting (Optional visual check)
    if element_type == 'quad':
        ed_matrix = extract_dofs(a, edof_map) # (Nel, 8)
        # Using a scalar field for visualization (e.g., displacement magnitude)
        stress_field = np.zeros(elements.shape[0]) # Placeholder for element plot
        # Re-calc stress for all elements for plotting
        for i in range(elements.shape[0]):
             ae = ed_matrix[i,:]
             el_c = nodes[elements[i]-1]
             s, _ = bilinear_element_stress_strain(el_c, D, ae)
             stress_field[i] = s[0] # Sigma_xx

        # Use mha021 plotting
        fig = plot_element_values(nodes, elements, ed_matrix, stress_field, 
                                  title="Sigma_xx Distribution (Quad)", scale=100)
        # fig.show() # Uncomment to display in notebook/browser

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
if __name__ == "__main__":
    # Task 1: CST Element [cite: 44]
    # Needs fine mesh for CST to converge on bending problems (locking/stiff behavior)
    run_analysis(element_type='cst', nelx=80, nely=8)
    
    # Task 2: Bilinear Quad Element [cite: 100]
    # Quads generally converge faster for bending
    run_analysis(element_type='quad', nelx=80, nely=8)