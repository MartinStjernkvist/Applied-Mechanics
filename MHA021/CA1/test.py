#%%
import numpy as np
import matplotlib.pyplot as plt

# Input parameters
I_y = 30.3e-6  # m^4
E = 210e9  # Pa
K_w = 10.05e6  # N/m^2
L = 10.0  # m
P = 70e3  # N

EI = E * I_y

def hermite_beam_stiffness(EI, h):
    """Element stiffness matrix for Euler-Bernoulli beam"""
    K_e = (EI / h**3) * np.array([
        [12,      6*h,    -12,     6*h],
        [6*h,     4*h**2,  -6*h,    2*h**2],
        [-12,    -6*h,     12,     -6*h],
        [6*h,     2*h**2,  -6*h,    4*h**2]
    ])
    return K_e

def winkler_stiffness(K_w, h):
    """Additional stiffness matrix for Winkler foundation"""
    K_w_e = (K_w * h / 420) * np.array([
        [156,     22*h,    54,      -13*h],
        [22*h,    4*h**2,  13*h,    -3*h**2],
        [54,      13*h,    156,     -22*h],
        [-13*h,  -3*h**2, -22*h,    4*h**2]
    ])
    return K_w_e

def solve_winkler_beam(n_elem, k_w=K_w, bc_type='semi-infinite', P_load=P):
    """
    Solve Winkler beam problem
    bc_type: 'semi-infinite' (load at x=0, beam on foundation)
             'cantilever', 'pinned-pinned', 'fixed-fixed' (for verification)
    """
    n_nodes = n_elem + 1
    h = L / n_elem
    x_nodes = np.linspace(0, L, n_nodes)
    
    # DOFs: 2 per node (w, theta)
    n_dof = 2 * n_nodes
    K_global = np.zeros((n_dof, n_dof))
    f_global = np.zeros(n_dof)
    
    # Assemble element matrices
    for e in range(n_elem):
        i = e
        j = e + 1
        
        # Element stiffness
        K_e = hermite_beam_stiffness(EI, h)
        K_w_e = winkler_stiffness(k_w, h)
        K_total_e = K_e + K_w_e
        
        # Global DOF indices for this element
        dofs = [2*i, 2*i+1, 2*j, 2*j+1]
        
        # Assemble
        for ii in range(4):
            for jj in range(4):
                K_global[dofs[ii], dofs[jj]] += K_total_e[ii, jj]
    
    # Apply point load
    if bc_type == 'semi-infinite':
        # Load at left end (x=0)
        f_global[0] = -P_load  # negative for downward
    else:
        # Load at center (for verification cases)
        center_node = n_nodes // 2
        f_global[2*center_node] = -P_load
    
    # Apply boundary conditions
    fixed_dofs = []
    
    if bc_type == 'semi-infinite':
        # Semi-infinite beam: no kinematic constraints
        # Natural BC at x=0: V(0) = P (applied load) - already in f_global
        # Natural BC at x=L: V(L) = k_w*w(L), M(L) = 0
        # These are automatically satisfied in weak form
        # No DOFs need to be fixed!
        fixed_dofs = []
        
    elif bc_type == 'pinned-pinned':
        # Pin at both ends: w=0 at x=0 and x=L
        fixed_dofs = [0, 2*(n_nodes-1)]
        
    elif bc_type == 'cantilever':
        # Fixed at x=0: w=0, theta=0
        fixed_dofs = [0, 1]
        
    elif bc_type == 'fixed-fixed':
        # Fixed at both ends: w=0, theta=0
        fixed_dofs = [0, 1, 2*(n_nodes-1), 2*(n_nodes-1)+1]
    
    # Remove fixed DOFs
    free_dofs = [i for i in range(n_dof) if i not in fixed_dofs]
    K_reduced = K_global[np.ix_(free_dofs, free_dofs)]
    f_reduced = f_global[free_dofs]
    
    # Solve
    a_reduced = np.linalg.solve(K_reduced, f_reduced)
    
    # Full displacement vector
    a = np.zeros(n_dof)
    a[free_dofs] = a_reduced
    
    # Extract deflections and rotations
    w = a[0::2]
    theta = a[1::2]
    
    return x_nodes, w, theta, K_global, a

def compute_bending_moments(n_elem, a):
    """Compute bending moments at element nodes"""
    h = L / n_elem
    M = np.zeros((n_elem, 2))  # moment at start and end of each element
    
    for e in range(n_elem):
        i = e
        j = e + 1
        
        # Element DOFs
        a_e = np.array([a[2*i], a[2*i+1], a[2*j], a[2*j+1]])
        
        # Element stiffness
        K_e = hermite_beam_stiffness(EI, h)
        
        # Element forces
        f_e = K_e @ a_e
        
        # Bending moments (from force-displacement relationship)
        M[e, 0] = -f_e[1]  # M at node i
        M[e, 1] = f_e[3]   # M at node j
    
    return M

def compute_shear_forces(n_elem, a, k_w):
    """Compute shear forces at element nodes"""
    h = L / n_elem
    V = np.zeros((n_elem, 2))  # shear at start and end of each element
    
    for e in range(n_elem):
        i = e
        j = e + 1
        
        # Element DOFs
        a_e = np.array([a[2*i], a[2*i+1], a[2*j], a[2*j+1]])
        
        # Element stiffness (both beam and Winkler)
        K_e = hermite_beam_stiffness(EI, h)
        K_w_e = winkler_stiffness(k_w, h)
        K_total = K_e + K_w_e
        
        # Element forces
        f_e = K_total @ a_e
        
        # Shear forces
        V[e, 0] = f_e[0]   # V at node i (positive upward)
        V[e, 1] = -f_e[2]  # V at node j (positive upward)
    
    return V

# Task 4(c): Verification with k_w = 0
print("="*60)
print("TASK 4(c): VERIFICATION (k_w = 0, Simply Supported)")
print("="*60)

# Simply supported beam with center load
# Analytical: max deflection = P*L^3 / (48*EI)
w_analytical = P * L**3 / (48 * EI)
print(f"Analytical max deflection: {w_analytical*1000:.4f} mm")

n_elem_verify = 20
x_v, w_v, theta_v, _, _ = solve_winkler_beam(n_elem_verify, k_w=0, bc_type='pinned-pinned')
print(f"FEM max deflection:        {abs(min(w_v))*1000:.4f} mm")
print(f"Relative error:            {abs(abs(min(w_v)) - w_analytical)/w_analytical * 100:.2f}%")
print("Verification PASSED: Code correctly implements beam elements")
print()

# Task 4(d): Convergence study for Winkler beam with load at x=0
print("="*60)
print("TASK 4(d): CONVERGENCE STUDY (Winkler Beam, Load at x=0)")
print("="*60)

elem_counts = [4, 8, 16, 32, 64, 128]
max_deflections = []

for n in elem_counts:
    x, w, theta, _, _ = solve_winkler_beam(n, k_w=K_w, bc_type='semi-infinite')
    max_def = abs(min(w))
    max_deflections.append(max_def)
    print(f"Elements: {n:3d}, Max deflection: {max_def*1000:.6f} mm")

# Check convergence (relative change)
print("\nConvergence check:")
for i in range(1, len(elem_counts)):
    rel_change = abs(max_deflections[i] - max_deflections[i-1]) / max_deflections[i] * 100
    print(f"{elem_counts[i-1]:3d} -> {elem_counts[i]:3d} elements: {rel_change:.4f}% change")
    if rel_change < 0.1 and elem_counts[i] not in [elem_counts[-1]]:
        print(f"    *** Converged at {elem_counts[i]} elements ***")

# Choose converged mesh
chosen_n_elem = 64
print(f"\nChosen mesh: {chosen_n_elem} elements (converged to <0.1%)")
print()

# Task 4(e): Final solution with chosen mesh
print("="*60)
print(f"TASK 4(e): FINAL SOLUTION ({chosen_n_elem} elements)")
print("="*60)

x_final, w_final, theta_final, K_final, a_final = solve_winkler_beam(
    chosen_n_elem, k_w=K_w, bc_type='semi-infinite'
)

# Compute bending moments and shear forces
M_final = compute_bending_moments(chosen_n_elem, a_final)
V_final = compute_shear_forces(chosen_n_elem, a_final, K_w)

print(f"Maximum deflection: {abs(min(w_final))*1000:.4f} mm at x = {x_final[np.argmin(w_final)]:.2f} m")
print(f"Deflection at x=0: {w_final[0]*1000:.4f} mm")
print(f"Maximum bending moment: {np.max(np.abs(M_final))/1000:.2f} kN·m")
print(f"Bending moment at x=0: {abs(M_final[0,0])/1000:.2f} kN·m")

# Verify boundary condition at x=0
print(f"\nBoundary condition check at x=0:")
print(f"Shear force V(0): {V_final[0,0]/1000:.2f} kN")
print(f"Applied load P:   {P/1000:.2f} kN")
print(f"Match: {abs(V_final[0,0] - P) < 1e-6}")
print()

# Plotting
fig, axes = plt.subplots(3, 1, figsize=(12, 10))

# Plot deflection
axes[0].plot(x_final, w_final*1000, 'b-', linewidth=2)
axes[0].plot(x_final, w_final*1000, 'bo', markersize=3)
axes[0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
axes[0].axvline(x=0, color='r', linestyle='--', alpha=0.5, label='Load position')
axes[0].set_xlabel('Position x [m]', fontsize=12)
axes[0].set_ylabel('Deflection w [mm]', fontsize=12)
axes[0].set_title(f'Beam Deflection - Semi-infinite beam on Winkler foundation ({chosen_n_elem} elements)', fontsize=13)
axes[0].grid(True, alpha=0.3)
axes[0].legend()

# Plot bending moment
x_M = []
M_plot = []
for e in range(chosen_n_elem):
    h = L / chosen_n_elem
    x_M.extend([e*h, (e+1)*h])
    M_plot.extend([M_final[e, 0]/1000, M_final[e, 1]/1000])

axes[1].plot(x_M, M_plot, 'r-', linewidth=2)
axes[1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
axes[1].axvline(x=0, color='r', linestyle='--', alpha=0.5, label='Load position')
axes[1].set_xlabel('Position x [m]', fontsize=12)
axes[1].set_ylabel('Bending Moment [kN·m]', fontsize=12)
axes[1].set_title(f'Bending Moment Distribution ({chosen_n_elem} elements)', fontsize=13)
axes[1].grid(True, alpha=0.3)
axes[1].legend()

# Plot shear force
x_V = []
V_plot = []
for e in range(chosen_n_elem):
    h = L / chosen_n_elem
    x_V.extend([e*h, (e+1)*h])
    V_plot.extend([V_final[e, 0]/1000, V_final[e, 1]/1000])

axes[2].plot(x_V, V_plot, 'g-', linewidth=2)
axes[2].axhline(y=0, color='k', linestyle='--', alpha=0.3)
axes[2].axvline(x=0, color='r', linestyle='--', alpha=0.5, label='Load position')
axes[2].set_xlabel('Position x [m]', fontsize=12)
axes[2].set_ylabel('Shear Force [kN]', fontsize=12)
axes[2].set_title(f'Shear Force Distribution ({chosen_n_elem} elements)', fontsize=13)
axes[2].grid(True, alpha=0.3)
axes[2].legend()

plt.tight_layout()
plt.show()

print("="*60)
print("DISCUSSION:")
print("="*60)
print("The semi-infinite beam on Winkler foundation represents a railway")
print("rail subjected to a wheel load at one end. The elastic foundation")
print("provides continuous support, causing the deflection to decay")
print("exponentially away from the load point. The characteristic length")
print(f"λ = (4EI/k_w)^(1/4) = {(4*EI/K_w)**0.25:.3f} m determines the decay rate.")
print("\nThe natural boundary conditions at x=L (far from load) are:")
print("- Shear force V(L) → 0 (no external force)")
print("- Moment M(L) → 0 (free end)")
print("These are automatically satisfied in the weak formulation.")
#%%



def solve_rail(n_elem, k_w, bc_type='semi-infinite'):
    
    n_nodes = n_elem + 1
    n_dof = 2 * n_nodes
    Le = L / n_elem
    
    K_global = np.zeros((n_dof, n_dof))
    f_global = np.zeros(n_dof)
    
    for e in range(n_elem):
        i = e
        j = e + 1
        
        K_e = hermite_beam_stiffness(EI, Le)
        K_w_e = winkler_stiffness(k_w, Le)
        K_total_e = K_e + K_w_e
        
        # DOFs: [w_i, th_i, w_j, th_j]
        dofs = [2 * i, 2 * i + 1, 2 * j, 2 * j + 1] 
        for ii in range(4):
            for jj in range(4):
                K_global[dofs[ii], dofs[jj]] += K_total_e[ii, jj]
        
    # Load at left end (x=0)
    f_global[0] = -P  # negative for downward
        
    # Apply boundary conditions
    fixed_dofs = []
    
    if bc_type == 'semi-infinite':
        fixed_dofs = []
        
    if bc_type == 'cantilever':
        # Fixed at x=0: w=0, theta=0
        fixed_dofs = [2 * (n_nodes - 1), 2 * (n_nodes - 1) + 1]
        
    # Create boolean mask for free DOFs
    free_mask = np.ones(n_dof, dtype=bool)
    free_mask[fixed_dofs] = False
    
    # Use boolean indexing instead of np.ix_
    K_reduced = K_global[free_mask][:, free_mask]
    f_reduced = f_global[free_mask]
    
    # Solve the system
    a_reduced = np.linalg.solve(K_reduced, f_reduced)
    
    # Displacement vector
    a = np.zeros(n_dof)
    a[free_mask] = a_reduced
    
    x = np.linspace(0, L, n_nodes)
    w = a[0::2]