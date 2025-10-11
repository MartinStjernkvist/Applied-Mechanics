#%%
# %matplotlib widget
from quadmesh import *

import numpy as np
import matplotlib.pyplot as plt
from sympy import *
import math
import pandas as pd

from IPython.display import display, Math
from mpl_toolkits.mplot3d import axes3d
from mpl_toolkits.mplot3d import Axes3D
from numpy.random import rand
from IPython.display import HTML
from matplotlib import animation
import scipy.io as sio
from scipy.optimize import fsolve
from matplotlib import rcParams # for changing default values
import matplotlib.ticker as ticker

import calfem.core as cfc
import calfem.vis_mpl as cfv
import calfem.mesh as cfm
import calfem.utils as cfu

from scipy.sparse import coo_matrix, csr_matrix
import matplotlib.cm as cm

from pathlib import Path

def new_prob(string):
    print_string = '\n--------------------------------------------\n' + 'Assignment ' + str(string) + '\n--------------------------------------------\n'
    return print(print_string)

SMALL_SIZE = 10
MEDIUM_SIZE = 12
BIGGER_SIZE = 14
dpi = 500

# Set the global font sizes
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.rc('figure', figsize=(8,4))

script_dir = Path(__file__).parent

def sfig(fig_name):
    fig_output_file = script_dir / "figures2" / fig_name
    fig_output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_output_file, dpi=dpi, bbox_inches='tight')
    
def fig(fig_name):
    '''
    standard matplotlib commands
    '''
    plt.legend()
    plt.grid(True, alpha = 0.3)
    sfig(fig_name)
    plt.show()

#%%
####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################



# GIVEN INFORMATION



####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################

# --------------------------------------------
# ASSIGNMENT 1 - Beam
# --------------------------------------------

L1 = 3
L2 = 0.3

sigma_yield = 550 * 10**6 
E_num = 220 * 10**9
b_num = 0.05
h_num = b_num
I_num= b_num * h_num**3 / 12 #moment of inertia
rho_num = 7800 #density
g_num = 9.81
m_num = 130
P_num = -m_num * g_num
poisson_num = 0.3
q0_num = -h_num * b_num * rho_num * g_num

Ks_num = 5/6
A_num = b_num * h_num
G_num = E_num / (2 * (1 + poisson_num)) 

# --------------------------------------------
# ASSIGNMENT 2 - Axissymetry
# --------------------------------------------

a_num = 150 # mm
b_num = 250 # mm
h0_num = 20 # mm
p_num = 120 * 10**6 # Pa
E2_num = 200 * 10**9 # Pa

# for calfem calculations:
thickness = h0_num
Emod = E2_num
# lower left corner
p1 = [a_num, 0]
# upper right corner
p2 = [b_num, h0_num]


#%% 

####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################



# Assigment 1 - EULER-BERNOULLI


####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################
new_prob('1 - EULER-BERNOULLI')

x, L, q0, P, E, I = symbols('x L q0 P E I')

w = Function('w')(x) # w is a function of x

diffeq1 = Eq(E * I * diff(w, x, 4), q0)

w = dsolve(diffeq1, w).rhs

M = -E * I * w.diff(x, 2)

# Boundary conditions for distributed load
boundary_conditions = [ 
                        w.subs(x, 0),                               #w(0) = 0
                        w.diff(x).subs(x, 0),                       #w'(0) = 0
                        M.subs(x, L),                               #w''(L) = 0
                        w.diff(x, 3).subs(x, L) - P / (-E * I)      #w'''(L) = -P
                        ]
print('\nboundary conditions:')
display(boundary_conditions)

integration_constants = solve(boundary_conditions, 'C1, C2, C3, C4', real=True)
print('\nintegration constants:')
display(integration_constants)

# displacement function with constants substituted
solution = w.subs(integration_constants)
display(simplify(solution))
w_func = lambdify((x, L, q0, P, E, I), solution, 'numpy')

# moment function with constants substituted
M_solution = M.subs(integration_constants)
M_func = lambdify((x, L, q0, P, E, I), M_solution, 'numpy')

def euler_bernoulli_analysis(L):
    
    x_vals = np.linspace(0, L, 200)
    w_vals = w_func(x_vals, L, q0_num, P_num, E_num, I_num)
    M_vals = M_func(x_vals, L, q0_num, P_num, E_num, I_num)
    
    # Compute stresses at z = -h/2 (bottom surface)
    z = -h_num/2
    sigma_xx = -M_vals * z / I_num  
    sigma_vM = np.abs(sigma_xx)  

    # maximum stresses
    max_sigma_xx = np.max(np.abs(sigma_xx))
    max_sigma_vM = np.max(sigma_vM)
    max_stress_location = x_vals[np.argmax(sigma_vM)]

    safety_factor = sigma_yield / max_sigma_vM
    will_yield = max_sigma_vM > sigma_yield
    
    # Plot 1: Deflection
    plt.figure()
    plt.plot(x_vals, w_vals*1e3, 'b-')
    plt.title(f'Beam Deflection at L={L}m')
    plt.xlabel('Position along beam (m)')
    plt.ylabel('Deflection (mm)')
    plt.grid(True, alpha=0.3)
    # plt.axhline(0, color='black', linestyle='--')
    plt.xlim(0, L)
    plt.tight_layout()
    sfig('deflection_' + str(L) + '.png')
    plt.show()

    # Plot 2: Normal stress σ_xx
    plt.figure()
    plt.plot(x_vals, sigma_xx/1e6, 'r-')
    plt.title('Normal Stress σ_xx at z=-h/2 (bottom surface)')
    plt.xlabel('Position along beam (m)')
    plt.ylabel('Normal Stress (MPa)')
    plt.grid(True, alpha=0.3)
    plt.axhline(0, color='black', linestyle='--')
    # plt.axhline(sigma_yield/1e6, color='orange', linestyle='--', label=f'Yield strength = {sigma_yield/1e6:.0f} MPa')
    # plt.axhline(-sigma_yield/1e6, color='orange', linestyle='--')
    plt.xlim(0, L)
    plt.legend()
    plt.tight_layout()
    sfig('sigmaxx_' + str(L) + '.png')
    plt.show()

    # Plot 3: von Mises stress
    plt.figure()
    plt.plot(x_vals, sigma_vM/1e6, 'g-')
    plt.title('von Mises Effective Stress σ_vM at z=-h/2')
    plt.xlabel('Position along beam (m)')
    plt.ylabel('von Mises Stress (MPa)')
    plt.grid(True, alpha=0.3)
    # plt.axhline(sigma_yield/1e6, color='orange', linestyle='--', label=f'Yield strength = {sigma_yield/1e6:.0f} MPa')
    plt.xlim(0, L)
    plt.legend()
    plt.tight_layout()
    sfig('vonmises_' + str(L) + '.png')
    plt.show()
    
    print(f"\n{'='*70}")
    print(f"STRESS ANALYSIS RESULTS FOR L={L}m")
    print(f"{'='*70}")
    print(f"Beam properties:")
    print(f"  Length: {L} m")
    print(f"  Cross-section: {b_num*1e3:.1f} mm × {h_num*1e3:.1f} mm")
    print(f"  Material: Steel (E = {E_num/1e9:.0f} GPa, σ_yield = {sigma_yield/1e6:.0f} MPa)")
    print(f"\nLoading:")
    print(f"\nStress at z = -h/2 (bottom surface, maximum tension):")
    print(f"  Maximum |σ_xx|: {max_sigma_xx/1e6:.2f} MPa")
    print(f"  Maximum σ_vM: {max_sigma_vM/1e6:.2f} MPa")
    print(f"  Location of max stress: x = {max_stress_location:.3f} m")
    print(f"\nYield assessment (von Mises criterion):")
    print(f"  Yield strength: {sigma_yield/1e6:.0f} MPa")
    print(f"  Safety factor: {safety_factor:.2f}")
    if will_yield:
        print(f"  ⚠️  BEAM WILL YIELD - Maximum stress exceeds yield strength!")
    else:
        print(f"  ✓  Beam is safe - No yielding expected")
    print(f"{'='*70}\n")
    
    # return x_vals, w_vals, M_vals, sigma_xx, sigma_vM, max_sigma_xx, sigma_vM, max_stress_location, safety_factor, will_yield
    return x_vals, w_vals, sigma_xx

# x_vals_L1, w_vals_L1, M_vals_L1, sigma_xx_L1, sigma_vM_L1, max_sigma_xx_L1, sigma_vM_L1, max_stress_location_L1, safety_factor_L1, will_yield_L1 = euler_bernoulli_analysis(L1)
# x_vals_L2, w_vals_L2, M_vals_L2, sigma_xx_L2, sigma_vM_L2, max_sigma_xx_L2, sigma_vM_L2, max_stress_location_L2, safety_factor_L2, will_yield_L2 = euler_bernoulli_analysis(L2)

x_vals_bernoulli_L1, w_vals_bernoulli_L1, sigma_xx_bernoulli_L1 = euler_bernoulli_analysis(L1)
x_vals_bernoulli_L2, w_vals_bernoulli_L2, sigma_xx_bernoulli_L2 = euler_bernoulli_analysis(L2)


# %%
####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################



# Assigment 1 - TIMOSHENKO



####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################
new_prob('1 - TIMOSHENKO')

x, q0, E, I, Ks, G, A, L, P = symbols('x q0 E I Ks G A L P', real=True)

f_phi = Function('phi') # phi is a function of x

## Define the differential equation in terms of phi
diffeq_phi = Eq(E * I * f_phi(x).diff(x, 3), q0)

## Solve the differential equation for phi(x) (eq. 3.35 LN)
phi = dsolve(diffeq_phi, f_phi(x)).rhs

## Solve the differential equation for w(x) (eq. 3.36 LN)
w = Function('w') # w is a function of x
diffeq_w = Eq(w(x).diff(x), -E*I/(G*Ks*A)*phi.diff(x,2) + phi)
w        = dsolve(diffeq_w, w(x)).rhs

## Define boundary conditions
M = -E * I * phi.diff(x)
# Boundary conditions for distributed load
boundary_conditions = [ w.subs(x, 0),                           #w(0) = 0
                        w.diff(x).subs(x, 0),                   #w'(0) = 0
                        M.subs(x, L),                           #w''(L) = 0
                        w.diff(x,3).subs(x, L) - P/(-E*I)]      #w'''(L) = -P

## Solve for the integration constants
integration_constants = solve(boundary_conditions, 'C1, C2, C3, C4', real=True)

## Substitute the integration constants into the solution
solution = w.subs(integration_constants)
display(solution)

w_func = lambdify((x, L, q0, P, E, I, Ks, A, G), solution, 'numpy')

# Create moment function with constants substituted
M_solution = M.subs(integration_constants)
M_func = lambdify((x, L, q0, P, E, I, Ks, A, G), M_solution, 'numpy')


def timoshenko_analysis(L):
    
    x_vals = np.linspace(0, L, 200)
    w_vals = w_func(x_vals, L, q0_num, P_num, E_num, I_num, A_num, G_num, Ks_num)
    M_vals = M_func(x_vals, L, q0_num, P_num, E_num, I_num, A_num, G_num, Ks_num)

    # Compute stresses at z = -h/2
    z = -h_num/2
    sigma_xx = -M_vals * z / I_num  # Normal stress from bending
    sigma_vM = np.abs(sigma_xx)  # von Mises stress

    # Find maximum stresses
    max_sigma_xx = np.max(np.abs(sigma_xx))
    max_sigma_vM = np.max(sigma_vM)
    max_stress_location = x_vals[np.argmax(sigma_vM)]
    safety_factor = sigma_yield / max_sigma_vM
    will_yield = max_sigma_vM > sigma_yield

    plt.figure()
    plt.plot(x_vals, w_vals*1e3, 'b-')
    plt.title(f'Beam Deflection at L={L}m (Timoshenko)')
    plt.xlabel('Position along beam (m)')
    plt.ylabel('Deflection (mm)')
    plt.grid(True, alpha=0.3)
    # plt.axhline(0, color='black', linestyle='--')
    plt.xlim(0, L)
    plt.tight_layout()
    sfig('deflection_timo_' + str(L) + '.png')
    plt.show()

    plt.figure()
    plt.plot(x_vals, sigma_xx/1e6, 'r-')
    plt.title('Normal Stress σ_xx at z=-h/2 (bottom surface)')
    plt.xlabel('Position along beam (m)')
    plt.ylabel('Normal Stress (MPa)')
    plt.grid(True, alpha=0.3)
    plt.axhline(0, color='black', linestyle='--')
    # plt.axhline(sigma_yield/1e6, color='orange', linestyle='--', label=f'Yield strength = {sigma_yield/1e6:.0f} MPa')
    # plt.axhline(-sigma_yield/1e6, color='orange', linestyle='--')
    plt.xlim(0, L)
    plt.legend()
    plt.tight_layout()
    sfig('sigmaxx_timo_' + str(L) + '.png')
    plt.show()

    plt.figure()
    plt.plot(x_vals, sigma_vM/1e6, 'g-')
    plt.title('von Mises Effective Stress σ_vM at z=-h/2')
    plt.xlabel('Position along beam (m)')
    plt.ylabel('von Mises Stress (MPa)')
    plt.grid(True, alpha=0.3)
    # plt.axhline(sigma_yield/1e6, color='orange', linestyle='--', label=f'Yield strength = {sigma_yield/1e6:.0f} MPa')
    plt.xlim(0, L)
    plt.legend()
    plt.tight_layout()
    sfig('vonmises_timo_' + str(L) + '.png')
    plt.show()


    sigma_MPa = sigma_xx / 1e6
    # Create a DataFrame
    df = pd.DataFrame({
        'Position (m)': x_vals,
        'Normal Stress (MPa)': sigma_MPa
    })

    # Save to CSV
    df.to_csv('normal_stress_timo_3m.csv', index=False)

    print(f"\n{'='*70}")
    print(f"STRESS ANALYSIS RESULTS FOR L={L}m (Timoshenko Beam Theory)")
    print(f"{'='*70}")
    print(f"Beam properties:")
    print(f"  Length: {L} m")
    print(f"  Cross-section: {b_num*1e3:.1f} mm × {h_num*1e3:.1f} mm")
    print(f"  Material: Steel (E = {E_num/1e9:.0f} GPa, G = {G_num/1e9:.1f} GPa)")
    print(f"  Shear correction factor Ks: {Ks_num:.3f}")
    print(f"  Yield strength: {sigma_yield/1e6:.0f} MPa")
    print(f"\nLoading:")
    print(f"  Distributed load q₀: {q0_num:.2f} N/m")
    print(f"  Point load P: {P_num:.2f} N (at free end)")
    print(f"\nStress at z = -h/2 (bottom surface, maximum tension):")
    print(f"  Maximum |σ_xx|: {max_sigma_xx/1e6:.2f} MPa")
    print(f"  Maximum σ_vM: {max_sigma_vM/1e6:.2f} MPa")
    print(f"  Location of max stress: x = {max_stress_location:.3f} m")
    print(f"\nYield assessment (von Mises criterion):")
    print(f"  Safety factor: {safety_factor:.2f}")
    print(f"{'='*70}\n")
    
    # return x_vals, w_vals, M_vals, sigma_xx, sigma_vM, max_sigma_xx, sigma_vM, max_sigma_xx, max_sigma_vM, max_stress_location,safety_factor, will_yield
    return x_vals, w_vals, sigma_xx

# run the analysis function
# x_vals_L1, w_vals_L1, M_vals_L1, sigma_xx_L1, sigma_vM_L1, max_sigma_xx_L1, sigma_vM_L1, max_sigma_xx_L1, max_sigma_vM_L1, max_stress_location_L1,safety_factor_L1, will_yield_L1 = timoshenko_analysis(L1)
# x_vals_L2, w_vals_L2, M_vals_L2, sigma_xx_L2, sigma_vM_L2, max_sigma_xx_L2, sigma_vM_L2, max_sigma_xx_L2, max_sigma_vM_L2, max_stress_location_L2,safety_factor_L2, will_yield_L2 = timoshenko_analysis(L2)

x_vals_timoshenko_L1, w_vals_timoshenko_L1, sigma_xx_timoshenko_L1 = timoshenko_analysis(L1)
x_vals_timoshenko_L2, w_vals_timoshenko_L2, sigma_xx_timoshenko_L2 = timoshenko_analysis(L2)

#%%
####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################



# Assignment 1 - calfem



####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################
new_prob('1 - calfem')

def calfem_analysis_A1(L):
    
    ptype = 1  # plane stress
    ep = [ptype, b_num]  # [ptype, thickness]
    Dmat = cfc.hooke(ptype, E_num, poisson_num)
    # Create mesh
    p1 = np.array([0., 0.])  # Lower left corner
    p2 = np.array([L, h_num])   # Upper right corner
    nelx = 100
    nely = 20
    ndof_per_node = 2
    nnode = (nelx + 1) * (nely + 1)
    nDofs = ndof_per_node * nnode

    # Generate mesh
    Ex, Ey, Edof, B1, B2, B3, B4, P1, P2, P3, P4 = quadmesh(p1, p2, nelx, nely, ndof_per_node)

    display('Ex', Ex)
    display('Ey', Ey)

    # Plot mesh
    cfv.figure()
    cfv.eldraw2(Ex, Ey)
    cfv.title('Mesh' + str(L))
    cfv.show()
    
    display(B2)

    # Initialize global stiffness matrix and force vector
    K = np.zeros([nDofs, nDofs])
    f = np.zeros([nDofs, 1])

    # Assemble stiffness matrix
    for eltopo, elx, ely in zip(Edof, Ex, Ey):
        Ke = cfc.planqe(elx, ely, ep, Dmat)
        cfc.assem(eltopo, K, Ke)

    # Aplying distributed load along the top edge downwards
    top_elements = range(nelx * (nely - 1), nelx * nely)

    for el_idx in top_elements:
        elx = Ex[el_idx, :]
        ely = Ey[el_idx, :]
        eltopo = Edof[el_idx, :]

        # Top edge connects nodes 3 and 4
        edge_length = np.sqrt((elx[2] - elx[3])**2 + (ely[2] - ely[3])**2)

        # Nodal force
        nodal_force = q0_num * edge_length / 2.0

        # Applying the force to f vector
        f[eltopo[5] - 1] += nodal_force  
        f[eltopo[7] - 1] += nodal_force  


    # Applying distributed load along the right edge
    right_elements = range(nelx - 1, nelx * nely, nelx)

    # Distributed load along right edge:
    q_right = P_num / h_num

    for el_idx in right_elements:
        elx = Ex[el_idx, :]
        ely = Ey[el_idx, :]
        eltopo = Edof[el_idx, :]

        # Right edge connects nodes 2 and 3
        edge_length = np.sqrt((elx[1] - elx[2])**2 + (ely[1] - ely[2])**2)

        nodal_force = q_right * edge_length / 2.0

        # Apply to right edge noddes
        f[eltopo[3] - 1] += nodal_force
        f[eltopo[5] - 1] += nodal_force  

    print(f"Total distributed load: {q0_num * L1:.2f} N")
    print(f"Point load: {P_num:.2f} N")
    print(f"Total load: {q0_num * L + P_num:.2f} N")

    # Applying boundary conditions 
    bc = B4
    bcval = np.zeros(np.size(bc))

    # Solving the system
    a, r = cfc.solveq(K, f, bc, bcval)

    # Extracting displacements
    print(f"\nMaximum displacement: {np.min(a):.6e} m")
    print(f"Maximum displacement (mm): {np.min(a)*1000:.4f} mm")

    Ed = cfc.extract_eldisp(Edof, a)

    # Deformations
    plt.figure()
    plotpar = [2, 1, 0] 
    cfv.eldraw2(Ex, Ey, plotpar)  # Drawing the original geometry
    plt.title('Original geometry')
    sfig('calfem_original_geometry_' + str(L) + '.png')

    # Compute nodal coordinates
    xv = np.linspace(p1[0], p2[0], nelx + 1)
    yv = np.linspace(p1[1], p2[1], nely + 1)
    coords = np.array([[x, y] for y in yv for x in xv])  # (N, 2) array

    # Deformed coordinates
    U = a[0::2].reshape(-1, 1)  # x-displacements
    V = a[1::2].reshape(-1, 1)  # y-displacements

    sfac = 1  # deformation scale
    coords_def = coords + sfac * np.hstack([U, V])

    plt.figure()
    for elx, ely in zip(Ex, Ey):
        # Undeformed
        plt.plot(np.append(elx, elx[0]), np.append(ely, ely[0]), 'k-', lw=0.8, alpha=0.5)

    for e, (elx, ely) in enumerate(zip(Ex, Ey)):
        # Get the deformed coordinates for each element
        nodes = ((Edof[e, ::ndof_per_node] - 1) // ndof_per_node).astype(int)
        x_def = coords_def[nodes, 0]
        y_def = coords_def[nodes, 1]
        plt.plot(np.append(x_def, x_def[0]), np.append(y_def, y_def[0]), 'r-', lw=1.0)

    plt.gca().set_aspect('equal')
    plt.title(f"Deformed mesh, L = {L} (scale factor = {sfac:.1f})")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    sfig('calfem_deformed_mesh_' + str(L) + '.png')
    plt.show()

    # Initialize arrays to store stresses
    num_elements = len(Ex)
    stress_xx_membrane = np.zeros(num_elements)
    stress_xx_bending = np.zeros(num_elements)
    stress_xx_total = np.zeros(num_elements)
    stress_yy = np.zeros(num_elements)
    stress_xy = np.zeros(num_elements)
    von_mises = np.zeros(num_elements)

    # z-coordinate (negative because it's below the mid-plane)
    z_eval = -h_num / 2

    # Element center x-coordinates
    elem_x_center = np.zeros(num_elements)

    # Loop through all elements
    for el_idx, (eltopo, elx, ely) in enumerate(zip(Edof, Ex, Ey)):
        ed = Ed[el_idx, :]
        
        # Compute stresses from plane stress FEM
        es, et = cfc.planqs(elx, ely, ep, Dmat, ed)
        
        stress_xx_membrane[el_idx] = es[0]
        stress_yy[el_idx] = es[1]
        stress_xy[el_idx] = es[2]
        
        # Element center x-coordinate
        elem_x_center[el_idx] = np.mean(elx)
        x_pos = elem_x_center[el_idx]
        
        # Bending moment: M(x) = -q0*(L1-x)²/2 - P*(L1-x)
        M_x = -q0_num * (L - x_pos)**2 / 2.0 - P_num * (L - x_pos)
        
        # Bending stress at z = -h/2
        stress_xx_bending[el_idx] = M_x * z_eval / I_num
        
        # Total normal stress
        stress_xx_total[el_idx] = stress_xx_membrane[el_idx] + stress_xx_bending[el_idx]
        
        # Von Mises stress
        von_mises[el_idx] = np.sqrt(
            stress_xx_total[el_idx]**2 - 
            stress_xx_total[el_idx] * stress_yy[el_idx] + 
            stress_yy[el_idx]**2 + 
            3 * stress_xy[el_idx]**2
        )

    # Sorting all arrays by x-coordinate for proper plotting
    sort_idx = np.argsort(elem_x_center)
    x_sorted = elem_x_center[sort_idx]
    stress_xx_membrane_sorted = stress_xx_membrane[sort_idx]
    stress_xx_bending_sorted = stress_xx_bending[sort_idx]
    stress_xx_total_sorted = stress_xx_total[sort_idx]
    stress_yy_sorted = stress_yy[sort_idx]
    stress_xy_sorted = stress_xy[sort_idx]
    von_mises_sorted = von_mises[sort_idx]

    x_unique = np.unique(x_sorted)
    n_x = len(x_unique)

    stress_xx_mem_avg = np.zeros(n_x)
    stress_xx_bend_avg = np.zeros(n_x)
    stress_xx_tot_avg = np.zeros(n_x)
    stress_yy_avg = np.zeros(n_x)
    stress_xy_avg = np.zeros(n_x)
    von_mises_avg = np.zeros(n_x)

    for i, x_val in enumerate(x_unique):
        mask = np.abs(x_sorted - x_val) < 1e-10
        stress_xx_mem_avg[i] = np.mean(stress_xx_membrane_sorted[mask])
        stress_xx_bend_avg[i] = np.mean(stress_xx_bending_sorted[mask])
        stress_xx_tot_avg[i] = np.mean(stress_xx_total_sorted[mask])
        stress_yy_avg[i] = np.mean(stress_yy_sorted[mask])
        stress_xy_avg[i] = np.mean(stress_xy_sorted[mask])
        von_mises_avg[i] = np.mean(von_mises_sorted[mask])

    # Normal stress
    plt.figure()
    plt.plot(x_unique, stress_xx_tot_avg/1e6, 'r-', linewidth=2.5, label='Total')
    plt.grid(True, alpha=0.3)
    plt.xlabel('x [m]')
    plt.ylabel('σ_xx [MPa]')
    plt.title(f'Total Normal Stress at z = {z_eval*1000:.1f} mm')
    sfig('calfem_normal_stress_' + str(L) + '.png')
    plt.legend()

    # Convert stress to MPa (same as in your plot)
    sigma_MPa = stress_xx_tot_avg / 1e6

    # Create DataFrame
    df = pd.DataFrame({
        'Position (m)': x_unique,
        'Normal Stress (MPa)': sigma_MPa
    })

    # Save to CSV
    # df.to_csv(fr'normal_stress_calfem_L=3m.csv', index=False)

    # print("Data saved to 'total_normal_stress.csv'")

    # Von mises
    plt.figure()
    plt.plot(x_unique, von_mises_avg/1e6, 'r-', linewidth=2.5, label='von Mises')
    # plt.axhline(y=sigma_yield/1e6, color='k', linestyle='--', linewidth=1.5, label='Yield strength')
    plt.grid(True, alpha=0.3)
    plt.xlabel('x [m]')
    plt.ylabel('σ_vm [MPa]')
    plt.title(f'Von Mises Stress at z = {z_eval*1000:.1f} mm')
    plt.legend()
    plt.tight_layout()
    sfig('calfem_von_mises_' + str(L) + '.png')
    plt.show()
    
calfem_analysis_A1(L1)
calfem_analysis_A1(L2)

# %%
####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################




# Assignment 1 - Read Abaqus data



####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################

# Read abaqus_results.rpt
datasets = {}
current_data = []
current_name = None

with open('abaqus_results.rpt', 'r') as f:
    for line in f:
        # Detect dataset header: line with at least two words, first is 'X'
        tokens = line.strip().split()
        if len(tokens) >= 2 and tokens[0] == 'X':
            # Save previous dataset
            if current_name and current_data:
                datasets[current_name] = np.array(current_data)
            # Start new dataset
            current_name = ' '.join(tokens)
            current_data = []
        elif line.strip() and not (line.strip().startswith('X') or line.strip() == ''):
            # Try to parse data lines
            try:
                values = [float(x.replace('E', 'e')) for x in line.split()]
                if len(values) == 2:
                    current_data.append(values)
            except Exception:
                pass  # skip lines that can't be parsed
    # Save last dataset
    if current_name and current_data:
        datasets[current_name] = np.array(current_data)

print("Available datasets from abaqus_results.rpt:")
for name in datasets.keys():
    print(f"  - '{name}'")
print()

# Read abaqus_results_mises.rpt
datasets_mises = {}
current_data = []
current_name = None

with open('abaqus_results_mises.rpt', 'r') as f:
    for line in f:
        # Detect dataset header: line with at least two words, first is 'X'
        tokens = line.strip().split()
        if len(tokens) >= 2 and tokens[0] == 'X':
            # Save previous dataset
            if current_name and current_data:
                datasets_mises[current_name] = np.array(current_data)
            # Start new dataset
            current_name = ' '.join(tokens)
            current_data = []
        elif line.strip() and not (line.strip().startswith('X') or line.strip() == ''):
            # Try to parse data lines
            try:
                values = [float(x.replace('E', 'e')) for x in line.split()]
                if len(values) == 2:
                    current_data.append(values)
            except Exception:
                pass  # skip lines that can't be parsed
    # Save last dataset
    if current_name and current_data:
        datasets_mises[current_name] = np.array(current_data)

print("Available datasets from abaqus_results_mises.rpt:")
for name in datasets_mises.keys():
    print(f"  - '{name}'")
print()

x_vals_abaqus_L1 = datasets['X s11_bottom_X_3m'][:, 0]
deflection_abaqus_L1 = datasets['X u2_middle_X_3m'][:, 1]
stress_abaqus_L1 = datasets['X s11_bottom_X_3m'][:, 1]
x_vals_abaqus_L2 = datasets['X s11_bottom_X_03m'][:, 0]
deflection_abaqus_L2 = datasets['X u2_middle_X_03m'][:, 1]
stress_abaqus_L2 = datasets['X s11_bottom_X_03m'][:, 1]

# Plot all datasets
for name, data in datasets.items():
    plt.figure()
    plt.plot(data[:, 0], data[:, 1], label=name)
    plt.xlabel(name.split()[0])
    plt.ylabel(name.split()[1] if len(name.split()) > 1 else '')
    plt.title(name)
    plt.legend()
    sfig(str(name))
    plt.show()


# Mesh convergence data
u2_3m = [-1.016e-4, -1.032e-4, -1.038e-4, -1.038e-4]
u2_03m = [-1.148e-1, -1.165e-1, -1.170e-1, -1.170e-1]
meshsize = [0.01, 0.005, 0.001, 0.0005]

name = 'Mesh convergence (3m)'
plt.figure()
plt.plot(meshsize, u2_3m,'X-')
plt.title(name)
plt.xlabel('meshsize (m)')
plt.ylabel('displacement $w$')
sfig(str(name))
plt.show()

name = 'Mesh convergence (03m)'
plt.figure()
plt.title(name)
plt.plot(meshsize, u2_03m,'X-')
plt.xlabel('meshsize (m)')
plt.ylabel('displacement $w$')
sfig(str(name))
plt.show()

#%%
####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################



# Assignment 1 - Comparison between models



####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################
new_prob('1 - Comparison between models')


plt.figure()

plt.plot(x_vals_bernoulli_L1, w_vals_bernoulli_L1, label='Euler-Bernoulli')
plt.plot(x_vals_timoshenko_L1, w_vals_timoshenko_L1, linestyle ='dashdot', label='Timoshenko')
# plt.plot(x_vals_timoshenko_L1, w_vals_timoshenko_L1, linestyle ='dashed', label='Timoshenko')
# plt.plot(x_vals_timoshenko_L1, w_vals_timoshenko_L1, linestyle ='dotted', label='Timoshenko')
# plt.plot(x_vals_calfem_L1, deflection_calfem_L1, label='Calfem')
plt.plot(x_vals_abaqus_L1, deflection_abaqus_L1, linestyle ='dotted', label='Abaqus')

plt.title('Deflection comparison (L=3m)')
plt.xlabel('x (m)')
plt.ylabel('w (mm)')
fig('comparison deflection 3m')


plt.figure()

plt.plot(x_vals_bernoulli_L2, w_vals_bernoulli_L2, label='Euler-Bernoulli')
plt.plot(x_vals_timoshenko_L2, w_vals_timoshenko_L2, linestyle ='dashdot', label='Timoshenko')
# plt.plot(x_vals_calfem_L2, deflection_calfem_L2, linestyle ='dashed', label='Calfem')
plt.plot(x_vals_abaqus_L2, deflection_abaqus_L2, linestyle ='dotted', label='Abaqus')

plt.title(f'Deflection comparison (L= 0.3m)')
plt.xlabel('x (m)')
plt.ylabel('w (mm)')
fig('comparison deflection 03m')


plt.figure()

plt.plot(x_vals_bernoulli_L1, sigma_xx_bernoulli_L1, label='Euler-Bernoulli')
plt.plot(x_vals_timoshenko_L1, sigma_xx_timoshenko_L1, linestyle ='dashdot', label='Timoshenko')
# plt.plot(x_vals_calfem_L1, stress_calfem_L1, linestyle ='dashed', label='Calfem')
plt.plot(x_vals_abaqus_L1, -stress_abaqus_L1, linestyle ='dotted', label='Abaqus')

plt.title('Normal stress comparison (L=3m)')
plt.xlabel('x (m)')
plt.ylabel('sigma (mm)')
fig('comparison stress 3m')


plt.figure()

plt.plot(x_vals_bernoulli_L2, sigma_xx_bernoulli_L2, label='Euler-Bernoulli')
plt.plot(x_vals_timoshenko_L2, sigma_xx_timoshenko_L2, linestyle ='dashdot', label='Timoshenko')
# plt.plot(x_vals_calfem_L2, stress_calfem_L2, linestyle ='dashed', label='Calfem')
plt.plot(x_vals_abaqus_L2, -stress_abaqus_L2, linestyle ='dotted', label='Abaqus')

plt.title('Normal stress comparison (L=0.3m)')
plt.xlabel('x (m)')
plt.ylabel('sigma (mm)')
fig('comparison stress 03m')

#%%
####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################



# Assignment 2 - numerical v2



####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################

p_num = 120 * 10**6

# Define symbols
p, q0, a, b, E, nu, h, r, f_r, A1, A2 = symbols('p q0 a b E nu h r f_r A1 A2', real = True)

D = E / (1 - nu**2)  # same as in problem 4 in the literature

# radial force = 0
f_r = 0
w = A1 * r/2 + A2 / r

w_prime = diff(w,r) # rotation field

eps_r = diff(w, r)
sigma_r = D * diff(w, r)

# Apply the boundary conditions
boundary_conditions = [
                        w.subs(r, a),               # inner boundary displacement 0
                        sigma_r.subs(r, b) - p,     # outer boundary stress
                       ]

# Solve for unknown constants
unknowns = (A1, A2)
sol= solve(boundary_conditions, unknowns)

# Formulate the deflection field
w_ = simplify(w.subs(sol)) # constants substituted

print("w(r) = ")
display(w_)

# Plot the deflection field for a given set of parameters
wp_f = simplify(w_.subs({f_r:0, p:p_num, q0:0, E:E2_num, nu:poisson_num, a:a_num, b:b_num, h:h0_num})) # parameters substituted

r_num  = np.linspace(150, 250, 401)
wr_num = [wp_f.subs({r:val}) for val in r_num]

print(r_num)
print(wr_num)

plt.figure()
plt.plot(r_num, wr_num, "b-")
plt.axvline(a_num, color='black', linestyle='--', label='a')
plt.axvline(b_num, color='grey', linestyle='--', label='b')
plt.title('Radial deflection')
plt.xlabel(r"$r$ [mm]")
plt.ylabel(r"$w$ [mm]")
plt.grid()
plt.legend()
plt.savefig('Radial deflection', dpi=dpi, bbox_inches='tight')
plt.show()

# Substitute the solution constants
sigma_rr_ = simplify(sigma_r.subs(sol))

print("sigma_rr(r) = ")
display(sigma_rr_)

# Plot the radial stress field for the same parameters
sigma_rr_f = simplify(sigma_rr_.subs({f_r: 0, p:p_num, q0:0, E:E2_num, nu:poisson_num, a:a_num, b:b_num, h:h0_num}))

sigma_rr_num = [sigma_rr_f.subs({r:val}) for val in r_num]

plt.figure()
plt.plot(r_num, sigma_rr_num, "r-")
plt.axvline(a_num, color='black', linestyle='--', label='a')
plt.axvline(b_num, color='grey', linestyle='--', label='b')
plt.axhline(0, color='gray', linestyle='-', linewidth=0.5)
plt.title('Normal stress')
plt.xlabel(r"$r$ [mm]")
plt.ylabel(r"$\sigma_{rr}$ [Pa]")
plt.grid()
plt.legend()
plt.savefig('Normal stress', dpi=dpi, bbox_inches='tight')
plt.show()
# %%
####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################




# Assigment 2 - Calfem, constant area



####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################
new_prob('2 - Calfem')

b_outer = 0.25
nu=0.3
E=200e9
h0 = 0.02
Width = b
num_el = 6
nnodes = num_el + 1
a_inner = 0.15

coords = np.linspace(b_outer, a_inner, nnodes)
#coords = coords.reshape(-1,1)

Edof = np.zeros((num_el, 2), dtype=int)
for i in range(num_el):
    Edof[i, 0] = i + 1 
    Edof[i, 1] = i + 2       

#display(coords)
display(Edof)

num_dofs = np.max(Edof)

#Dmat = E / (1-nu**2) * np.array([[1, nu],
                                 #[nu,1]])

num_dofs = np.max(Edof)
K = np.zeros((num_dofs, num_dofs))
f = np.zeros((num_dofs, 1))

print("K shape:", K.shape)

for el in range(num_el):
    r1 = coords[el]
    r2 = coords[el + 1]
    Le = r2 - r1
    r_mean = 0.5 * (r1 + r2)
    
    # Constant thickness
    h_e = h0

    print('h_e',np.shape(h_e))
    print('r_mean',np.shape(r_mean))

    
    Ke = (2 * np.pi * E * h_e * r_mean / Le) * np.array([[1, -1],
                                                        [-1,  1]])
    cfc.assem(Edof[el, :], K, Ke)

# Bcs
bc = np.array([nnodes])  
bcVal = np.array([0.0])

# Applying distributed load
sigma_r = -120e6  # 1 MPa
r_inner = coords[0]
h_inner = h0
f[0, 0] = 2 * np.pi * r_inner * h_inner * sigma_r  # negative radial direction


a, r= cfc.solveq(K, f, bc, bcVal)


plt.figure()
plt.plot(coords, a * 1e3, 'o-', label='$u_r(r)$ [mm]')
plt.xlabel('r [m]')
plt.ylabel('Radial displacement [mm]')
plt.title('Axisymmetric radial displacement')
plt.grid(True)
plt.legend()
plt.savefig('Axisymmetric radial displacement (constant height)', dpi=dpi, bbox_inches='tight')
plt.show()


for i in range(nnodes):
    print(f"Node {i+1}: r = {coords[i]:.4f} m, ur = {a[i,0]*1e6:.3f} μm")

# Computing radial stress
sigma_rr_vals = np.zeros(num_el)
r_centers = np.zeros(num_el)

for el in range(num_el):
    r1 = coords[el]
    r2 = coords[el + 1]
    Le = r2 - r1
    r_mean = 0.5 * (r1 + r2)
    r_centers[el] = r_mean

    u1 = a[el, 0]
    u2 = a[el + 1, 0]
    du_dr = (u2 - u1) / Le

    # Plane stress approximation
    sigma_rr = E / (1 - nu**2) * (du_dr + nu * (u1 + u2) / (2 * r_mean))
    sigma_rr_vals[el] = sigma_rr

# Plotting radial stress
plt.figure()
plt.plot(r_centers, sigma_rr_vals/1e6, 'o-', label=r'$\sigma_{rr}$ [MPa]')
plt.xlabel('r [m]')
plt.ylabel(r'$\sigma_{rr}$ [MPa]')
plt.title('Radial stress distribution σ_rr(r)')
plt.grid(True)
plt.legend()
plt.savefig('Radial stress distribution (constant height)', dpi=dpi, bbox_inches='tight')
plt.show()


# %%
####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################




# Assigment 2 - Calfem, varying area



####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################


b_outer = 0.25
nu=0.3
E=200e9
h0 = 0.02
Width = b
num_el = 6
nnodes = num_el + 1
a_inner = 0.15

coords = np.linspace(b_outer, a_inner, nnodes)

display(coords)
#coords = coords.reshape(-1,1)

Edof = np.zeros((num_el, 2), dtype=int)
for i in range(num_el):
    Edof[i, 0] = i + 1       # First column
    Edof[i, 1] = i + 2       # Second column

#display(coords)
display(Edof)

num_dofs = np.max(Edof)

#Dmat = E / (1-nu**2) * np.array([[1, nu],
                                 #[nu,1]])

num_dofs = np.max(Edof)
K = np.zeros((num_dofs, num_dofs))
f = np.zeros((num_dofs, 1))

print("K shape:", K.shape)

for el in range(num_el):
    r1 = coords[el]
    r2 = coords[el + 1]
    Le = r2 - r1
    r_mean = 0.5 * (r1 + r2)
    
    # variable thickness
    h_e = h0 * (r_mean - a_inner) / (b_outer - a_inner) + h0

    print('h_e',np.shape(h_e))
    print('r_mean',np.shape(r_mean))

    # local stiffness (axisymmetric linear element)
    Ke = (2 * np.pi * E * h_e * r_mean / Le) * np.array([[1, -1],
                                                        [-1,  1]])
    cfc.assem(Edof[el, :], K, Ke)

# --- Boundary conditions ---
bc = np.array([nnodes])  # fix outer radius (clamped)
bcVal = np.array([0.0])

# Applying distributed load
sigma_r = -120e6 
r_inner = coords[0]
h_inner = h0
f[0, 0] = 2 * np.pi * r_inner * h_inner * sigma_r  # negative radial direction


a, r= cfc.solveq(K, f, bc, bcVal)


plt.figure()
plt.plot(coords, a * 1e3, 'o-', label='$u_r(r)$ [mm]')
plt.xlabel('r [m]')
plt.ylabel('Radial displacement [mm]')
plt.title('Axisymmetric radial displacement')
plt.grid(True)
plt.legend()
plt.savefig('Axisymmetric radial displacement (varying height)', dpi=dpi, bbox_inches='tight')
plt.show()


for i in range(nnodes):
    print(f"Node {i+1}: r = {coords[i]:.4f} m, ur = {a[i,0]*1e6:.3f} μm")

# Computing normal stress, radial

sigma_rr_vals = np.zeros(num_el)
r_centers = np.zeros(num_el)

for el in range(num_el):
    r1 = coords[el]
    r2 = coords[el + 1]
    Le = r2 - r1
    r_mean = 0.5 * (r1 + r2)
    r_centers[el] = r_mean

    u1 = a[el, 0]
    u2 = a[el + 1, 0]
    du_dr = (u2 - u1) / Le

    # Plane stress approximation
    sigma_rr = E / (1 - nu**2) * (du_dr + nu * (u1 + u2) / (2 * r_mean))
    sigma_rr_vals[el] = sigma_rr

# Plotting radial stress
plt.figure()
plt.plot(r_centers, sigma_rr_vals/1e6, 'o-', label=r'$\sigma_{rr}$ [MPa]')
plt.xlabel('r [m]')
plt.ylabel(r'$\sigma_{rr}$ [MPa]')
plt.title('Radial stress distribution σ_rr(r)')
plt.grid(True)
plt.legend()
plt.savefig('Radial stress distribution (varying height)', dpi=dpi, bbox_inches='tight')
plt.show()
#%%