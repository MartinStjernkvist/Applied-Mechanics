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

diffeq1 = Eq(E*I * diff(w, x, 4), q0)

w = dsolve(diffeq1, w).rhs

M = -E*I*w.diff(x, 2)

#C1, C2, C3, C4 = symbols('C1 C2 C3 C4')

# Boundary conditions for distributed load
boundary_conditions1 = [ w.subs(x, 0),                          #w(0) = 0
                        w.diff(x).subs(x, 0),                   #w'(0) = 0
                        M.subs(x, L),                           #w''(L) = 0
                        w.diff(x,3).subs(x, L) - P/(-E*I)]      #w'''(L) = -P

integration_constants = solve(boundary_conditions1, 'C1, C2, C3, C4', real=True)

display(integration_constants)

solution = w.subs(integration_constants)

display(simplify(solution))

w_func = lambdify((x, L, q0, P, E, I), solution, 'numpy')

# Create moment function with constants substituted
M_solution = M.subs(integration_constants)
M_func = lambdify((x, L, q0, P, E, I), M_solution, 'numpy')

# Yield strength for steel
sigma_yield = 550e6 

# Numerical values
L1 = 3; L2 = 0.3

L=L1
E=220e9
b=h=0.05
I = b*h**3/12; #moment of inertia
rho = 7800; #density
g = 9.81
m = 130
P = -m*g
poisson = 0.3
q0 = -h*b*rho*g

x_vals = np.linspace(0, L, 200)
w_vals = w_func(x_vals, L, q0, P, E, I)

# Compute stresses at z = -h/2
z = -h/2
M_vals = M_func(x_vals, L, q0, P, E, I)
sigma_xx = -M_vals * z / I  # Normal stress from bending
sigma_vM = np.abs(sigma_xx)  # von Mises stress

# Find maximum stresses
max_sigma_xx = np.max(np.abs(sigma_xx))
max_sigma_vM = np.max(sigma_vM)
max_stress_location = x_vals[np.argmax(sigma_vM)]
safety_factor = sigma_yield / max_sigma_vM
will_yield = max_sigma_vM > sigma_yield

# --- Euler-Bernoulli, L1 ---

# Plot 1: Deflection
plt.figure()
plt.plot(x_vals, w_vals*1e3, 'b-')
plt.title(f'Beam Deflection at L={L}m', fontweight='bold')
plt.xlabel('Position along beam (m)')
plt.ylabel('Deflection (mm)')
plt.grid(True, alpha=0.3)
plt.axhline(0, color='black', linestyle='--')
plt.xlim(0, L)
plt.tight_layout()
plt.savefig('deflection_L1.png', dpi=300, bbox_inches='tight')
plt.show()

# Plot 2: Normal stress σ_xx
plt.figure()
plt.plot(x_vals, sigma_xx/1e6, 'r-')
plt.title('Normal Stress σ_xx at z=-h/2 (bottom surface)', fontweight='bold')
plt.xlabel('Position along beam (m)')
plt.ylabel('Normal Stress (MPa)')
plt.grid(True, alpha=0.3)
plt.axhline(0, color='black', linestyle='--')
plt.axhline(sigma_yield/1e6, color='orange', linestyle='--', label=f'Yield strength = {sigma_yield/1e6:.0f} MPa')
plt.axhline(-sigma_yield/1e6, color='orange', linestyle='--')
plt.xlim(0, L)
plt.legend()
plt.tight_layout()
plt.savefig('sigmaxx_L1.png', dpi=300, bbox_inches='tight')
plt.show()

# Plot 3: von Mises stress
plt.figure()
plt.plot(x_vals, sigma_vM/1e6, 'g-')
plt.title('von Mises Effective Stress σ_vM at z=-h/2', fontweight='bold')
plt.xlabel('Position along beam (m)')
plt.ylabel('von Mises Stress (MPa)')
plt.grid(True, alpha=0.3)
plt.axhline(sigma_yield/1e6, color='orange', linestyle='--', label=f'Yield strength = {sigma_yield/1e6:.0f} MPa')
plt.xlim(0, L)
plt.legend()
plt.tight_layout()
plt.savefig('vonmises_L1.png', dpi=300, bbox_inches='tight')
plt.show()

# Print results
print(f"\n{'='*70}")
print(f"STRESS ANALYSIS RESULTS FOR L={L}m")
print(f"{'='*70}")
print(f"Beam properties:")
print(f"  Length: {L} m")
print(f"  Cross-section: {b*1e3:.1f} mm × {h*1e3:.1f} mm")
print(f"  Material: Steel (E = {E/1e9:.0f} GPa, σ_yield = {sigma_yield/1e6:.0f} MPa)")
print(f"\nLoading:")
print(f"  Distributed load q₀: {q0:.2f} N/m")
print(f"  Point load P: {P:.2f} N (at free end)")
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


L=L2

x_vals = np.linspace(0, L, 200)
w_vals = w_func(x_vals, L, q0, P, E, I)

# Compute stresses at z = -h/2 (bottom surface)
z = -h/2
M_vals = M_func(x_vals, L, q0, P, E, I)
sigma_xx = -M_vals * z / I  # Normal stress from bending
sigma_vM = np.abs(sigma_xx)  # von Mises stress

# Find maximum stresses
max_sigma_xx = np.max(np.abs(sigma_xx))
max_sigma_vM = np.max(sigma_vM)
max_stress_location = x_vals[np.argmax(sigma_vM)]
safety_factor = sigma_yield / max_sigma_vM
will_yield = max_sigma_vM > sigma_yield

# --- Euler-Bernoulli, L2 ---

# Plot 1: Deflection
plt.figure()
plt.plot(x_vals, w_vals*1e3, 'b-')
plt.title(f'Beam Deflection at L={L}m', fontweight='bold')
plt.xlabel('Position along beam (m)')
plt.ylabel('Deflection (mm)')
plt.grid(True, alpha=0.3)
plt.axhline(0, color='black', linestyle='--')
plt.xlim(0, L)
plt.tight_layout()
plt.savefig('deflection_L2.png', dpi=300, bbox_inches='tight')
plt.show()

# Plot 2: Normal stress σ_xx
plt.figure()
plt.plot(x_vals, sigma_xx/1e6, 'r-')
plt.title('Normal Stress σ_xx at z=-h/2 (bottom surface)', fontweight='bold')
plt.xlabel('Position along beam (m)')
plt.ylabel('Normal Stress (MPa)')
plt.grid(True, alpha=0.3)
plt.axhline(0, color='black', linestyle='--')
plt.axhline(sigma_yield/1e6, color='orange', linestyle='--', label=f'Yield strength = {sigma_yield/1e6:.0f} MPa')
plt.axhline(-sigma_yield/1e6, color='orange', linestyle='--')
plt.xlim(0, L)
plt.legend()
plt.tight_layout()
plt.savefig('sigmaxx_L2.png', dpi=300, bbox_inches='tight')
plt.show()

# Plot 3: von Mises stress
plt.figure()
plt.plot(x_vals, sigma_vM/1e6, 'g-')
plt.title('von Mises Effective Stress σ_vM at z=-h/2', fontweight='bold')
plt.xlabel('Position along beam (m)')
plt.ylabel('von Mises Stress (MPa)')
plt.grid(True, alpha=0.3)
plt.axhline(sigma_yield/1e6, color='orange', linestyle='--', label=f'Yield strength = {sigma_yield/1e6:.0f} MPa')
plt.xlim(0, L)
plt.legend()
plt.tight_layout()
plt.savefig('vonmises_L2.png', dpi=300, bbox_inches='tight')
plt.show()

# Print results
print(f"\n{'='*70}")
print(f"STRESS ANALYSIS RESULTS FOR L={L}m")
print(f"{'='*70}")
print(f"Beam properties:")
print(f"  Length: {L} m")
print(f"  Cross-section: {b*1e3:.1f} mm × {h*1e3:.1f} mm")
print(f"  Material: Steel (E = {E/1e9:.0f} GPa, σ_yield = {sigma_yield/1e6:.0f} MPa)")
print(f"\nLoading:")
print(f"  Distributed load q₀: {q0:.2f} N/m")
print(f"  Point load P: {P:.2f} N (at free end)")
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

#############
# Point load

x, q0, E, I, Ks, G, A, L, P = symbols('x q0 E I Ks G A L P', real=True)

f_phi = Function('phi') # phi is a function of x

## Define the differential equation in terms of phi
diffeq_phi = Eq(E*I*f_phi(x).diff(x, 3), q0)

## Solve the differential equation for phi(x) (eq. 3.35 LN)
phi = dsolve(diffeq_phi, f_phi(x)).rhs

## Solve the differential equation for w(x) (eq. 3.36 LN)
w = Function('w') # w is a function of x
diffeq_w = Eq(w(x).diff(x), -E*I/(G*Ks*A)*phi.diff(x,2) + phi)
w        = dsolve(diffeq_w, w(x)).rhs

## Define boundary conditions
M = -E*I*phi.diff(x)
# Boundary conditions for distributed load
boundary_conditions = [ w.subs(x, 0),                           #w(0) = 0
                        w.diff(x).subs(x, 0),                   #w'(0) = 0
                        M.subs(x, L),                           #w''(L) = 0
                        w.diff(x,3).subs(x, L) - P/(-E*I)]      #w'''(L) = -P

## Solve for the integration constants
integration_constants = solve(boundary_conditions, 'C1, C2, C3, C4', real=True)

## Substitute the integration constants into the solution
solution1 = w.subs(integration_constants)

display(solution1)

w_func = lambdify((x, L, q0, P, E, I, Ks, A, G), solution1, 'numpy')
# Create moment function with constants substituted
M_solution = M.subs(integration_constants)
M_func = lambdify((x, L, q0, P, E, I, Ks, A, G), M_solution, 'numpy')

sigma_yield = 550e6  

# Numerical values
L1 = 3; L2 = 0.3

L=L1
E=220e9
b=h=0.05
I = b*h**3/12; #moment of inertia
rho = 7800; #density
g = 9.81
m = 130
P = -m*g
poisson = 0.3
q0 = -h*b*rho*g
Ks = 5/6
A = b*h
G = E/(2*(1+poisson)) 


x_vals = np.linspace(0, L, 200)
w_vals = w_func(x_vals, L, q0, P, E, I, A, G, Ks)

# Compute stresses at z = -h/2
z = -h/2
M_vals = M_func(x_vals, L, q0, P, E, I, A, G, Ks)
sigma_xx = -M_vals * z / I  # Normal stress from bending
sigma_vM = np.abs(sigma_xx)  # von Mises stress

# Find maximum stresses
max_sigma_xx = np.max(np.abs(sigma_xx))
max_sigma_vM = np.max(sigma_vM)
max_stress_location = x_vals[np.argmax(sigma_vM)]
safety_factor = sigma_yield / max_sigma_vM
will_yield = max_sigma_vM > sigma_yield


# --- Timoshenko, L1 ---

plt.figure()
plt.plot(x_vals, w_vals*1e3, 'b-')
plt.title(f'Beam Deflection at L={L}m (Timoshenko)', fontweight='bold')
plt.xlabel('Position along beam (m)')
plt.ylabel('Deflection (mm)')
plt.grid(True, alpha=0.3)
plt.axhline(0, color='black', linestyle='--')
plt.xlim(0, L)
plt.tight_layout()
plt.savefig('deflection_timo_L1.png', dpi=300, bbox_inches='tight')
plt.show()

plt.figure()
plt.plot(x_vals, sigma_xx/1e6, 'r-')
plt.title('Normal Stress σ_xx at z=-h/2 (bottom surface)', fontweight='bold')
plt.xlabel('Position along beam (m)')
plt.ylabel('Normal Stress (MPa)')
plt.grid(True, alpha=0.3)
plt.axhline(0, color='black', linestyle='--')
plt.axhline(sigma_yield/1e6, color='orange', linestyle='--', label=f'Yield strength = {sigma_yield/1e6:.0f} MPa')
plt.axhline(-sigma_yield/1e6, color='orange', linestyle='--')
plt.xlim(0, L)
plt.legend()
plt.tight_layout()
plt.savefig('sigmaxx_timo_L1.png', dpi=300, bbox_inches='tight')
plt.show()

plt.figure()
plt.plot(x_vals, sigma_vM/1e6, 'g-')
plt.title('von Mises Effective Stress σ_vM at z=-h/2', fontweight='bold')
plt.xlabel('Position along beam (m)')
plt.ylabel('von Mises Stress (MPa)')
plt.grid(True, alpha=0.3)
plt.axhline(sigma_yield/1e6, color='orange', linestyle='--', label=f'Yield strength = {sigma_yield/1e6:.0f} MPa')
plt.xlim(0, L)
plt.legend()
plt.tight_layout()
plt.savefig('vonmises_timo_L1.png', dpi=300, bbox_inches='tight')
plt.show()

# Print results
print(f"\n{'='*70}")
print(f"STRESS ANALYSIS RESULTS FOR L={L}m (Timoshenko Beam Theory)")
print(f"{'='*70}")
print(f"Beam properties:")
print(f"  Length: {L} m")
print(f"  Cross-section: {b*1e3:.1f} mm × {h*1e3:.1f} mm")
print(f"  Material: Steel (E = {E/1e9:.0f} GPa, G = {G/1e9:.1f} GPa)")
print(f"  Shear correction factor Ks: {Ks:.3f}")
print(f"  Yield strength: {sigma_yield/1e6:.0f} MPa")
print(f"\nLoading:")
print(f"  Distributed load q₀: {q0:.2f} N/m")
print(f"  Point load P: {P:.2f} N (at free end)")
print(f"\nStress at z = -h/2 (bottom surface, maximum tension):")
print(f"  Maximum |σ_xx|: {max_sigma_xx/1e6:.2f} MPa")
print(f"  Maximum σ_vM: {max_sigma_vM/1e6:.2f} MPa")
print(f"  Location of max stress: x = {max_stress_location:.3f} m")
print(f"\nYield assessment (von Mises criterion):")
print(f"  Safety factor: {safety_factor:.2f}")
if will_yield:
    print(f"  ⚠️  BEAM WILL YIELD - Maximum stress exceeds yield strength!")
else:
    print(f"  ✓  Beam is safe - No yielding expected")
print(f"{'='*70}\n")

L=L2

x_vals = np.linspace(0, L, 200)
w_vals = w_func(x_vals, L, q0, P, E, I, A, G, Ks)

# Compute stresses at z = -h/2
z = -h/2
M_vals = M_func(x_vals, L, q0, P, E, I, A, G, Ks)
sigma_xx = -M_vals * z / I  # Normal stress from bending
sigma_vM = np.abs(sigma_xx)  # von Mises stress

# Find maximum stresses
max_sigma_xx = np.max(np.abs(sigma_xx))
max_sigma_vM = np.max(sigma_vM)
max_stress_location = x_vals[np.argmax(sigma_vM)]
safety_factor = sigma_yield / max_sigma_vM
will_yield = max_sigma_vM > sigma_yield

# --- Timoshenko, L2 ---

# Plot 1: Deflection
plt.figure()
plt.plot(x_vals, w_vals*1e3, 'b-')
plt.title(f'Beam Deflection at L={L}m (Timoshenko)', fontweight='bold')
plt.xlabel('Position along beam (m)')
plt.ylabel('Deflection (mm)')
plt.grid(True, alpha=0.3)
plt.axhline(0, color='black', linestyle='--')
plt.xlim(0, L)
plt.tight_layout()
plt.savefig('deflection_timo_L2.png', dpi=300, bbox_inches='tight')
plt.show()

# Plot 2: Normal stress σ_xx
plt.figure()
plt.plot(x_vals, sigma_xx/1e6, 'r-')
plt.title('Normal Stress σ_xx at z=-h/2 (bottom surface)', fontweight='bold')
plt.xlabel('Position along beam (m)')
plt.ylabel('Normal Stress (MPa)')
plt.grid(True, alpha=0.3)
plt.axhline(0, color='black', linestyle='--')
plt.axhline(sigma_yield/1e6, color='orange', linestyle='--', label=f'Yield strength = {sigma_yield/1e6:.0f} MPa')
plt.axhline(-sigma_yield/1e6, color='orange', linestyle='--')
plt.xlim(0, L)
plt.legend()
plt.tight_layout()
plt.savefig('sigmaxx_timo_L2.png', dpi=300, bbox_inches='tight')
plt.show()

# Plot 3: von Mises stress
plt.figure()
plt.plot(x_vals, sigma_vM/1e6, 'g-')
plt.title('von Mises Effective Stress σ_vM at z=-h/2', fontweight='bold')
plt.xlabel('Position along beam (m)')
plt.ylabel('von Mises Stress (MPa)')
plt.grid(True, alpha=0.3)
plt.axhline(sigma_yield/1e6, color='orange', linestyle='--', label=f'Yield strength = {sigma_yield/1e6:.0f} MPa')
plt.xlim(0, L)
plt.legend()
plt.tight_layout()
plt.savefig('vonmises_timo_L2.png', dpi=300, bbox_inches='tight')
plt.show()

# Print results
print(f"\n{'='*70}")
print(f"STRESS ANALYSIS RESULTS FOR L={L}m (Timoshenko Beam Theory)")
print(f"{'='*70}")
print(f"Beam properties:")
print(f"  Length: {L} m")
print(f"  Cross-section: {b*1e3:.1f} mm × {h*1e3:.1f} mm")
print(f"  Material: Steel (E = {E/1e9:.0f} GPa, G = {G/1e9:.1f} GPa)")
print(f"  Shear correction factor Ks: {Ks:.3f}")
print(f"  Yield strength: {sigma_yield/1e6:.0f} MPa")
print(f"\nLoading:")
print(f"  Distributed load q₀: {q0:.2f} N/m")
print(f"  Point load P: {P:.2f} N (at free end)")
print(f"\nStress at z = -h/2 (bottom surface, maximum tension):")
print(f"  Maximum |σ_xx|: {max_sigma_xx/1e6:.2f} MPa")
print(f"  Maximum σ_vM: {max_sigma_vM/1e6:.2f} MPa")
print(f"  Location of max stress: x = {max_stress_location:.3f} m")
print(f"\nYield assessment (von Mises criterion):")
print(f"  Safety factor: {safety_factor:.2f}")
if will_yield:
    print(f"  ⚠️  BEAM WILL YIELD - Maximum stress exceeds yield strength!")
else:
    print(f"  ✓  Beam is safe - No yielding expected")
print(f"{'='*70}\n")


#%%
####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################



# Assignment 1 - calfem

# See the other file


####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################

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
        
print(datasets.items())

# Plot all datasets
for name, data in datasets.items():
    plt.figure()
    plt.plot(data[:, 0], data[:, 1], label=name)
    plt.xlabel(name.split()[0])
    plt.ylabel(name.split()[1] if len(name.split()) > 1 else '')
    plt.title(name)
    plt.legend()
    plt.savefig(str(name), dpi=400, bbox_inches='tight')
    plt.show()

    
u2_3m = [-1.016e-4, -1.032e-4, -1.038e-4, -1.038e-4]
u2_03m = [-1.148e-1, -1.165e-1, -1.170e-1, -1.170e-1]
meshsize = [0.01, 0.005, 0.001, 0.0005]

name = 'Mesh convergence (3m)'
plt.figure()
plt.plot(meshsize, u2_3m,'X-')
plt.title(name)
plt.xlabel('meshsize (m)')
plt.ylabel('displacement $w$')
plt.savefig(str(name), dpi=400, bbox_inches='tight')
plt.show()

name = 'Mesh convergence (03m)'
plt.figure()
plt.title(name)
plt.plot(meshsize, u2_03m,'X-')
plt.xlabel('meshsize (m)')
plt.ylabel('displacement $w$')
plt.savefig(str(name), dpi=400, bbox_inches='tight')
plt.show()


datasets = {}
current_data = []
current_name = None

with open('abaqus_results_mises.rpt', 'r') as f:
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
        
print(datasets.items())

# Plot all datasets
for name, data in datasets.items():
    if name == 'X 3m':
        plt.figure()
        plt.plot(data[:, 0], data[:, 1], label= 'smises_bottom_X_03m')
        plt.xlabel(name.split()[0])
        plt.ylabel('smises_bottom_X_03m')
        plt.title('smises_bottom_X_03m')
        plt.legend()
        plt.savefig(str('smises_bottom_X_03m'), dpi=400, bbox_inches='tight')
        plt.show()
    if name == 'X m':
        plt.figure()
        plt.plot(data[:, 0], data[:, 1], label= 'smises_bottom_X_3m')
        plt.xlabel(name.split()[0])
        plt.ylabel('smises_bottom_X_3m')
        plt.title('smises_bottom_X_3m')
        plt.legend()
        plt.savefig(str('smises_bottom_X_3m'), dpi=400, bbox_inches='tight')
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


# np.savez('bernoulli_3.npz', x_values=, y_values=, z_values=)
# np.savez('timoshenko_3.npz', x_values=, y_values=, z_values=)
# np.savez('calfem_3.npz', x_values=, y_values=, z_values=)
# np.savez('abaqus_3.npz', x_values=, y_values=, z_values=)

# np.savez('bernoulli_03.npz', x_values=, y_values=, z_values=)
# np.savez('timoshenko_03.npz', x_values=, y_values=, z_values=)
# np.savez('calfem_03.npz', x_values=, y_values=, z_values=)
# np.savez('abaqus_03.npz', x_values=, y_values=, z_values=)

# bernoulli_3 = np.load('bernoulli_3.npz')
# timoshenko_3 = np.load('timoshenko_3.npz')
# calfem_3 = np.load('calfem_3.npz')
# abaqus_3 = np.load('abaqus_3.npz')

# x_bernoulli_3, deflection_bernoulli_3, stress_bernoulli_3 = bernoulli_3['x_values'], bernoulli_3['y_values'], bernoulli_3['z_values']
# x_timoshenko_3, deflection_timoshenko_3, stress_timoshenko_3 = timoshenko_3['x_values'], timoshenko_3['y_values'], timoshenko_3['z_values']
# x_calfem_3, deflection_calfem_3, stress_calfem_3 = calfem_3['x_values'], calfem_3['y_values'], calfem_3['z_values']
# x_abaqus_3, deflection_abaqus_3, stress_abaqus_3 = abaqus_3['x_values'], abaqus_3['y_values'], abaqus_3['z_values']

# bernoulli_03 = np.load('bernoulli_03.npz')
# timoshenko_03 = np.load('timoshenko_03.npz')
# calfem_03 = np.load('calfem_03.npz')
# abaqus_03 = np.load('abaqus_03.npz')

# x_bernoulli_03, deflection_bernoulli_03, stress_bernoulli_03 = bernoulli_03['x_values'], bernoulli_03['y_values'], bernoulli_03['z_values']
# x_timoshenko_03, deflection_timoshenko_03, stress_timoshenko_03 = timoshenko_03['x_values'], timoshenko_03['y_values'], timoshenko_03['z_values']
# x_calfem_03, deflection_calfem_03, stress_calfem_03 = calfem_03['x_values'], calfem_03['y_values'], calfem_03['z_values']
# x_abaqus_03, deflection_abaqus_03, stress_abaqus_03 = abaqus_03['x_values'], abaqus_03['y_values'], abaqus_03['z_values']


# plt.figure()

# plt.plot(x_bernoulli_3, deflection_bernoulli_3, label='Euler-Bernoulli')
# plt.plot(x_timoshenko_3, deflection_timoshenko_3, label='')
# plt.plot(x_calfem_3, deflection_calfem_3, label='Calfem')
# plt.plot(x_abaqus_3, deflection_abaqus_3, label='Abaqus')

# plt.title('Deflection comparison (L=3m)')
# plt.xlabel('x (m)')
# plt.ylabel('w (mm)')
# plt.grid(True)
# plt.legend()
# plt.savefig('comparison deflection 3m', dpi=dpi, bbox_inches='tight')
# plt.show()



# plt.figure()

# plt.plot(x_bernoulli_03, deflection_bernoulli_03, label='Euler-Bernoulli')
# plt.plot(x_timoshenko_03, deflection_timoshenko_03, label='')
# plt.plot(x_calfem_03, deflection_calfem_03, label='Calfem')
# plt.plot(x_abaqus_03, deflection_abaqus_03, label='Abaqus')

# plt.title('Deflection comparison (L=3m)')
# plt.xlabel('x (m)')
# plt.ylabel('w (mm)')
# plt.grid(True)
# plt.legend()
# plt.savefig('comparison deflection 03m', dpi=dpi, bbox_inches='tight')
# plt.show()


# plt.figure()

# plt.plot(x_bernoulli_3, stress_bernoulli_3, label='Euler-Bernoulli')
# plt.plot(x_timoshenko_3, stress_timoshenko_3, label='')
# plt.plot(x_calfem_3, stress_calfem_3, label='Calfem')
# plt.plot(x_abaqus_3, stress_abaqus_3, label='Abaqus')

# plt.title('Stress comparison (L=3m)')
# plt.xlabel('x (m)')
# plt.ylabel('w (mm)')
# plt.grid(True)
# plt.legend()
# plt.savefig('comparison stress 3m', dpi=dpi, bbox_inches='tight')
# plt.show()


# plt.figure()

# plt.plot(x_bernoulli_03, stress_bernoulli_03, label='Euler-Bernoulli')
# plt.plot(x_timoshenko_03, stress_timoshenko_03, label='')
# plt.plot(x_calfem_03, stress_calfem_03, label='Calfem')
# plt.plot(x_abaqus_03, stress_abaqus_03, label='Abaqus')

# plt.title('Stress comparison (L=0.3m)')
# plt.xlabel('x (m)')
# plt.ylabel('w (mm)')
# plt.grid(True)
# plt.legend()
# plt.savefig('comparison stress 03m', dpi=dpi, bbox_inches='tight')
# plt.show()




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
# See other file




####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################



# %%
####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################




# Assigment 2 - Calfem, varying area
# See other file



####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################
#%%