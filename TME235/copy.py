#%%
%matplotlib widget
from funcs_n_imports import *
from quadmesh import *

#%%
#####################################################################################################
# GIVEN INFORMATION
####################################################################################################

# ASSIGNMENT 1 - Beam
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

Ks_num = 5/6
A_num = b_num * h_num
G_num = E_num / (2 * (1 + poisson_num)) 

# ASSIGNMENT 2 - Axissymetry



#%% 
####################################################################################################
# EULER-BERNOULLI
####################################################################################################
new_prob('1 - EULER-BERNOULLI')

x, L, q0, P, E, I = symbols('x L q0 P E I')

w = Function('w')(x) # w is a function of x

diffeq1 = Eq(E*I * diff(w, x, 4), q0)

w = dsolve(diffeq1, w).rhs

M = -E*I*w.diff(x, 2)

#C1, C2, C3, C4 = symbols('C1 C2 C3 C4')

# Boundary conditions for distributed load
boundary_conditions = [ 
                        w.subs(x, 0),                      #w(0) = 0
                        w.diff(x).subs(x, 0),                 #w'(0) = 0
                        M.subs(x, L),                       #w''(L) = 0
                        w.diff(x,3).subs(x, L) - P/(-E*I)      #w'''(L) = -P
                        ]
print('\nboundary conditions:')
display(boundary_conditions)

integration_constants = solve(boundary_conditions, 'C1, C2, C3, C4', real=True)
print('\nintegration constants:')
display(integration_constants)

solution = w.subs(integration_constants)
display(simplify(solution))

w_func = lambdify((x, L, q0, P, E, I), solution, 'numpy')

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
w_vals = w_func(x_vals, L, q0, P_num, E_num, I_num)

plt.figure()
plt.plot(x_vals, w_vals*1e3, 'b-', linewidth=2)
plt.title('Beam Deflection under Combined Loading at L=3', fontsize=16)
plt.xlabel('Position along beam (m)', fontsize=14)
plt.ylabel('Deflection (mm)', fontsize=14)
plt.grid(True)
plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
plt.ylim(bottom=min(w_vals)*1e3*1.1)
plt.xlim(0, L)
plt.show()

L=L2


x_vals = np.linspace(0, L, 200)
w_vals = w_func(x_vals, L, q0, P, E, I)

plt.figure()
plt.plot(x_vals, w_vals*1e3, 'b-', linewidth=2)
plt.title('Beam Deflection under Combined Loading at L=0.3', fontsize=16)
plt.xlabel('Position along beam (m)', fontsize=14)
plt.ylabel('Deflection (mm)', fontsize=14)
plt.grid(True)
plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
plt.ylim(bottom=min(w_vals)*1e3*1.1)
plt.xlim(0, L)
plt.show()

# %%
##################################################
# TIMOSHENKO
##################################################
new_prob(2)

#############
# Point load

x, q0, E, I, Ks, G, A, L, P = symbols('x q0 E I Ks G A L P', real=True)

f_phi = Function('phi') # phi is a function of x

## Define the differential equation in terms of phi
diffeq_phi = Eq(E*I*f_phi(x).diff(x, 3), 0)

## Solve the differential equation for phi(x) (eq. 3.35 LN)
phi = dsolve(diffeq_phi, f_phi(x)).rhs

## Solve the differential equation for w(x) (eq. 3.36 LN)
w = Function('w') # w is a function of x
diffeq_w = Eq(w(x).diff(x), -E*I/(G*Ks*A)*phi.diff(x,2) + phi)
w        = dsolve(diffeq_w, w(x)).rhs

## Define boundary conditions
M = -E*I*phi.diff(x)
bc_eqs = [
    Eq(w.subs(x, 0), 0),                # w(0) = 0
    Eq(diff(w, x).subs(x, 0), 0),       # w'(0) = 0
    Eq(diff(w, x, 2).subs(x, L), 0),    # w''(L) = 0
    Eq(diff(w, x, 3).subs(x, L), -P/(E*I))  # w'''(L) = -P/(EI)
]

## Solve for the integration constants
integration_constants = solve(bc_eqs, 'C1, C2, C3, C4', real=True)

## Substitute the integration constants into the solution
solution1 = w.subs(integration_constants)
display(solution1)

##################
# Distributed load

## Define symbolic variables
x, q0, E, I, Ks, G, A, L = symbols('x q0 E I Ks G A L', real=True)

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
boundary_conditions1 = [ w.subs(x, 0), 0,               #w(0) = 0
                        w.diff(x).subs(x, 0),           #w'(0) = 0
                        M.subs(x, L), 0,                #w''(L) = 0
                        w.diff(x,3).subs(x, L), 0]      #w'''(L) = 0

## Solve for the integration constants
integration_constants = solve(boundary_conditions1, 'C1, C2, C3, C4', real=True)

## Substitute the integration constants into the solution
solution2 = w.subs(integration_constants)
display(solution2)

solution_total = solution1 + solution2
display(simplify(solution_total))

w_func = lambdify((x, L, q0, P, E, I, Ks, A, G), solution_total, 'numpy')

## Plugging in values for the length of the beam and plotting

L = L1
q0 = -((m_num * g_num)/L)    

x_vals = np.linspace(0, L, 200)
w_vals = w_func(x_vals, L, q0, P_num, E_num, I_num, A_num, G_num, Ks_num)

plt.figure()
plt.plot(x_vals, w_vals*1e3, 'b-', linewidth=2)
plt.title('Beam Deflection, combined loading (L=3)', fontsize=16)
plt.xlabel('Position along beam (m)', fontsize=14)
plt.ylabel('Deflection (mm)', fontsize=14)
plt.grid(True)
plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
plt.ylim(bottom=min(w_vals) * 1e3 * 1.1)
plt.xlim(0, L)
plt.show()
plt.savefig('TIMOSHENKO_1', dpi=dpi, bbox_inches='tight')

L=L2
q0 = -((m_num * g_num)/L2)

x_vals = np.linspace(0, L, 200)
w_vals = w_func(x_vals, L, q0, P_num, E_num, I_num, A_num, G_num, Ks_num)

plt.figure()
plt.plot(x_vals, w_vals*1e3, 'b-', linewidth=2)
plt.title('Beam deflection, combined loading (L=0.3)', fontsize=16)
plt.xlabel('Position along beam (m)', fontsize=14)
plt.ylabel('Deflection (mm)', fontsize=14)
plt.grid(True)
plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
plt.ylim(bottom=min(w_vals) * 1e3 * 1.1)
plt.xlim(0, L)
plt.show()
plt.savefig('TIMOSHENKO_2', dpi=dpi, bbox_inches='tight')

# %%
####################################################################################################
# AXISYMMETRY NUMERICAL
####################################################################################################
new_prob('1 - AXISYMMETRY NUMERICAL')

# Define symbols
F, q0, a, b, E, nu, h, r, A1, A2, A3, A4 = symbols('F q0 a b E nu h r A1 A2 A3 A4', real = True)

D = 1 / 12 / (1 - nu**2) * E * (h**3) # bending stiffness

q = q0*(r-a)/(b-a)  # distributed load

# Formulate general solutions
w = integrate(1 / r * integrate(r * integrate( 1 / r * integrate(q * r / D, r), r), r), r)+\
    A1 * r**2 * log(r / b) + A2 * r**2 + A3 * log(r / b) + A4 # deflection field

w_prime = diff(w,r) # rotation field

M_r   = D*(-diff(w_prime, r) - nu / r * w_prime )   # radial bending moment field
M_phi = D*(-1 / r * w_prime - nu*diff(w_prime, r))  # circumferential bending moment field
V    = diff(M_r, r) + 1 / r * (M_r - M_phi)         # shear force field

# Apply the boundary conditions
boundary_conditions = [
                        M_r.subs(r, a),         # inner boundary radial bending moment free
                        V.subs(r, a) - F,       # inner boundary shear force applied
                        w.subs(r, b),           # outer boundary deflection fixed
                        w_prime.subs(r, b)      # outer boundary rotation fixed
                       ]

# Solve for unknown constants
unknowns = (A1, A2, A3, A4)
sol= solve(boundary_conditions, unknowns)

# Formulate the deflection field
w_ = simplify(w.subs(sol)) # constants substituted

print("w(r) = ", w_)

# Plot the deflection field for a given set of parameters
wp_f = simplify(w_.subs({F:1., q0:0., E:200e3, nu:0.3, a:100, b:500, h:4})) # parameters substituted

r_num  = np.linspace(100., 500., 401)
wr_num = [wp_f.subs({r:val}) for val in r_num]

plt.figure()
plt.plot(r_num, wr_num, "b-")
plt.title('Deflection')
plt.xlabel(r"$r$ [mm]")
plt.ylabel(r"$w$ [mm]")
plt.grid()
plt.show()
plt.savefig('AXISYMMETRY_1', dpi=dpi, bbox_inches='tight')

#%%