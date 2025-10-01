"""
    Problem 12: Solve the deflection of a beam using Timoshenko beam theory

    Governing differential equation: E*I*d^3(phi)/dx^3 = q(x)
                                    dw/dx = -E*I/(G*Ks*A)*d^2(phi)/dx^2 + phi
    Unknown: w(x) - deflection of the beam
    BCs: w(0)=0, M(0)=0, w(L)=0, M(L)=0
"""

from sympy import *

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
boundary_conditions = [ w.subs(x, 0), # w(0)=0
                        M.subs(x, 0), # M(0)=0
                        w.subs(x, L), # w(L)=0
                        M.subs(x, L)  # M(L)=0
                        ]

## Solve for the integration constants
integration_constants = solve(boundary_conditions, 'C1, C2, C3, C4', real=True)

## Substitute the integration constants into the solution
solution = w.subs(integration_constants)

solution
