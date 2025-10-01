"""
    Problem 13: Determine the deflection ur of a rotating axisymmetric disk

    Governing differential equation: d/dr(1/r * d/dr(r * ur)) = -(1-nu^2)/E * fr
    where fr = rho * omega^2 * r
    Unknown: ur - deflection the axisymmetric disk
    BCs: sigma_rr(r=a)=0, sigma_rr(r=b)=0
"""

from sympy import *

## Define symbolic variables
r, nu, E, omega, rho, a, b = symbols('r nu E omega rho a b', real=True)
fr = rho * omega**2 * r # body force

f_ur = Function('ur') # ur is a function of r

## Define the differential equation in terms of the radial displacement ur
diffeq = Eq((1/r*(f_ur(r)*r).diff(r)).diff(r), -(1-nu**2) / E * fr)

## Solve the differential equation for ur(r)
ur = dsolve(diffeq, f_ur(r)).rhs

simplify(ur)

## Apply boundary conditions
epsilon_rr     = ur.diff(r)
epsilon_phiphi = ur / r
sigma_rr       = E/(1-nu**2) * (epsilon_rr + nu*epsilon_phiphi)

boundary_conditions = [ sigma_rr.subs(r, a), # sigma_rr(r=a)=0
                        sigma_rr.subs(r, b)] # sigma_rr(r=b)=0

## Solve for the integration constants
integration_constants = solve(boundary_conditions, 'C1, C2', real=True)

## Substitute the integration constants into the solution
solution = ur.subs(integration_constants)

simplify(solution)

