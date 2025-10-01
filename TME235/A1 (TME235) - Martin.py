#%%
%matplotlib widget
# %matplotlib inline
from scipy.optimize import fsolve
# from scipy.differentiate import hessian
import numpy as np
from numpy import einsum
import matplotlib.pyplot as plt
import sympy as sp
from IPython.display import display, Math
from mpl_toolkits.mplot3d import axes3d
from mpl_toolkits.mplot3d import Axes3D
from numpy.random import rand
from IPython.display import HTML
from matplotlib import animation

import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker


import numpy as np
from sympy import *
import matplotlib.pyplot as plt
from matplotlib import rcParams # for changing default values
import matplotlib.ticker as ticker

##################################################
# Functions
##################################################

def new_prob(num):
    print_string = '\n----------------------\n' + 'Assignment E' + str(num) + '\n----------------------\n'
    return print(print_string)

SMALL_SIZE = 10
MEDIUM_SIZE = 12
BIGGER_SIZE = 14

# Set the global font sizes
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.rc('figure', figsize=(8, 5))



##################################################
# Given (quadmesh.py)
##################################################

# Purpose:
# Generates a 2D rectangular mesh
#
# Input:
# p1    - Lower left point of rectangle [x1, y1]
# p2    - Upper right point of rectangle [x2, y2]
# nelx  - Number of elments in x direction
# nely  - Number of elments in y direction
# ndofs - Number of degrees of freedom per node
#
# Output:
# Edof  - Connectivity matrix for mesh, cf. Calfem Toolbox
# Ex    - Elementwise x-coordinates, cf. Calfem Toolbox
# Ey    - Elementwise y-coordinates, cf. Calfem Toolbox
# Bi    - Matrix containing boundary dofs for segment i (i=1,2,3,4)
#         First column -> 1st dofs, second column -> 2nd dofs and so on   
#         size = (num boundary nodes on segment) x ndofs
#         B1 = Bottom side     B2 = Right side
#         B3 = Upper side      B4 = Left side  
#
import numpy as np
def ex_ey_quadmesh(p1,p2,nelx,nely,ndofs):
    xv=np.linspace(p1[0],p2[0],nelx+1)
    yv=np.linspace(p1[1],p2[1],nely+1)

    nel  = nelx*nely;
    Ex=np.zeros((nel,4))
    Ey=np.zeros((nel,4))
    for m in range(0,nely):
        for n in range(0,nelx):
            Ex[n+m*nelx,0]=xv[n]
            Ex[n+m*nelx,1]=xv[n+1]
            Ex[n+m*nelx,2]=xv[n+1]
            Ex[n+m*nelx,3]=xv[n]
        #
            Ey[n+m*nelx,0]=yv[m]
            Ey[n+m*nelx,1]=yv[m]
            Ey[n+m*nelx,2]=yv[m+1]
            Ey[n+m*nelx,3]=yv[m+1]
            
    return Ex, Ey

def edof_quadmesh(nelx,nely,ndofs):
    Edof=np.zeros((nelx*nely,4*ndofs),'i')
    for m in range(0,nely):
        for n in range(0,nelx):
            Edof[n+m*nelx,0]=n*ndofs+1+m*(nelx+1)*ndofs
            Edof[n+m*nelx,1]=n*ndofs+2+m*(nelx+1)*ndofs
            Edof[n+m*nelx,2]=(n+1)*ndofs+1+m*(nelx+1)*ndofs
            Edof[n+m*nelx,3]=(n+1)*ndofs+2+m*(nelx+1)*ndofs
        #
            Edof[n+m*nelx,4]=(n+1)*ndofs+1+(m+1)*(nelx+1)*ndofs
            Edof[n+m*nelx,5]=(n+1)*ndofs+2+(m+1)*(nelx+1)*ndofs
            Edof[n+m*nelx,6]=n*ndofs+1+(m+1)*(nelx+1)*ndofs      
            Edof[n+m*nelx,7]=n*ndofs+2+(m+1)*(nelx+1)*ndofs
    return Edof

def B1B2B3B4_quadmesh(nelx,nely,ndofs):
    #lower boundary, dofs
    B1=np.linspace(1,(nelx+1)*ndofs,(nelx+1)*ndofs)
    B1=B1.astype(int)
    B2=np.zeros(((nely+1)*ndofs),'i')
    nn=0
    for n in range(0,nely+1):
        B2[nn]=(nelx+1)*ndofs*(n+1)-1
        if ndofs>1:
            B2[nn+1]=(nelx+1)*ndofs*(n+1)+0
        nn=nn+ndofs

    B3=np.linspace(1,(nelx+1)*ndofs,(nelx+1)*ndofs)+(nelx+1)*ndofs*nely
    B3=B3.astype(int)

    B4=np.zeros(((nely+1)*ndofs),'i')
    nn=0
    for n in range(0,nely+1):
        B4[nn]=(nelx+1)*ndofs*n+1
        if ndofs>1:
            B4[nn+1]=(nelx+1)*ndofs*n+2
        nn=nn+ndofs
   
    P1=np.zeros((2),'i'); P2=np.zeros((2),'i'); P3=np.zeros((2),'i'); P4=np.zeros((2),'i')
    for m in range(0,2):
        P1[m]=B1[m]
        P2[m]=B2[m]
        P4[m]=B3[m]
    P3[0]=B3[-1]-1
    P3[1]=B3[-1]
    return B1,B2,B3,B4,P1,P2,P3,P4

def quadmesh(p1,p2,nelx,nely,ndofs):
    Ex, Ey=ex_ey_quadmesh(p1,p2,nelx,nely,ndofs)
    Edof=edof_quadmesh(nelx,nely,ndofs)
    B1,B2,B3,B4,P1,P2,P3,P4=B1B2B3B4_quadmesh(nelx,nely,ndofs)
    return Ex,Ey,Edof,B1,B2,B3,B4,P1,P2,P3,P4

##################################################
# E1
##################################################
new_prob(1)

#%% ###################### Method 1: dsolve ##############################
# E1
##################################################
new_prob(1)
print("\nCantilever Beam - Point Load at Free End")

## Define symbolic variables
P, x, L, E, I = symbols('P x L E I', real=True)
w = Function('w')(x) # w is a function of x

## Define the load (0 for point load at end)
q = 0

## Define the differential equation
diffeq = Eq(E * I * w.diff(x, 4), q)
print('\ndifferential equation:')
display(diffeq)

## Solve the differential equation for w(x)
w_general = dsolve(diffeq, w).rhs
print('\ngeneral equation for w:')
display(w_general)

## bending moment
M = -E * I * w_general.diff(x, 2)
print('\nmomentum equation:')
display(M)

## shear 
V = - E * I * w_general.diff(x,3)
print('\nshear equation:')
display(V)

## Define boundary conditions, rewrite so that element = 0 
boundary_conditions = [ w_general.subs(x, 0),               # w(0) = 0
                        w_general.diff(x).subs(x, 0),       # w'(0) = 0
                        V.subs(x, L) + P,                   # V(L) = P
                        M.subs(x, L)]                       # M(L) = 0
print('\nboundary conditions:')
display(boundary_conditions)

## solve for the integration constants
integration_constants = solve(boundary_conditions, 'C1, C2, C3, C4', real=True)  # C1-C4 are generated by the `dsolve` call
print('\nintegration constants:')
display(integration_constants)

## apply the constants to get the final deflection expression
solution = simplify(w_general.subs(integration_constants)) 

print(f"\nDeflection: w(x) =")
display(solution)
print(f"\nMax deflection at x=L: w_max = {simplify(solution.subs(x, L))}")

## moment and shear with solution
M_solution = simplify(M.subs(integration_constants))
V_solution = simplify(V.subs(integration_constants))

print(f"\nMoment: M(x) =")
display(M_solution)

print(f"\nShear: V(x) =")
display(V_solution)



# %%
##################################################
# E2
##################################################
new_prob(2)



# %%