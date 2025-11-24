# MTF073 Computational Fluid Dynamics
# Task 2: 2D convection-diffusion
# Håkan Nilsson, 2025
# Department of Mechanics and Maritime Sciences
# Division of Fluid Dynamics
# Note that this is not efficient code. It is for educational purposes!

# This file gives the overall algorithm of the code. Make sure to read and
# understand it!
#
# The code requires that:
# * the file T2_codeFunctions_template.py is in the same path as this file
# * the folder named 'data' is in the same path as this file
#
# Note that:
# * Except for the "Inputs" below and the arguments to the
#   createAdditionalPlots function, you should not change anything here,
#   but only do changes to T2_codeFunctions_template.py (cF)
# * Each call to a cF function should only change the arrays in the
#   first row of the argument list. This, together with the name of the function
#   makes it easier to read and understand the code (and avoid coding mistakes).
# * You can easily go to the called function by putting the marker on it and
#   pressing Ctrl-g

# Clear all variables when running entire code:
from IPython import get_ipython
get_ipython().run_line_magic('reset', '-sf')
# Packages needed
import numpy as np
import copy
import matplotlib.pyplot as plt
# Close all plots when running entire code:
plt.close('all')
# All functions of the code (some to be implemented by you):
import T2_codeFunctions_template as cF

#===================== Inputs =====================

# Case number (same as case in description, 1-25)
caseID = 1

# Geometric and mesh inputs (mesh is read from file)
# L, H, mI, mJ, nI, nJ are set later, from the imported mesh
grid_type = 'coarse'  # Either 'coarse' or 'fine'

# Physical properties
rho     = 1      # Density
k       = 1      # Thermal conductivity 
Cp      = 500    # Specific heat
gamma = k/Cp   # Calculated diffusion coefficient

unsteady = False # True or False

if unsteady:
    # For unsteady:
    # ADD CODE HERE: PLAY WITH THESE ONCE YOUR CODE IS WORKING
    deltaT = 1   # ADD CODE HERE
    endTime = 50 # ADD CODE HERE
    # Note that a frame is saved every "saveInterval" time step if
    # unsteady = True and createAnimatedPlots = True! Don't overload
    # your computer! Set createAnimatedPlots to False to save time.
    saveInterval = 2 # Save T at every "saveInterval" time step, for
                     # animated plot
    createAnimatedPlots = False # True or False
    # Set any number of probe positions, relative to L and H (0-1)
    probeX = np.array([0.1, 0.9, 0.1, 0.9])
    probeY = np.array([0.1, 0.1, 0.9, 0.9])
else:
    # For steady-state:
    deltaT = 1e30  # DO NOT CHANGE. WHY IS IT SET LIKE THIS?
    endTime = 1e30 # DO NOT CHANGE. WHY IS IT SET LIKE THIS?

# Boundary condition value preparation
T_init  = 0      # Initial guess for temperature
T_east  = T_init # Default, initialization for (Homogeneous) Neumann
T_west  = T_init # Default, initialization for (Homogeneous) Neumann
T_north = T_init # Default, initialization for (Homogeneous) Neumann
T_south = T_init # Default, initialization for (Homogeneous) Neumann
q_wall  = 0      # Default heat flux at a wall
T_in    = 20     # Inlet temperature
T_north = 10     # North wall Dirichlet value

# Solver inputs
nExplCorrIter = 2000   # Maximum number of explicit correction iterations
nLinSolIter   = 10     # Number of linear solver iterations
resTol        = 0.001  # Convergence criterium for residuals
solver        = 'GS'   # Either GS (Gauss-Seidel) or TDMA

#====================== Code ======================

# Read grid and velocity data:
grid_numbers = [1,1,1,1,1,2,2,2,2,2,3,3,3,3,3,4,4,4,4,4,5,5,5,5,5]
grid_number  = grid_numbers[caseID-1]
path = 'data/grid%d/%s_grid' % (grid_number,grid_type)
pointXvector = np.genfromtxt('%s/xc.dat' % (path)) # x node coordinates
pointYvector = np.genfromtxt('%s/yc.dat' % (path)) # y node coordinates
u_datavector = np.genfromtxt('%s/u.dat' % (path))  # u velocity at the nodes
v_datavector = np.genfromtxt('%s/v.dat' % (path))  # v veloctiy at the nodes

# Preparation of "nan", to fill empty slots in consistently numbered arrays.
# This makes it easier to check in Variable Explorer that values that should
# never be set are never set (or used). Plots simply omit nan values.
nan = float("nan")

# Allocate arrays (nan used to make clear where values need to be set)
# Note that some arrays could actually be 1D since they only have a variation
# in one direction, but they are kept 2D so the indexing is similar for all.
mI     = len(pointXvector);          # Number of mesh points X direction
mJ     = len(pointYvector);          # Number of mesh points X direction
nI     = mI + 1;                     # Number of nodes in X direction, incl. boundaries
nJ     = mJ + 1;                     # Number of nodes in Y direction, incl. boundaries
pointX = np.zeros((mI,mJ))*nan       # X coords of the mesh points, in points
pointY = np.zeros((mI,mJ))*nan       # Y coords of the mesh points, in points
nodeX  = np.zeros((nI,nJ))*nan       # X coords of the nodes, in nodes
nodeY  = np.zeros((nI,nJ))*nan       # Y coords of the nodes, in nodes
dx_PE  = np.zeros((nI,nJ))*nan       # X distance to east node, in nodes
dx_WP  = np.zeros((nI,nJ))*nan       # X distance to west node, in nodes
dy_PN  = np.zeros((nI,nJ))*nan       # Y distance to north node, in nodes
dy_SP  = np.zeros((nI,nJ))*nan       # Y distance to south node, in nodes
dx_we  = np.zeros((nI,nJ))*nan       # X size of the control volume, in nodes
dy_sn  = np.zeros((nI,nJ))*nan       # Y size of the control volume, in nodes
fxe    = np.zeros((nI,nJ))*nan       # Interpolation factor, in nodes
fxw    = np.zeros((nI,nJ))*nan       # Interpolation factor, in nodes
fyn    = np.zeros((nI,nJ))*nan       # Interpolation factor, in nodes
fys    = np.zeros((nI,nJ))*nan       # Interpolation factor, in nodes
aE     = np.zeros((nI,nJ))*nan       # Array for east coefficient, in nodes
aW     = np.zeros((nI,nJ))*nan       # Array for west coefficient, in nodes
aN     = np.zeros((nI,nJ))*nan       # Array for north coefficient, in nodes
aS     = np.zeros((nI,nJ))*nan       # Array for south coefficient, in nodes
aP     = np.zeros((nI,nJ))*nan       # Array for central coefficient, in nodes
Su     = np.zeros((nI,nJ))*nan       # Array for source term for temperature, in nodes
Sp     = np.zeros((nI,nJ))*nan       # Array for source term for temperature, in nodes
T      = np.zeros((nI,nJ))*nan       # Array for temperature, in nodes
T_o    = np.zeros((nI,nJ))*nan       # Array for old temperature, in nodes
De     = np.zeros((nI,nJ))*nan       # Diffusive coefficient for east face, in nodes
Dw     = np.zeros((nI,nJ))*nan       # Diffusive coefficient for west face, in nodes
Dn     = np.zeros((nI,nJ))*nan       # Diffusive coefficient for north face, in nodes
Ds     = np.zeros((nI,nJ))*nan       # Diffusive coefficient for south face, in nodes
Fe     = np.zeros((nI,nJ))*nan       # Convective coefficients for east face, in nodes
Fw     = np.zeros((nI,nJ))*nan       # Convective coefficients for west face, in nodes
Fn     = np.zeros((nI,nJ))*nan       # Convective coefficients for north face, in nodes
Fs     = np.zeros((nI,nJ))*nan       # Convective coefficients for south face, in nodes
P      = np.zeros((nI,nJ))*nan       # Array for TDMA, in nodes
Q      = np.zeros((nI,nJ))*nan       # Array for TDMA, in nodes
u      = u_datavector.reshape(nI,nJ) # Values of x-velocity, in nodes
v      = v_datavector.reshape(nI,nJ) # Values of y-velocity, in nodes
res    = []                          # Array for appending residual each iteration
savedT = []                          # Array for saving T, for animated plot
probeValues = []                     # Array for saving probe values
# Set wall velocities to exactly zero:
u[u == 1e-10] = 0
v[v == 1e-10] = 0

# Create mesh - point coordinates
# (only changes arrays in first row of argument list)
cF.createMesh(pointX, pointY,
              mI, mJ, pointXvector, pointYvector)

# Calculate length and height:
L = pointX[mI-1,0] - pointX[0,0]
H = pointY[0,mJ-1] - pointY[0,0]
# Scale probe locations with L and H
if unsteady:
    probeX*=L
    probeY*=H

# Calculate node positions
# (only changes arrays in first row of argument list)
cF.calcNodePositions(nodeX, nodeY,
                     nI, nJ, pointX, pointY)

# Calculate distances once and keep
# (only changes arrays in first row of argument list)
cF.calcDistances(dx_PE, dx_WP, dy_PN, dy_SP, dx_we, dy_sn,
                 nI, nJ, nodeX, nodeY, pointX, pointY)

# Calculate interpolation factors once and keep
# (only changes arrays in first row of argument list)
cF.calcInterpolationFactors(fxe, fxw, fyn, fys,
                            nI, nJ, dx_PE, dx_WP, dy_PN, dy_SP, dx_we, dy_sn)

# Initialize dependent variable array
# (only changes arrays in first row of argument list)
cF.initArray(T,
             T_init)

# Set Dirichlet boundary conditions
# (only changes arrays in first row of argument list)
cF.setDirichletBCs(T,
                   nI, nJ, u, v, T_in, T_west, T_east, T_south, T_north)

# All cases have constant coefficients and source terms (VERIFY YOURSELF!),
# so we set them outside the main loop...

# Calculate constant diffusive (D) coefficients
# Note that D is here supposed to include the multiplication with area
# (only changes arrays in first row of argument list)
cF.calcD(De, Dw, Dn, Ds,
         gamma, nI, nJ, dx_PE, dx_WP, dy_PN, dy_SP, dx_we, dy_sn)

# Calculate constant convective (F) coefficients
# Note that F is here supposed to include the multiplication with area
# (only changes arrays in first row of argument list)
cF.calcF(Fe, Fw, Fn, Fs,
         rho, nI, nJ, dx_we, dy_sn, fxe, fxw, fyn, fys, u, v)

# Add time loop
saveCounter = 0
for time in np.arange(deltaT, endTime + deltaT, deltaT):

    if unsteady:
        print('Time: ',time)
        saveCounter+=1
    
    # Set old T
    T_o = copy.deepcopy(T)
        
    # Calculate source terms
    # (only changes arrays in first row of argument list)
    cF.calcSourceTerms(Su, Sp,
                       nI, nJ, q_wall, Cp, u, v, dx_we, dy_sn, rho, deltaT, T_o, caseID)
              
    # Calculate coefficients for Hybrid scheme
    # (only changes arrays in first row of argument list)
    cF.calcHybridCoeffs(aE, aW, aN, aS, aP,
                        nI, nJ, De, Dw, Dn, Ds, Fe, Fw, Fn, Fs,
                        fxe, fxw, fyn, fys, dy_sn, Sp, u, v,
                        nodeX, nodeY, L, H, caseID)

    # The following loop includes:
    # * application of the linear solver (nLinSolIter number of times),
    # * explicit updates after the linear solver is applied, and
    # * calculation and reporting of residuals after each explicit update loop.
    # A case that has no explicit updates should only need nExplCorrIter = 1,
    # and sufficient nLinSolIter to converge the results.
    for explCorrIter in range(nExplCorrIter):

        # Solve for T using Gauss-Seidel
        if solver == 'GS':
            # Solve T eq. a number of Gauss-Seidel loops:
            # (only changes arrays in first row of argument list)
            cF.solveGaussSeidel(T,
                                nI, nJ, aE, aW, aN, aS, aP, Su, nLinSolIter)
        
        # Solve for T using TDMA
        if solver == 'TDMA':
            # Solve T eq. a number of TDMA loops:
            # (only changes arrays in first row of argument list)
            cF.solveTDMA(T, P, Q,
                          nI, nJ, aE, aW, aN, aS, aP, Su, nLinSolIter)
    
        # Copy T to boundaries (and corners) where (non-)homegeneous Neumann is applied:
        # (only changes arrays in first row of argument list)
        # (could be moved to after iteration loop if implementation is implicit)
        cF.correctBoundaries(T,
                             nI, nJ, q_wall, k, dx_PE, dx_WP, dy_PN, dy_SP,
                             u, v, nodeX, nodeY, L, H, caseID)

        # Calculate normalized residuals
        # (only changes arrays in first row of argument list)
        cF.calcNormalizedResiduals(res,
                                    nI, nJ, explCorrIter, T,
                                    aP, aE, aW, aN, aS, Su, Sp)

        # Stop iterations if converged:
        if res[-1]/res[0] < resTol:
            break

    # Store data for plots over time / animations
    if unsteady:
        probeValues.append(cF.probe(nodeX, nodeY, T,probeX, probeY))
    if unsteady and createAnimatedPlots and not saveCounter%saveInterval:
        savedT.append(T.copy())
    
#================ Plotting section ================
# Create default plots
# No arrays are changed    
cF.createDefaultPlots(
                      nI, nJ, pointX, pointY, nodeX, nodeY,
                      dx_WP, dx_PE, dy_SP, dy_PN, Fe, Fw, Fn, Fs,
                      aE, aW, aN, aS, L, H, T, u, v, k,
                      explCorrIter, res, grid_type, caseID)

# Create time evolution plots
# No arrays are changed    
if unsteady:
    cF.createTimeEvolutionPlots(
                                probeX, probeY, probeValues, caseID, grid_type)

# Create animated plots:
# No arrays are changed    
if unsteady and createAnimatedPlots:
    cF.createAnimatedPlots(
                          nodeX, nodeY, savedT)

# Create additional plots
# Implement this function for additional plots!
# No arrays should be changed!
cF.createAdditionalPlots(
                         )