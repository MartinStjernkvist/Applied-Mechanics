# MTF073 Computational Fluid Dynamics
# Task 3: Pressure-velocity coupling
# Håkan Nilsson, 2025
# Department of Mechanics and Maritime Sciences
# Division of Fluid Dynamics
# Note that this is not efficient code. It is for educational purposes!

# This file gives the overall algorithm of the code. Make sure to read and
# understand it!
#
# The code requires that:
# * the file T3_codeFunctions_template.py is in the same path as this file
# * the folder named 'meshAndVelocityData' is in the same path as this file
# * the file reset.py is in the same path as this file
#
# Note that:
# * Except for the "Inputs" below and the arguments to the
#   createAdditionalPlots function, you should not change anything here,
#   but only do changes to T3_codeFunctions_template.py (cF)
# * Each call to a cF function should only change the arrays in the
#   first row of the argument list. This, together with the name of the function
#   makes it easier to read and understand the code (and avoid coding mistakes).
# * You can easily go to the called function by putting the marker on it and
#   pressing Ctrl-g

# Clear all variables when running entire code:
from reset import universal_reset
universal_reset(protect={'universal_reset'}, verbose=True)

# Packages needed
import numpy as np
import matplotlib.pyplot as plt
plt.close('all')
import sys # For sys.exit()
# All functions of the code (some to be implemented by you):
import T3_codeFunctions_template as cF

#================= Inputs =====================

# Case number (same as case in description, 1-25)
caseID    =  1

# Geometric and mesh inputs (mesh is read from file)
grid_type = 'coarse'  # Either 'coarse', 'fine' or 'newCoarse' (or your own)

# Case inputs:
rho   =  1     # Density
match caseID:
    case 1 | 2 | 3 | 4 | 5:
        mu    =  0.002 # Dynamic viscosity
                       # (0.002 needed to give similar results as in Task 2)
    case 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 | 21 | 22 | 23 | 24 | 25:
        mu    =  0.003 # Dynamic viscosity 
                        # (0.003 needed for convergence using poor meshes 'coarse' and 'fine')
    case 6 | 7 | 8 | 9 | 10:
        mu    =  0.001 # Dynamic viscosity
                       # (0.001 needed to give similar results as in Task 2)
    case _:
        sys.exit("Improper caseID!")

# Solver inputs
nSIMPLEiter    = 1000           # Maximum number of SIMPLE iterations
nLinSolIter_pp = 10             # Number of linear solver iterations for pp-equation
nLinSolIter_uv = 3              # Number of Gauss-Seidel iterations for u/v-equations
resTol         = 0.001          # Set convergence criteria for residuals
alphaUV        = 0.7            # Under-relaxation factor for u and v
alphaP         = 0.3            # Under-relaxation factor for p
linSol_pp      = 'TDMA'         # Either 'GS' or 'TDMA'
scheme         = 'Hybrid'       # Either 'FOU_CD' or 'Hybrid'
RhieChow       = 'equiCorr'     # Either 'noCorr', 'equiCorr' or 'nonEquiCorr'
pRef_i = 3 # P=0 in some internal node (1..nI-2, not on boundary)
pRef_j = 3 # P=0 in some internal node (1..nJ-2, not on boundary)

##############################################################################
# DO NOT CHANGE ANYTHING BELOW!                                              #
# Exception: The arguments to the createAdditionalPlots function, if needed. #
# Read and understand the code and the algorithm!                            #
##############################################################################

# ================ Code =======================

# Read grid and Task 2 velocity data:
meshAndVelocityData = np.load('meshAndVelocityData/Case_'+str(caseID)+'_'+ \
                              'meshAndVelocityArrays_'+grid_type+'.npz')
pointX = meshAndVelocityData['pointX']
pointY = meshAndVelocityData['pointY']
mI, mJ = pointX.shape           # Number of mesh points in X (i) and Y (j) directions
nI     = mI + 1                 # Number of nodes in X direction, incl. boundaries
nJ     = mJ + 1                 # Number of nodes in Y direction, incl. boundaries
uTask2 = np.zeros((nI,nJ))*float("nan")
vTask2 = np.zeros((nI,nJ))*float("nan")
# Only load velocity data for provided meshes:
match grid_type:
    case 'fine' | 'coarse' | 'newCoarse' :
        uTask2 = meshAndVelocityData['u']
        vTask2 = meshAndVelocityData['v']
    # case _:                             # Commented to allow user-made meshes
    #     sys.exit("Improper grid_type!") # Commented to allow user-made meshes
del meshAndVelocityData # Make sure that the file is released
import gc               # Make sure that the file is released
gc.collect()            # Make sure that the file is released
# Calculate length and height:
L = pointX[-1,0] - pointX[0,0]
H = pointY[0,-1] - pointY[0,0]

# Preparation of "nan", to fill empty slots in consistently numbered arrays.
# This makes it easier to check in Variable Explorer that values that should
# never be set are never set (or used). Plots simply omit nan values.
nan = float("nan")

# Allocate arrays (nan used to make clear where values need to be set)
# Note that some arrays could actually be 1D since they only have a variation
# in one direction, but they are kept 2D so the indexing is similar for all.
#####################################
# Better alternative:               #
# nodeX = np.full((nI, nJ), np.nan) #
#####################################
nodeX  = np.zeros((nI,nJ))*nan  # X coords of the nodes, in nodes
nodeY  = np.zeros((nI,nJ))*nan  # Y coords of the nodes, in nodes
dx_PE  = np.zeros((nI,nJ))*nan  # X distance to east node, in nodes
dx_WP  = np.zeros((nI,nJ))*nan  # X distance to west node, in nodes
dy_PN  = np.zeros((nI,nJ))*nan  # Y distance to north node, in nodes
dy_SP  = np.zeros((nI,nJ))*nan  # Y distance to south node, in nodes
dx_we  = np.zeros((nI,nJ))*nan  # X size of the control volume, in nodes
dy_sn  = np.zeros((nI,nJ))*nan  # Y size of the control volume, in nodes
aE_uv  = np.zeros((nI,nJ))*nan  # East coefficient for velocities, in nodes
aW_uv  = np.zeros((nI,nJ))*nan  # West coefficient for velocities, in nodes
aN_uv  = np.zeros((nI,nJ))*nan  # North coefficient for velocities, in nodes
aS_uv  = np.zeros((nI,nJ))*nan  # South coefficient for velocities, in nodes
aP_uv  = np.zeros((nI,nJ))*nan  # Central coefficient for velocities, in nodes
Su_u   = np.zeros((nI,nJ))*nan  # Source term for u-velocity, in nodes
Su_v   = np.zeros((nI,nJ))*nan  # Source term for v-velocity, in nodes
aE_pp  = np.zeros((nI,nJ))*nan  # East coefficient for p', in nodes
aW_pp  = np.zeros((nI,nJ))*nan  # West coefficient for p', in nodes
aN_pp  = np.zeros((nI,nJ))*nan  # North coefficient for p', in nodes
aS_pp  = np.zeros((nI,nJ))*nan  # South coefficient for p', in nodes
aP_pp  = np.zeros((nI,nJ))*nan  # Central coefficient for p', in nodes
Su_pp  = np.zeros((nI,nJ))*nan  # Source term for p', in nodes
u      = np.zeros((nI,nJ))*nan  # u-velocity, in nodes
v      = np.zeros((nI,nJ))*nan  # v-velocity, in nodes
p      = np.zeros((nI,nJ))*nan  # Pressure, in nodes (lower-case p)
pp     = np.zeros((nI,nJ))*nan  # Pressure correction, in nodes
De     = np.zeros((nI,nJ))*nan  # Diffusive coefficient for east face, in nodes
Dw     = np.zeros((nI,nJ))*nan  # Diffusive coefficient for west face, in nodes
Dn     = np.zeros((nI,nJ))*nan  # Diffusive coefficient for north face, in nodes
Ds     = np.zeros((nI,nJ))*nan  # Diffusive coefficient for south face, in nodes
Fe     = np.zeros((nI,nJ))*nan  # Convective coefficients for east face, in nodes
Fw     = np.zeros((nI,nJ))*nan  # Convective coefficients for west face, in nodes
Fn     = np.zeros((nI,nJ))*nan  # Convective coefficients for north face, in nodes
Fs     = np.zeros((nI,nJ))*nan  # Convective coefficients for south face, in nodes
de     = np.zeros((nI,nJ))*nan  # Coefficient for east face, in nodes
dw     = np.zeros((nI,nJ))*nan  # Coefficient for west face, in nodes
dn     = np.zeros((nI,nJ))*nan  # Coefficient for north face, in nodes
ds     = np.zeros((nI,nJ))*nan  # Coefficient for south face, in nodes
fxe    = np.zeros((nI,nJ))*nan  # Interpolation factor, in nodes
fxw    = np.zeros((nI,nJ))*nan  # Interpolation factor, in nodes
fyn    = np.zeros((nI,nJ))*nan  # Interpolation factor, in nodes
fys    = np.zeros((nI,nJ))*nan  # Interpolation factor, in nodes
res_u  = []                     # Array for appending u-residual each iteration
res_v  = []                     # Array for appending v-residual each iteration
res_c  = []                     # Array for appending continuity error each iteration

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

# Initialize dependent variable arrays
# Note that F is here supposed to include the multiplication with area
# (only changes arrays in first row of argument list)
cF.initArrays(u, v, p, Fe, Fw, Fn, Fs)

# Set Dirichlet boundary conditions
# Note that F is here supposed to include the multiplication with area
# (only changes arrays in first row of argument list)
cF.setInletVelocityAndFlux(u, v, Fe, Fw, Fn, Fs,
                            nI, nJ, rho, dx_we, dy_sn, nodeX, nodeY, grid_type, caseID)

# Initialize outlet flux
# Note that F is here supposed to include the multiplication with area
# (only changes arrays in first row of argument list)
cF.initOutletFlux(Fe, Fw, Fn, Fs,
                  nI, nJ, rho, dx_we, dy_sn, nodeX, nodeY, grid_type, caseID)

# Correct outlet flux to get global continuity
# Note that F is here supposed to include the multiplication with area
# (only changes arrays in first row of argument list)
cF.correctGlobalContinuity(Fe, Fw, Fn, Fs,
                            nI, nJ)            

# Calculate diffusions conductances of mom.eq. discretization once and keep
# Note that D is here supposed to include the multiplication with area
# (only changes arrays in first row of argument list)
cF.calcD(De, Dw, Dn, Ds,
          mu, nI, nJ, dx_PE, dx_WP, dy_PN, dy_SP, dx_we, dy_sn)
        
# SIMPLE loop
for iter in range(nSIMPLEiter):

    # Calculate under-relaxed momentum equation coefficients
    # (only changes arrays in first row of argument list)
    match scheme:
        case 'FOU_CD':
            # OPTIONAL!
            # First-Order Upwind for convection.
            # Central Differencing for diffusion.
            cF.calcMomEqCoeffs_FOU_CD(aE_uv, aW_uv, aN_uv, aS_uv, aP_uv,
                                      nI, nJ, alphaUV, De, Dw, Dn, Ds,
                                      Fe, Fw, Fn, Fs)
        case 'Hybrid':
            # Hybrid scheme.
            cF.calcMomEqCoeffs_Hybrid(aE_uv, aW_uv, aN_uv, aS_uv, aP_uv,
                                      nI, nJ, alphaUV, De, Dw, Dn, Ds,
                                      Fe, Fw, Fn, Fs, fxe, fxw, fyn, fys)
        case _:
            sys.exit("Improper scheme!")

    # Calculate under-relaxed momentum equation source terms
    # (only changes arrays in first row of argument list)
    cF.calcMomEqSu(Su_u, Su_v,
                    nI, nJ, p, dx_WP, dx_PE, dy_SP, dy_PN, dx_we, dy_sn,
                    alphaUV, aP_uv, u, v, fxe, fxw, fyn, fys)

    # Solve u-mom. eq. a number of Gauss-Seidel loops:
    # (only changes arrays in first row of argument list)
    cF.solveGaussSeidel(u,
                        nI, nJ, aE_uv, aW_uv, aN_uv, aS_uv, aP_uv, Su_u,
                        nLinSolIter_uv)

    # Solve v-mom. eq. a number of Gauss-Seidel loops:
    # (only changes arrays in first row of argument list)
    cF.solveGaussSeidel(v,
                        nI, nJ, aE_uv, aW_uv, aN_uv, aS_uv, aP_uv, Su_v,
                        nLinSolIter_uv)

    # Calculate face fluxes using Rhie & Chow (or not Rhie & Chow)
    match RhieChow:
        case 'noCorr':
            # No Rhie & Chow correction.
            # Note that F is here supposed to include the multiplication with area
            # (only changes arrays in first row of argument list)
            cF.calcRhieChow_noCorr(Fe, Fw, Fn, Fs,
                                    nI, nJ, rho, u, v,
                                    dx_we, dy_sn, fxe, fxw, fyn, fys)
        case 'equiCorr':
            # Equidistant implementation of Rhie & Chow correction term.
            # Note that F is here supposed to include the multiplication with area
            # (only changes arrays in first row of argument list)
            cF.calcRhieChow_equiCorr(Fe, Fw, Fn, Fs,
                                      nI, nJ, rho, u, v,
                                      dx_we, dy_sn, fxe, fxw, fyn, fys, aP_uv, p)
        case 'nonEquiCorr':
            # OPTIONAL!
            # Non-equidistant implementation of everything.
            # Note that F is here supposed to include the multiplication with area
            # (only changes arrays in first row of argument list)
            cF.calcRhieChow_nonEquiCorr(Fe, Fw, Fn, Fs,
                                        nI, nJ, rho, u, v,
                                        dx_we, dy_sn, fxe, fxw, fyn, fys, aP_uv, p,
                                        dx_WP, dx_PE, dy_SP, dy_PN)
        case _:
            sys.exit("Improper RhieChow choice!")
    
    # Calculate pressure correction equation coefficients
    # (only changes arrays in first row of argument list)
    cF.calcPpEqCoeffs(aE_pp, aW_pp, aN_pp, aS_pp, aP_pp, de, dw, dn, ds,
                      nI, nJ, rho, dx_we, dy_sn, fxe, fxw, fyn, fys, aP_uv)

    # Calculate pressure correction equation source term
    # (only changes arrays in first row of argument list)
    cF.calcPpEqSu(Su_pp,
                  nI, nJ, Fe, Fw, Fn, Fs)

    # Fix pressure by forcing pp to zero in reference node, through source terms
    # MAKES CONVERGENCE POOR, SO BETTER TO SKIP IT FOR NOW. TRY IF YOU LIKE
    # (only changes arrays in first row of argument list)
    # fixPp(Su_pp, aP_pp,
    #       pRef_i, pRef_j, aE_pp, aW_pp, aN_pp, aS_pp)

    # Solve pressure correction equation
    pp[:,:] = 0
    match linSol_pp:
        case 'GS':
            # Solve pp eq. a number of Gauss-Seidel loops:
            # (only changes arrays in first row of argument list)
            cF.solveGaussSeidel(pp,
                                nI, nJ, aE_pp, aW_pp, aN_pp, aS_pp, aP_pp, Su_pp,
                                nLinSolIter_pp)
        case 'TDMA':
            # Solve pp eq. a number of TDMA loops:
            # (only changes arrays in first row of argument list)
            cF.solveTDMA(pp,
                          nI, nJ, aE_pp, aW_pp, aN_pp, aS_pp, aP_pp, Su_pp,
                          nLinSolIter_pp)
        case _:
            sys.exit("Improper linSolPp!")
                    
    # Set pressure correction level explicitly
    # (only changes arrays in first row of argument list)
    cF.setPressureCorrectionLevel(pp,
                                  nI, nJ, pRef_i, pRef_j)

    # Correct pressure correction homogeneous Neumann boundary conditions
    # (only changes arrays in first row of argument list)
    cF.correctPressureCorrectionBC(pp,
                                    nI, nJ)

    # Correct pressure, using explicit under-relaxation
    # (only changes arrays in first row of argument list)
    cF.correctPressure(p,
                        nI, nJ, alphaP, pp)

    # Extrapolate pressure to boundaries, using constant gradient,
    # required to get correct Suu in u-mom. equation!
    # Also set reasonable corner values, for post-processing.
    # (only changes arrays in first row of argument list)
    cF.correctPressureBCandCorners(p,
                                    nI, nJ, dx_PE, dx_WP, dy_PN, dy_SP)

    # Correct velocity components using pp solution
    # (only changes arrays in first row of argument list)
    cF.correctVelocity(u, v,
                        nI, nJ, fxe, fxw, fyn, fys, pp, dy_sn, dx_we, aP_uv)

    # Extraplate velocity at outlet
    # (only changes arrays in first row of argument list)
    cF.correctOutletVelocity(u, v,
                              nI, nJ, rho, dx_we, dy_sn, nodeX, nodeY, grid_type, caseID)

    # Correct face fluxes using pp solution
    # Note that F is here supposed to include the multiplication with area
    # (only changes arrays in first row of argument list)
    cF.correctFaceFlux(Fe, Fw, Fn, Fs,
                        nI, nJ, rho, dy_sn, dx_we, de, dw, dn, ds, pp)

    # Calculate normalized residuals
    # (only changes arrays in first row of argument list)
    cF.calcNormalizedResiduals(res_u, res_v, res_c,
                                nI, nJ, iter, u, v,
                                aP_uv, aE_uv, aW_uv, aN_uv, aS_uv, Su_u, Su_v,
                                Fe, Fw, Fn, Fs)

    print('Iter: %5d, resU = %.5e, resV = %.5e, resCon = %.5e'
        % (iter, res_u[-1], res_v[-1], res_c[-1]))
    
    #  Check convergence
    if max([res_u[-1], res_v[-1], res_c[-1]]) < resTol:
        break

#================ Postprocessing section ================

# Create default plots
# No arrays are changed    
cF.createDefaultPlots(
                      nI, nJ, pointX, pointY, nodeX, nodeY, pRef_i, pRef_j,
                      caseID, grid_type, u, v, uTask2, vTask2, p,
                      iter, res_u, res_v, res_c)

# Create additional plots
# Implement this function for additional plots!
# No arrays should be changed!
cF.createAdditionalPlots(
                          )