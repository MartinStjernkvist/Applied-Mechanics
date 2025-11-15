# %%
# MTF073 Computational Fluid Dynamics
# Task 1: 2D diffusion
# Håkan Nilsson, 2025
# Department of Mechanics and Maritime Sciences
# Division of Fluid Dynamics
# Note that this is not efficient code. It is for educational purposes!

# Clear all variables when running entire code:
from IPython import get_ipython

#get_ipython().run_line_magic('reset', '-sf')
# Packages needed
import numpy as np
import matplotlib.pyplot as plt

# Close all plots when running entire code:
plt.close('all')
import sys  # For sys.exit()
# All functions of the code (some to be implemented by you):
import Raphael_codeFunctions as cF

# Info: Use cF to call functions that are defined in codeFunctions_template.py.
# Hint: Click on the function name and press Ctrl-g to go to the function.
# Rule: Only change values of arrays in the first line of the argument list in the functions!
# Hint: Use the other arguments for the calculations inside the functions.
# Rule: Don't change the list of arguments to any function (except createAdditionalPlots)!

# ===================== Inputs =====================

# Geometric and mesh inputs

L = 1.5  # Length of the domain in X direction
H = 0.5  # Length of the domain in Y direction
mI = 7  # Number of mesh points X direction.
mJ = 4  # Number of mesh points Y direction.
mesh_type = 'equidistant'  # Set 'non-equidistant' or 'equidistant'

# Case-specific input

caseID = 8  # Your case number (only used for testing the code with reference data)
h = 1000  # Keep as it is if you do not have a convective boundary condition
T_inf = 10  # Keep as it is if you do not have a convective boundary condition

# Solver inputs

nExplCorrIter = 1000  # Maximum number of explicit correction iterations
nLinSolIter = 10  # Number of linear solver (Gauss-Seidel) iterations
resTol = 0.001  # Convergence criteria for residuals

# ====================== Code ======================

# Preparation of "nan", to fill empty slots in consistently numbered arrays.
# This makes it easier to check in Variable Explorer that values that should
# never be set are never set (or used). Plots simply omit nan values.
nan = float("nan")

# Allocate arrays (nan used to make clear where values need to be set)
# Note that some arrays could actually be 1D since they only have a variation
# in one direction, but they are kept 2D so the indexing is similar for all.
nI = mI + 1  # Number of nodes in X direction, incl. boundaries
nJ = mJ + 1  # Number of nodes in Y direction, incl. boundaries
pointX = np.zeros((mI, mJ)) * nan  # X coords of the mesh points
pointY = np.zeros((mI, mJ)) * nan  # Y coords of the mesh points
nodeX = np.zeros((nI, nJ)) * nan  # X coords of the nodes
nodeY = np.zeros((nI, nJ)) * nan  # Y coords of the nodes
dx_PE = np.zeros((nI, nJ)) * nan  # X distance to east node
dx_WP = np.zeros((nI, nJ)) * nan  # X distance to west node
dy_PN = np.zeros((nI, nJ)) * nan  # Y distance to north node
dy_SP = np.zeros((nI, nJ)) * nan  # Y distance to south node
dx_we = np.zeros((nI, nJ)) * nan  # X size of the control volume
dy_sn = np.zeros((nI, nJ)) * nan  # Y size of the control volume
fxe = np.zeros((nI, nJ)) * nan  # Interpolation factor, in nodes
fxw = np.zeros((nI, nJ)) * nan  # Interpolation factor, in nodes
fyn = np.zeros((nI, nJ)) * nan  # Interpolation factor, in nodes
fys = np.zeros((nI, nJ)) * nan  # Interpolation factor, in nodes
aE = np.zeros((nI, nJ)) * nan  # Array for east coefficient, in nodes
aW = np.zeros((nI, nJ)) * nan  # Array for west coefficient, in nodes
aN = np.zeros((nI, nJ)) * nan  # Array for north coefficient, in nodes
aS = np.zeros((nI, nJ)) * nan  # Array for south coefficient, in nodes
aP = np.zeros((nI, nJ)) * nan  # Array for central coefficient, in nodes
Su = np.zeros((nI, nJ)) * nan  # Array for source term for temperature, in nodes
Sp = np.zeros((nI, nJ)) * nan  # Array for source term for temperature, in nodes
T = np.zeros((nI, nJ)) * nan  # Array for temperature, in nodes
k = np.zeros((nI, nJ)) * nan  # Array for conductivity, in nodes
k_e = np.zeros((nI, nJ)) * nan  # Array for conductivity at east face
k_w = np.zeros((nI, nJ)) * nan  # Array for conductivity at west face
k_n = np.zeros((nI, nJ)) * nan  # Array for conductivity at north face
k_s = np.zeros((nI, nJ)) * nan  # Array for conductivity at south face
res = []  # Array for appending residual each iteration
glob_imbal_plot = []  # Array for appending glob_imbalance each iteration

# Set mesh point positions
match mesh_type:
    case 'equidistant':
        cF.createEquidistantMesh(pointX, pointY,
                                 mI, mJ, L, H)
    case 'non-equidistant':
        cF.createNonEquidistantMesh(pointX, pointY,
                                    mI, mJ, L, H)
    case _:
        sys.exit("Improper mesh type!")

# Calculate node positions
cF.calcNodePositions(nodeX, nodeY,
                     nI, nJ, pointX, pointY)

# Calculate distances once and keep
cF.calcDistances(dx_PE, dx_WP, dy_PN, dy_SP, dx_we, dy_sn,
                 nI, nJ, nodeX, nodeY, pointX, pointY)

# Calculate interpolation factors once and keep
cF.calcInterpolationFactors(fxe, fxw, fyn, fys,
                            nI, nJ, dx_PE, dx_WP, dy_PN, dy_SP, dx_we, dy_sn)

# Initialize dependent variable array
cF.initArray(T)

# Set Dirichlet boundary conditions according to your case
cF.setDirichletBCs(T,
                   nI, nJ, L, H, nodeX, nodeY, caseID)

# The following loop includes:
# * explicit updates before the linear solver is applied (again),
# * application of the linear solver (nLinSolIter number of times),
# * explicit updates after the linear solver is applied, and
# * calculation and reporting of residuals after each explicit update loop.
# A case that has no explicit updates should only need nExplCorrIter = 1,
# and sufficient nLinSolIter to converge the results.
for explCorrIter in range(nExplCorrIter):

    # Update conductivity arrays k, k_e, k_w, k_n, k_s, according to your case
    # (could be moved to before iteration loop if independent of solution,
    # but keep here if you want to easily test different cases)
    cF.updateConductivityArrays(k, k_e, k_w, k_n, k_s,
                                nI, nJ, nodeX, nodeY, fxe, fxw, fyn, fys, L, H, T, caseID)

    # Update source term arrays Su, Sp according to your case
    # (could be moved to before iteration loop if independent of solution,
    # but keep here if you want to easily test different cases)
    cF.updateSourceTerms(Su, Sp,
                         nI, nJ, dx_we, dy_sn, dx_WP, dx_PE, dy_SP, dy_PN,
                         T, k_w, k_e, k_s, k_n, h, T_inf, caseID)

    # Calculate coefficients according to your case
    # (could be moved to before iteration loop if independent of solution,
    # but keep here if you want to easily test different cases)
    cF.calcCoeffs(aE, aW, aN, aS, aP,
                  nI, nJ, k_w, k_e, k_s, k_n,
                  dy_sn, dx_we, dx_WP, dx_PE, dy_SP, dy_PN, Sp, caseID)

    # Solve T eq. a number of Gauss-Seidel loops
    cF.solveGaussSeidel(T,
                        nI, nJ, aE, aW, aN, aS, aP, Su, nLinSolIter)

    # Copy T to boundaries (and corners) where homegeneous Neumann is applied
    # (could be moved to after iteration loop if implementation is implicit)
    cF.correctBoundaries(T,
                         nI, nJ, k_w, k_e, k_s, k_n,
                         dy_sn, dx_we, dx_WP, dx_PE, dy_SP, dy_PN,
                         h, T_inf, caseID)

    # Calculate and print normalized residuals
    cF.calcNormalizedResiduals(res, glob_imbal_plot,
                               nI, nJ, explCorrIter, T,
                               aP, aE, aW, aN, aS, Su, Sp)

    # Stop iterations if converged
    if res[-1] < resTol:
        break

# ================ Post-processing section ================
# Global heat rate imbalance:
print('Global heat rate imbalance: %.2g%%' % (100 * glob_imbal_plot[-1]))

# ================ Plotting section ================

# Create default plots
# No arrays are changed
cF.createDefaultPlots(
    nI, nJ, pointX, pointY, nodeX, nodeY,
    L, H, T, k,
    explCorrIter, res, glob_imbal_plot, caseID)

# Create additional plots
# Implement this function for additional plots!
# No arrays should be changed!
cF.createAdditionalPlots()

# %%