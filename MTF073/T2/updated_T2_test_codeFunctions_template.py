#%%
# The header of the corresponding code for Task 1 is relevant also here.
# It is not repeated. Please remind yourself if needed.

# Clear all variables when running entire code:
from IPython import get_ipython
get_ipython().run_line_magic('reset', '-sf')
import numpy as np
import copy
import T2_codeFunctions as cF

#===================== Inputs =====================

# Case number (same as case in description, 1-25)
caseID  = 8
rho     = 1      # Density
k       = 1      # Thermal conductivity 
Cp      = 500    # Specific heat
gamma = k/Cp     # Calculated diffusion coefficient

# Boundary condition value preparation
T_in    = 20     # Inlet temperature
T_init  = 0      # Initial guess for temperature
T_east  = T_init # Default, initialization for (Homogeneous) Neumann
T_west  = T_init # Default, initialization for (Homogeneous) Neumann
T_north = T_init # Default, initialization for (Homogeneous) Neumann
T_south = T_init # Default, initialization for (Homogeneous) Neumann
q_wall  = 100      # Default heat flux at a wall (specified later)

# Functions to check (True / False):
check_calcDistances = True
check_calcInterpolationFactors = True
check_setDirichletBCs = True
check_calcSourceTerms = True
check_calcD = True
check_calcF = True
check_calcHybridCoeffs = True
check_solveGaussSeidel = True
check_solveTDMA = True
check_correctBoundaries = True

# Using numpy.testing.assert_allclose
# https://numpy.org/doc/stable/reference/generated/numpy.testing.assert_allclose.html#numpy.testing.assert_allclose
# Testing tolerances:
aTol = 1e-15
rTol = 1e-07
# Uncomment below to check if arrays are exactly equal
# (will only be the case if the order of operations are exactly the same)
# aTol = 0
# rTol = 0

###########################################
# DON'T CHANGE ANYTHING BELOW!            #
# DON'T TRY TO UNDERSTAND THE CODE BELOW! #
###########################################
# For Håkan: First switch useModData to False to generate modified arrays. Then switch back.
# This does not work for students.
# Note: Modified arrays is for those where the output of the function will not (necessarily)
# be the same as the reference data.
# The names of the modified data arrays end with an abbreviation of the function name.

nLinSolIter = 10
# Operation (Set False once to generate all modified data and then only True):
useModData =  True
# useModData =  False

# Load reference arrays
refData = np.load('refData/Case_'+str(caseID)+'_'+'refArrays.npz')
pointX_ref = refData['pointX']
pointY_ref = refData['pointY']
nodeX_ref = refData['nodeX']
nodeY_ref = refData['nodeY']
dx_PE_ref = refData['dx_PE']
dx_WP_ref = refData['dx_WP']
dy_PN_ref = refData['dy_PN']
dy_SP_ref = refData['dy_SP']
dx_we_ref = refData['dx_we']
dy_sn_ref = refData['dy_sn']
fxe_ref = refData['fxe']
fxw_ref = refData['fxw']
fyn_ref = refData['fyn']
fys_ref = refData['fys']
aE_ref = refData['aE']
aW_ref = refData['aW']
aN_ref = refData['aN']
aS_ref = refData['aS']
aP_ref = refData['aP']
Su_ref = refData['Su']
Sp_ref = refData['Sp']
T_ref = refData['T']
T_o_ref = refData['T_o']
T_o_ref = refData['T'] # Use T instead of T_o since T_o=0 for steady-state
De_ref = refData['De']
Dw_ref = refData['Dw']
Dn_ref = refData['Dn']
Ds_ref = refData['Ds']
Fe_ref = refData['Fe']
Fw_ref = refData['Fw']
Fn_ref = refData['Fn']
Fs_ref = refData['Fs']
P_ref = refData['P']
Q_ref = refData['Q']
u_ref = refData['u']
v_ref = refData['v']

# Copy reference arrays
pointX = copy.deepcopy(pointX_ref)
pointY = copy.deepcopy(pointY_ref)
nodeX = copy.deepcopy(nodeX_ref)
nodeY = copy.deepcopy(nodeY_ref)
dx_PE = copy.deepcopy(dx_PE_ref)
dx_WP = copy.deepcopy(dx_WP_ref)
dy_PN = copy.deepcopy(dy_PN_ref)
dy_SP = copy.deepcopy(dy_SP_ref)
dx_we = copy.deepcopy(dx_we_ref)
dy_sn = copy.deepcopy(dy_sn_ref)
fxe = copy.deepcopy(fxe_ref)
fxw = copy.deepcopy(fxw_ref)
fyn = copy.deepcopy(fyn_ref)
fys = copy.deepcopy(fys_ref)
aE = copy.deepcopy(aE_ref)
aW = copy.deepcopy(aW_ref)
aN = copy.deepcopy(aN_ref)
aS = copy.deepcopy(aS_ref)
aP = copy.deepcopy(aP_ref)
Su = copy.deepcopy(Su_ref)
Sp = copy.deepcopy(Sp_ref)
T = copy.deepcopy(T_ref)
T_o = copy.deepcopy(T_o_ref)
De = copy.deepcopy(De_ref)
Dw = copy.deepcopy(Dw_ref)
Dn = copy.deepcopy(Dn_ref)
Ds = copy.deepcopy(Ds_ref)
Fe = copy.deepcopy(Fe_ref)
Fw = copy.deepcopy(Fw_ref)
Fn = copy.deepcopy(Fn_ref)
Fs = copy.deepcopy(Fs_ref)
P = copy.deepcopy(P_ref)
Q = copy.deepcopy(Q_ref)
u = copy.deepcopy(u_ref)
v = copy.deepcopy(v_ref)

if useModData:
    # Load modified arrays
    modData = np.load('refData/Case_'+str(caseID)+'_'+'modArrays.npz')
    dx_PE_cD = modData['dx_PE_cD']
    dx_WP_cD = modData['dx_WP_cD']
    dy_PN_cD = modData['dy_PN_cD']
    dy_SP_cD = modData['dy_SP_cD']
    dx_we_cD = modData['dx_we_cD']
    dy_sn_cD = modData['dy_sn_cD']
    fxe_cIF = modData['fxe_cIF']
    fxw_cIF = modData['fxw_cIF']
    fyn_cIF = modData['fyn_cIF']
    fys_cIF = modData['fys_cIF']
    T_sDBCs = modData['T_sDBCs']
    Su_cST_steady = modData['Su_cST_steady']
    Sp_cST_steady = modData['Sp_cST_steady']
    Su_cST_unsteady = modData['Su_cST_unsteady']
    Sp_cST_unsteady = modData['Sp_cST_unsteady']
    De_cD = modData['De_cD']
    Dw_cD = modData['Dw_cD']
    Dn_cD = modData['Dn_cD']
    Ds_cD = modData['Ds_cD']
    Fe_cF = modData['Fe_cF']
    Fw_cF = modData['Fw_cF']
    Fn_cF = modData['Fn_cF']
    Fs_cF = modData['Fs_cF']
    aE_cHC = modData['aE_cHC']
    aW_cHC = modData['aW_cHC']
    aN_cHC = modData['aN_cHC']
    aS_cHC = modData['aS_cHC']
    aP_cHC = modData['aP_cHC']
    T_sGS = modData['T_sGS']
    T_sTDMA = modData['T_sTDMA']
    P_sTDMA = modData['P_sTDMA']
    Q_sTDMA = modData['Q_sTDMA']
    T_cB = modData['T_cB']
    
# Set numerical domain size
nI, nJ = np.shape(nodeX_ref)
mI, mJ = np.shape(pointX_ref)

# Calculate length and height:
L = pointX[mI-1,0] - pointX[0,0]
H = pointY[0,mJ-1] - pointY[0,0]

def compare(phi, phi_ref, rTol, aTol, fName, aName, ref, func):
    fName_aName = fName + ', ' + aName
    try:
        np.testing.assert_allclose(phi, phi_ref, rtol=rTol, atol=aTol, verbose=False, err_msg=fName+': Array '+aName)
        print(fName_aName.ljust(32)+'OK')
    except AssertionError as e:
        print(fName_aName.ljust(32)+'NOT OK')
        print('    Compare '+aName+'_'+ref+' (reference output)'+' and '+aName+'_'+func+' (your output)')
        print('    -----------------------------------------------------')
        print('    '+str(e).splitlines()[-3])
        print('    '+str(e).splitlines()[-2])
        print('    '+str(e).splitlines()[-1])
        print('    -----------------------------------------------------')

# Test all student-implemented functions:
# ###############################################################################
# All values should be calculated, so set to zero first to avoid that a
# function that doesn't calculate any value is marked as OK
dx_PE*=0
dx_WP*=0
dy_PN*=0
dy_SP*=0
dx_we*=0
dy_sn*=0
cF.calcDistances(dx_PE, dx_WP, dy_PN, dy_SP, dx_we, dy_sn,
                 nI, nJ, nodeX, nodeY, pointX, pointY)
if check_calcDistances and useModData:
    compare(dx_PE, dx_PE_cD, rTol, aTol, 'calcDistances', 'dx_PE', 'cD', 'cD_your')
    compare(dx_WP, dx_WP_cD, rTol, aTol, 'calcDistances', 'dx_WP', 'cD', 'cD_your')
    compare(dy_PN, dy_PN_cD, rTol, aTol, 'calcDistances', 'dy_PN', 'cD', 'cD_your')
    compare(dy_SP, dy_SP_cD, rTol, aTol, 'calcDistances', 'dy_SP', 'cD', 'cD_your')
    compare(dx_we, dx_we_cD, rTol, aTol, 'calcDistances', 'dx_we', 'cD', 'cD_your')
    compare(dy_sn, dy_sn_cD, rTol, aTol, 'calcDistances', 'dy_sn', 'cD', 'cD_your')
    # Save your modified arrays:
    dx_PE_cD_your = copy.deepcopy(dx_PE)
    dx_WP_cD_your = copy.deepcopy(dx_WP)
    dy_PN_cD_your = copy.deepcopy(dy_PN)
    dy_SP_cD_your = copy.deepcopy(dy_SP)
    dx_we_cD_your = copy.deepcopy(dx_we)
    dy_sn_cD_your = copy.deepcopy(dy_sn)
if not check_calcDistances and useModData:
    print('calcDistances:               NOT CHECKED')
# Save reference modified arrays:
if not useModData:
    dx_PE_cD = copy.deepcopy(dx_PE)
    dx_WP_cD = copy.deepcopy(dx_WP)
    dy_PN_cD = copy.deepcopy(dy_PN)
    dy_SP_cD = copy.deepcopy(dy_SP)
    dx_we_cD = copy.deepcopy(dx_we)
    dy_sn_cD = copy.deepcopy(dy_sn)
# Reset modifed arrays:
dx_PE = copy.deepcopy(dx_PE_ref)
dx_WP = copy.deepcopy(dx_WP_ref)
dy_PN = copy.deepcopy(dy_PN_ref)
dy_SP = copy.deepcopy(dy_SP_ref)
dx_we = copy.deepcopy(dx_we_ref)
dy_sn = copy.deepcopy(dy_sn_ref)
###############################################################################
# All values should be calculated, so set to zero first to avoid that a
# function that doesn't calculate any value is marked as OK
fxe*=0
fxw*=0
fyn*=0
fys*=0
cF.calcInterpolationFactors(fxe, fxw, fyn, fys,
                            nI, nJ, dx_PE, dx_WP, dy_PN, dy_SP, dx_we, dy_sn)
if check_calcInterpolationFactors and useModData:
    compare(fxe, fxe_cIF, rTol, aTol, 'calcInterpolationFactors', 'fxe', 'cIF', 'cIF_your')
    compare(fxw, fxw_cIF, rTol, aTol, 'calcInterpolationFactors', 'fxw', 'cIF', 'cIF_your')
    compare(fyn, fyn_cIF, rTol, aTol, 'calcInterpolationFactors', 'fyn', 'cIF', 'cIF_your')
    compare(fys, fys_cIF, rTol, aTol, 'calcInterpolationFactors', 'fys', 'cIF', 'cIF_your')
    # Save your modified arrays:
    fxe_cIF_your = copy.deepcopy(fxe)
    fxw_cIF_your = copy.deepcopy(fxw)
    fyn_cIF_your = copy.deepcopy(fyn)
    fys_cIF_your = copy.deepcopy(fys)
if not check_calcInterpolationFactors and useModData:
    print('calcInterpolationFactors:    NOT CHECKED')
# Save reference modified arrays:
if not useModData:
    fxe_cIF = copy.deepcopy(fxe)
    fxw_cIF = copy.deepcopy(fxw)
    fyn_cIF = copy.deepcopy(fyn)
    fys_cIF = copy.deepcopy(fys)
# Reset modified arrays:
fxe = copy.deepcopy(fxe_ref)
fxw = copy.deepcopy(fxw_ref)
fyn = copy.deepcopy(fyn_ref)
fys = copy.deepcopy(fys_ref)
###############################################################################
# All values should be calculated, so set to zero first to avoid that a
# function that doesn't calculate any value is marked as OK
# Note that we are only interested in the boundary values (which are set by the function)
T[:,:] = 0
cF.setDirichletBCs(T,
                   nI, nJ, u, v, T_in, T_west, T_east, T_south, T_north)
if check_setDirichletBCs and useModData:
    compare(T, T_sDBCs, rTol, aTol, 'setDirichletBCs', 'T', 'sDBCs', 'sDBCs_your')
    # Save your modified arrays:
    T_sDBCs_your = copy.deepcopy(T)
    T_sDBCs_your = copy.deepcopy(T)
if not check_setDirichletBCs and useModData:
    print('setDirichletBCs:             NOT CHECKED')
# Save reference modified arrays:
if not useModData:
    T_sDBCs = copy.deepcopy(T)
    T_sDBCs = copy.deepcopy(T)
# Reset modified arrays:
T = copy.deepcopy(T_ref)
###############################################################################
# All values should be calculated, so set to nan first to avoid that a
# function that doesn't calculate any value is marked as OK
# We don't set to zero since zero may also be correct and a function that is
# not yet implemented would then also return zero.
Su*=float("nan")
Sp*=float("nan")
# Steady-state only
deltaT = 1e30
cF.calcSourceTerms(Su, Sp,
                   nI, nJ, q_wall, Cp, u, v, dx_we, dy_sn, rho, deltaT, T_o, caseID)
if check_calcSourceTerms and useModData:
    compare(Su, Su_cST_steady, rTol, aTol, 'calcSourceTerms (steady)', 'Su', 'cST_steady', 'cST_steady_your')
    compare(Sp, Sp_cST_steady, rTol, aTol, 'calcSourceTerms (steady)', 'Sp', 'cST_steady', 'cST_steady_your')
    # Save your modified arrays:
    Su_cST_steady_your = copy.deepcopy(Su)
    Sp_cST_steady_your = copy.deepcopy(Sp)
if not check_calcSourceTerms and useModData:
    print('calcSourceTerms (steady):    NOT CHECKED')
# Save reference modified arrays:
if not useModData:
    Su_cST_steady = copy.deepcopy(Su)
    Sp_cST_steady = copy.deepcopy(Sp)
# Reset modified arrays:
Su = copy.deepcopy(Su_ref)
Sp = copy.deepcopy(Sp_ref)
###############################################################################
# All values should be calculated, so set to nan first to avoid that a
# function that doesn't calculate any value is marked as OK
# We don't set to zero since zero may also be correct and a function that is
# not yet implemented would then also return zero.
Su*=float("nan")
Sp*=float("nan")
# Unsteady only
deltaT = 2 # Do not set to 1, since missing division with deltaT will not be captured.
cF.calcSourceTerms(Su, Sp,
                   nI, nJ, q_wall, Cp, u, v, dx_we, dy_sn, rho, deltaT, T_o, caseID)
if check_calcSourceTerms and useModData:
    compare(Su, Su_cST_unsteady, rTol, aTol, 'calcSourceTerms (unsteady)', 'Su', 'cST_unsteady', 'cST_unsteady_your')
    compare(Sp, Sp_cST_unsteady, rTol, aTol, 'calcSourceTerms (unsteady)', 'Sp', 'cST_unsteady', 'cST_unsteady_your')
    # Save your modified arrays:
    Su_cST_unsteady_your = copy.deepcopy(Su)
    Sp_cST_unsteady_your = copy.deepcopy(Sp)
if not check_calcSourceTerms and useModData:
    print('calcSourceTerms (unsteady):  NOT CHECKED')
# Save reference modified arrays:
if not useModData:
    Su_cST_unsteady = copy.deepcopy(Su)
    Sp_cST_unsteady = copy.deepcopy(Sp)
# Reset modified arrays:
Su = copy.deepcopy(Su_ref)
Sp = copy.deepcopy(Sp_ref)
###############################################################################
# All values should be calculated, so set to zero first to avoid that a
# function that doesn't calculate any value is marked as OK
De*=0
Dw*=0
Dn*=0
Ds*=0
cF.calcD(De, Dw, Dn, Ds,
         gamma, nI, nJ, dx_PE, dx_WP, dy_PN, dy_SP, dx_we, dy_sn)
if check_calcD and useModData:
    compare(De, De_cD, rTol, aTol, 'calcD', 'De', 'cD', 'cD_your')
    compare(Dw, Dw_cD, rTol, aTol, 'calcD', 'Dw', 'cD', 'cD_your')
    compare(Dn, Dn_cD, rTol, aTol, 'calcD', 'Dn', 'cD', 'cD_your')
    compare(Ds, Ds_cD, rTol, aTol, 'calcD', 'Ds', 'cD', 'cD_your')
    # Save your modified arrays:
    De_cD_your = copy.deepcopy(De)
    Dw_cD_your = copy.deepcopy(Dw)
    Dn_cD_your = copy.deepcopy(Dn)
    Ds_cD_your = copy.deepcopy(Ds)
if not check_calcD and useModData:
    print('calcD:                       NOT CHECKED')
# Save reference modified arrays:
if not useModData:
    De_cD = copy.deepcopy(De)
    Dw_cD = copy.deepcopy(Dw)
    Dn_cD = copy.deepcopy(Dn)
    Ds_cD = copy.deepcopy(Ds)
# Reset modified arrays:
De = copy.deepcopy(De_ref)
Dw = copy.deepcopy(Dw_ref)
Dn = copy.deepcopy(Dn_ref)
Ds = copy.deepcopy(Ds_ref)
###############################################################################
# All values should be calculated, so set to zero first to avoid that a
# function that doesn't calculate any value is marked as OK
Fe*=0
Fw*=0
Fn*=0
Fs*=0
cF.calcF(Fe, Fw, Fn, Fs,
         rho, nI, nJ, dx_we, dy_sn, fxe, fxw, fyn, fys, u, v)
if check_calcF and useModData:
    compare(Fe, Fe_cF, rTol, aTol, 'calcF', 'Fe', 'cF', 'cF_your')
    compare(Fw, Fw_cF, rTol, aTol, 'calcF', 'Fw', 'cF', 'cF_your')
    compare(Fn, Fn_cF, rTol, aTol, 'calcF', 'Fn', 'cF', 'cF_your')
    compare(Fs, Fs_cF, rTol, aTol, 'calcF', 'Fs', 'cF', 'cF_your')
    # Save your modified arrays:
    Fe_cF_your = copy.deepcopy(Fe)
    Fw_cF_your = copy.deepcopy(Fw)
    Fn_cF_your = copy.deepcopy(Fn)
    Fs_cF_your = copy.deepcopy(Fs)
if not check_calcF and useModData:
    print('calcF:                       NOT CHECKED')
# Save reference modified arrays:
if not useModData:
    Fe_cF = copy.deepcopy(Fe)
    Fw_cF = copy.deepcopy(Fw)
    Fn_cF = copy.deepcopy(Fn)
    Fs_cF = copy.deepcopy(Fs)
# Reset modified arrays:
Fe = copy.deepcopy(Fe_ref)
Fw = copy.deepcopy(Fw_ref)
Fn = copy.deepcopy(Fn_ref)
Fs = copy.deepcopy(Fs_ref)
###############################################################################
# All values should be calculated, so set to zero first to avoid that a
# function that doesn't calculate any value is marked as OK
aE*=0
aW*=0
aN*=0
aS*=0
aP*=0
cF.calcHybridCoeffs(aE, aW, aN, aS, aP,
                    nI, nJ, De, Dw, Dn, Ds, Fe, Fw, Fn, Fs,
                    fxe, fxw, fyn, fys, dy_sn, Sp, u, v,
                    nodeX, nodeY, L, H, caseID)
if check_calcHybridCoeffs and useModData:
    compare(aE, aE_cHC, rTol, aTol, 'calcHybridCoeffs', 'aE', 'cHC', 'cHC_your')
    compare(aW, aW_cHC, rTol, aTol, 'calcHybridCoeffs', 'aW', 'cHC', 'cHC_your')
    compare(aN, aN_cHC, rTol, aTol, 'calcHybridCoeffs', 'aN', 'cHC', 'cHC_your')
    compare(aS, aS_cHC, rTol, aTol, 'calcHybridCoeffs', 'aS', 'cHC', 'cHC_your')
    compare(aP, aP_cHC, rTol, aTol, 'calcHybridCoeffs', 'aP', 'cHC', 'cHC_your')
    # Save your modified arrays:
    aE_cHC_your = copy.deepcopy(aE)
    aW_cHC_your = copy.deepcopy(aW)
    aN_cHC_your = copy.deepcopy(aN)
    aS_cHC_your = copy.deepcopy(aS)
    aP_cHC_your = copy.deepcopy(aP)
if not check_calcHybridCoeffs and useModData:
    print('calcHybridCoeffs:            NOT CHECKED')
# Save reference modified arrays:
if not useModData:
    aE_cHC = copy.deepcopy(aE)
    aW_cHC = copy.deepcopy(aW)
    aN_cHC = copy.deepcopy(aN)
    aS_cHC = copy.deepcopy(aS)
    aP_cHC = copy.deepcopy(aP)
# Reset modified arrays:
aE = copy.deepcopy(aE_ref)
aW = copy.deepcopy(aW_ref)
aN = copy.deepcopy(aN_ref)
aS = copy.deepcopy(aS_ref)
aP = copy.deepcopy(aP_ref)
###############################################################################
# Depends on input values, so do not set to zero
# T*=0
cF.solveGaussSeidel(T,
                    nI, nJ, aE, aW, aN, aS, aP, Su, nLinSolIter)
if check_solveGaussSeidel and useModData:
    compare(T, T_sGS, rTol, aTol, 'solveGaussSeidel', 'T', 'sGS', 'sGS_your')
    # Save your modified arrays:
    T_sGS_your = copy.deepcopy(T)
if not check_solveGaussSeidel and useModData:
    print('solveGaussSeidel:            NOT CHECKED')
# Save reference modified arrays:
if not useModData:
    T_sGS = copy.deepcopy(T)
# Reset modified arrays:
T = copy.deepcopy(T_ref)
###############################################################################
# Depends on input values, so do not set to zero
# T*=0
# All values should be calculated, so set to zero first
P*=0
Q*=0
cF.solveTDMA(T, P, Q,
              nI, nJ, aE, aW, aN, aS, aP, Su, nLinSolIter)
if check_solveTDMA and useModData:
    compare(T, T_sTDMA, rTol, aTol, 'solveTDMA', 'T', 'sTDMA', 'sTDMA_your')
    compare(P, P_sTDMA, rTol, aTol, 'solveTDMA', 'P', 'sTDMA', 'sTDMA_your')
    compare(Q, Q_sTDMA, rTol, aTol, 'solveTDMA', 'Q', 'sTDMA', 'sTDMA_your')
    # Save your modified arrays:
    T_sTDMA_your = copy.deepcopy(T)
    P_sTDMA_your = copy.deepcopy(P)
    Q_sTDMA_your = copy.deepcopy(Q)
if not check_solveTDMA and useModData:
    print('solveTDMA:                   NOT CHECKED')
# Save reference modified arrays:
if not useModData:
    T_sTDMA = copy.deepcopy(T)
    P_sTDMA = copy.deepcopy(P)
    Q_sTDMA = copy.deepcopy(Q)
# Reset modified arrays:
T = copy.deepcopy(T_ref)
P = copy.deepcopy(P_ref)
Q = copy.deepcopy(Q_ref)
###############################################################################
# Keep internal values and set boundary values to zero
T[:,0]*=0
T[:,-1]*=0
T[0,:]*=0
T[-1,:]*=0
cF.correctBoundaries(T,
                     nI, nJ, q_wall, k, dx_PE, dx_WP, dy_PN, dy_SP,
                     u, v, nodeX, nodeY, L, H, caseID)
if check_correctBoundaries and useModData:
    compare(T, T_cB, rTol, aTol, 'correctBoundaries', 'T', 'cB', 'cB_your')
    # Save your modified arrays:
    T_cB_your = copy.deepcopy(T)
if not check_correctBoundaries and useModData:
    print('correctBoundaries:           NOT CHECKED')
# Save reference modified arrays:
if not useModData:
    T_cB = copy.deepcopy(T)
# Reset modified arrays:
T = copy.deepcopy(T_ref)
###############################################################################

#================ Save modified data ================
if not useModData:
    # Save all arrays in .npz file
    print('Saving modified data')
    np.savez('refData/Case_'+str(caseID)+'_'+'modArrays.npz',
            dx_PE_cD = dx_PE_cD,
            dx_WP_cD = dx_WP_cD,
            dy_PN_cD = dy_PN_cD,
            dy_SP_cD = dy_SP_cD,
            dx_we_cD = dx_we_cD,
            dy_sn_cD = dy_sn_cD,
            fxe_cIF = fxe_cIF,
            fxw_cIF = fxw_cIF,
            fyn_cIF = fyn_cIF,
            fys_cIF = fys_cIF,
            T_sDBCs = T_sDBCs,
            Su_cST_steady = Su_cST_steady,
            Sp_cST_steady = Sp_cST_steady,
            Su_cST_unsteady = Su_cST_unsteady,
            Sp_cST_unsteady = Sp_cST_unsteady,
            De_cD = De_cD,
            Dw_cD = Dw_cD,
            Dn_cD = Dn_cD,
            Ds_cD = Ds_cD,
            Fe_cF = Fe_cF,
            Fw_cF = Fw_cF,
            Fn_cF = Fn_cF,
            Fs_cF = Fs_cF,
            aE_cHC = aE_cHC,
            aW_cHC = aW_cHC,
            aN_cHC = aN_cHC,
            aS_cHC = aS_cHC,
            aP_cHC = aP_cHC,
            T_sGS = T_sGS,
            T_sTDMA = T_sTDMA,
            P_sTDMA = P_sTDMA,
            Q_sTDMA = Q_sTDMA,
            T_cB = T_cB)
#%%