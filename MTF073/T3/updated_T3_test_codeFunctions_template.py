# The header of the corresponding code for Task 1 is relevant also here.
# It is not repeated. Please remind yourself if needed.

# Clear all variables when running entire code:
from reset import universal_reset
universal_reset(protect={'universal_reset'}, verbose=True)

# Packages needed
import numpy as np
import copy
import sys # For sys.exit()
import T3_codeFunctions_template as cF

#===================== Inputs =====================

# Case number (same as case in description, 1-25)
caseID = 1

# Functions to check:
check_calcDistances = True
check_calcInterpolationFactors = True
check_setInletVelocityAndFlux = True
check_initOutletFlux = True
check_correctGlobalContinuity = True
check_calcD = True
check_calcMomEqCoeffs_FOU_CD = True
check_calcMomEqCoeffs_Hybrid = True
check_calcMomEqSu = True
check_solveGaussSeidel_u = True
check_solveGaussSeidel_v = True
check_solveGaussSeidel_pp = True
check_calcRhieChow_noCorr = True
check_calcRhieChow_equiCorr = True
check_calcRhieChow_nonEquiCorr = True
check_calcPpEqCoeffs = True
check_calcPpEqSu = True
check_solveTDMA = True
check_setPressureCorrectionLevel = True
check_correctPressureCorrectionBC = True
check_correctPressure = True
check_correctPressureBCandCorners = True
check_correctVelocity = True
check_correctOutletVelocity = True
check_correctFaceFlux = True

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
# For Håkan:
# First run the solution code for all cases.
# Move refData_new to refData
# Then run this code (for solution code) for all cases with switch useModData = False to generate modified arrays.
# Then switch back to useModData = True and check that OK is reported for all case (using solution code).
# Copy the refData folder to the template folder and test with useModData = True.
# This does not work for students.
# Note: Modified arrays is for those where the output of the function will not (necessarily)
# be the same as the reference data.
# The names of the modified data arrays end with an abbreviation of the function name.

rho     = 1     # Density
mu      = 0.003 # Dynamic viscosity
nLinSolIter_pp = 10             # Number of linear solver iterations for pp-equation
nLinSolIter_uv = 3              # Number of Gauss-Seidel iterations for u/v-equations
alphaUV = 0.7   # Under-relaxation factor for u and v
alphaP  = 0.3   # Under-relaxation factor for p
grid_type = 'coarse'
pRef_i = 3
pRef_j = 3
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
aE_uv_ref = refData['aE_uv']
aW_uv_ref = refData['aW_uv']
aN_uv_ref = refData['aN_uv']
aS_uv_ref = refData['aS_uv']
aP_uv_ref = refData['aP_uv']
Su_u_ref = refData['Su_u']
Su_v_ref = refData['Su_v']
aE_pp_ref = refData['aE_pp']
aW_pp_ref = refData['aW_pp']
aN_pp_ref = refData['aN_pp']
aS_pp_ref = refData['aS_pp']
aP_pp_ref = refData['aP_pp']
Su_pp_ref = refData['Su_pp']
u_ref = refData['u']
v_ref = refData['v']
p_ref = refData['p']
pp_ref = refData['pp']
De_ref = refData['De']
Dw_ref = refData['Dw']
Dn_ref = refData['Dn']
Ds_ref = refData['Ds']
Fe_ref = refData['Fe']
Fw_ref = refData['Fw']
Fn_ref = refData['Fn']
Fs_ref = refData['Fs']
de_ref = refData['de']
dw_ref = refData['dw']
dn_ref = refData['dn']
ds_ref = refData['ds']
fxe_ref = refData['fxe']
fxw_ref = refData['fxw']
fyn_ref = refData['fyn']
fys_ref = refData['fys']
uTask2_ref = refData['uTask2']
vTask2_ref = refData['vTask2']
del refData
import gc
gc.collect()

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
aE_uv = copy.deepcopy(aE_uv_ref)
aW_uv = copy.deepcopy(aW_uv_ref)
aN_uv = copy.deepcopy(aN_uv_ref)
aS_uv = copy.deepcopy(aS_uv_ref)
aP_uv = copy.deepcopy(aP_uv_ref)
Su_u = copy.deepcopy(Su_u_ref)
Su_v = copy.deepcopy(Su_v_ref)
aE_pp = copy.deepcopy(aE_pp_ref)
aW_pp = copy.deepcopy(aW_pp_ref)
aN_pp = copy.deepcopy(aN_pp_ref)
aS_pp = copy.deepcopy(aS_pp_ref)
aP_pp = copy.deepcopy(aP_pp_ref)
Su_pp = copy.deepcopy(Su_pp_ref)
u = copy.deepcopy(u_ref)
v = copy.deepcopy(v_ref)
p = copy.deepcopy(p_ref)
pp = copy.deepcopy(pp_ref)
De = copy.deepcopy(De_ref)
Dw = copy.deepcopy(Dw_ref)
Dn = copy.deepcopy(Dn_ref)
Ds = copy.deepcopy(Ds_ref)
Fe = copy.deepcopy(Fe_ref)
Fw = copy.deepcopy(Fw_ref)
Fn = copy.deepcopy(Fn_ref)
Fs = copy.deepcopy(Fs_ref)
de = copy.deepcopy(de_ref)
dw = copy.deepcopy(dw_ref)
dn = copy.deepcopy(dn_ref)
ds = copy.deepcopy(ds_ref)
fxe = copy.deepcopy(fxe_ref)
fxw = copy.deepcopy(fxw_ref)
fyn = copy.deepcopy(fyn_ref)
fys = copy.deepcopy(fys_ref)
uTask2 = copy.deepcopy(uTask2_ref)
vTask2 = copy.deepcopy(vTask2_ref)

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
    u_sIVAF = modData['u_sIVAF']
    v_sIVAF = modData['v_sIVAF']
    Fe_sIVAF = modData['Fe_sIVAF']
    Fw_sIVAF = modData['Fw_sIVAF']
    Fn_sIVAF = modData['Fn_sIVAF']
    Fs_sIVAF = modData['Fs_sIVAF']
    Fe_iOF = modData['Fe_iOF']
    Fw_iOF = modData['Fw_iOF']
    Fn_iOF = modData['Fn_iOF']
    Fs_iOF = modData['Fs_iOF']
    Fe_cGC = modData['Fe_cGC']
    Fw_cGC = modData['Fw_cGC']
    Fn_cGC = modData['Fn_cGC']
    Fs_cGC = modData['Fs_cGC']
    De_cD = modData['De_cD']
    Dw_cD = modData['Dw_cD']
    Dn_cD = modData['Dn_cD']
    Ds_cD = modData['Ds_cD']
    aE_uv_cMEC_FOU_CD = modData['aE_uv_cMEC_FOU_CD']
    aW_uv_cMEC_FOU_CD = modData['aW_uv_cMEC_FOU_CD']
    aN_uv_cMEC_FOU_CD = modData['aN_uv_cMEC_FOU_CD']
    aS_uv_cMEC_FOU_CD = modData['aS_uv_cMEC_FOU_CD']
    aP_uv_cMEC_FOU_CD = modData['aP_uv_cMEC_FOU_CD']
    aE_uv_cMEC_Hybrid = modData['aE_uv_cMEC_Hybrid']
    aW_uv_cMEC_Hybrid = modData['aW_uv_cMEC_Hybrid']
    aN_uv_cMEC_Hybrid = modData['aN_uv_cMEC_Hybrid']
    aS_uv_cMEC_Hybrid = modData['aS_uv_cMEC_Hybrid']
    aP_uv_cMEC_Hybrid = modData['aP_uv_cMEC_Hybrid']
    Su_u_cMES = modData['Su_u_cMES']
    Su_v_cMES = modData['Su_v_cMES']
    u_sGS = modData['u_sGS']
    v_sGS = modData['v_sGS']
    pp_sGS = modData['pp_sGS']
    Fe_cRCnC = modData['Fe_cRCnC']
    Fw_cRCnC = modData['Fw_cRCnC']
    Fn_cRCnC = modData['Fn_cRCnC']
    Fs_cRCnC = modData['Fs_cRCnC']
    Fe_cRCeC = modData['Fe_cRCeC']
    Fw_cRCeC = modData['Fw_cRCeC']
    Fn_cRCeC = modData['Fn_cRCeC']
    Fs_cRCeC = modData['Fs_cRCeC']
    Fe_cRCnEC = modData['Fe_cRCnEC']
    Fw_cRCnEC = modData['Fw_cRCnEC']
    Fn_cRCnEC = modData['Fn_cRCnEC']
    Fs_cRCnEC = modData['Fs_cRCnEC']
    aE_pp_cPEC = modData['aE_pp_cPEC']
    aW_pp_cPEC = modData['aW_pp_cPEC']
    aN_pp_cPEC = modData['aN_pp_cPEC']
    aS_pp_cPEC = modData['aS_pp_cPEC']
    aP_pp_cPEC = modData['aP_pp_cPEC']
    de_cPEC = modData['de_cPEC']
    dw_cPEC = modData['dw_cPEC']
    dn_cPEC = modData['dn_cPEC']
    ds_cPEC = modData['ds_cPEC']
    Su_pp_cPES = modData['Su_pp_cPES']
    pp_sTDMA = modData['pp_sTDMA']
    pp_sPCL = modData['pp_sPCL']
    pp_cPCBC = modData['pp_cPCBC']
    p_cP = modData['p_cP']
    p_cPBCC = modData['p_cPBCC']
    u_cV = modData['u_cV']
    v_cV = modData['v_cV']
    u_cOV = modData['u_cOV']
    v_cOV = modData['v_cOV']
    Fe_cFF = modData['Fe_cFF']
    Fw_cFF = modData['Fw_cFF']
    Fn_cFF = modData['Fn_cFF']
    Fs_cFF = modData['Fs_cFF']
    del modData
    import gc
    gc.collect()
    
# Set numerical domain size
nI, nJ = np.shape(nodeX_ref)
mI, mJ = np.shape(pointX_ref)

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
###############################################################################
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
u*=0
v*=0
Fe*=0
Fw*=0
Fn*=0
Fs*=0
cF.setInletVelocityAndFlux(u, v, Fe, Fw, Fn, Fs,
                           nI, nJ, rho, dx_we, dy_sn, nodeX, nodeY, grid_type, caseID)
if check_setInletVelocityAndFlux and useModData:
    compare(u, u_sIVAF, rTol, aTol, 'setInletVelocityAndFlux', 'u', 'sIVAF', 'sIVAF_your')
    compare(v, v_sIVAF, rTol, aTol, 'setInletVelocityAndFlux', 'v', 'sIVAF', 'sIVAF_your')
    compare(Fe, Fe_sIVAF, rTol, aTol, 'setInletVelocityAndFlux', 'Fe', 'sIVAF', 'sIVAF_your')
    compare(Fw, Fw_sIVAF, rTol, aTol, 'setInletVelocityAndFlux', 'Fw', 'sIVAF', 'sIVAF_your')
    compare(Fn, Fn_sIVAF, rTol, aTol, 'setInletVelocityAndFlux', 'Fn', 'sIVAF', 'sIVAF_your')
    compare(Fs, Fs_sIVAF, rTol, aTol, 'setInletVelocityAndFlux', 'Fs', 'sIVAF', 'sIVAF_your')
    # Save your modified arrays:
    u_sIVAF_your = copy.deepcopy(u)
    v_sIVAF_your = copy.deepcopy(v)
    Fe_sIVAF_your = copy.deepcopy(Fe)
    Fw_sIVAF_your = copy.deepcopy(Fw)
    Fn_sIVAF_your = copy.deepcopy(Fn)
    Fs_sIVAF_your = copy.deepcopy(Fs)
if not check_setInletVelocityAndFlux and useModData:
    print('setInletVelocityAndFlux:    NOT CHECKED')
# Save reference modified arrays:
if not useModData:
    u_sIVAF = copy.deepcopy(u)
    v_sIVAF = copy.deepcopy(v)
    Fe_sIVAF = copy.deepcopy(Fe)
    Fw_sIVAF = copy.deepcopy(Fw)
    Fn_sIVAF = copy.deepcopy(Fn)
    Fs_sIVAF = copy.deepcopy(Fs)
# Reset modified arrays:
u = copy.deepcopy(u_ref)
v = copy.deepcopy(v_ref)
Fe = copy.deepcopy(Fe_ref)
Fw = copy.deepcopy(Fw_ref)
Fn = copy.deepcopy(Fn_ref)
Fs = copy.deepcopy(Fs_ref)
###############################################################################
# All values should be calculated, so set to zero first to avoid that a
# function that doesn't calculate any value is marked as OK
Fe*=0
Fw*=0
Fn*=0
Fs*=0
cF.initOutletFlux(Fe, Fw, Fn, Fs,
                  nI, nJ, rho, dx_we, dy_sn, nodeX, nodeY, grid_type, caseID)
if check_initOutletFlux and useModData:
    compare(Fe, Fe_iOF, rTol, aTol, 'initOutletFlux', 'Fe', 'iOF', 'iOF_your')
    compare(Fw, Fw_iOF, rTol, aTol, 'initOutletFlux', 'Fw', 'iOF', 'iOF_your')
    compare(Fn, Fn_iOF, rTol, aTol, 'initOutletFlux', 'Fn', 'iOF', 'iOF_your')
    compare(Fs, Fs_iOF, rTol, aTol, 'initOutletFlux', 'Fs', 'iOF', 'iOF_your')
    # Save your modified arrays:
    Fe_iOF_your = copy.deepcopy(Fe)
    Fw_iOF_your = copy.deepcopy(Fw)
    Fn_iOF_your = copy.deepcopy(Fn)
    Fs_iOF_your = copy.deepcopy(Fs)
if not check_initOutletFlux and useModData:
    print('initOutletFlux:    NOT CHECKED')
# Save reference modified arrays:
if not useModData:
    Fe_iOF = copy.deepcopy(Fe)
    Fw_iOF = copy.deepcopy(Fw)
    Fn_iOF = copy.deepcopy(Fn)
    Fs_iOF = copy.deepcopy(Fs)
# Reset modified arrays:
Fe = copy.deepcopy(Fe_ref)
Fw = copy.deepcopy(Fw_ref)
Fn = copy.deepcopy(Fn_ref)
Fs = copy.deepcopy(Fs_ref)
###############################################################################
# Outlet fluxes should be corrected, so set them to 1 (out of domain) to avoid that a
# function that doesn't correct any value is marked as OK
for i in range (1,nI-1):
    if Fs[i,1] < 0:
        Fs[i,1] = -1
    if Fn[i,nJ-2] > 0:
        Fn[i,nJ-2] = 1
for j in range (1,nJ-1):
    if Fw[1,j] < 0:
        Fw[1,j] = -1
    if Fe[nI-2,j] > 0:
        Fe[nI-2,j] = 1
cF.correctGlobalContinuity(Fe, Fw, Fn, Fs,
                           nI, nJ)
if check_correctGlobalContinuity and useModData:
    compare(Fe, Fe_cGC, rTol, aTol, 'correctGlobalContinuity', 'Fe', 'cGC', 'cGC_your')
    compare(Fw, Fw_cGC, rTol, aTol, 'correctGlobalContinuity', 'Fw', 'cGC', 'cGC_your')
    compare(Fn, Fn_cGC, rTol, aTol, 'correctGlobalContinuity', 'Fn', 'cGC', 'cGC_your')
    compare(Fs, Fs_cGC, rTol, aTol, 'correctGlobalContinuity', 'Fs', 'cGC', 'cGC_your')
    # Save your modified arrays:
    Fe_cGC_your = copy.deepcopy(Fe)
    Fw_cGC_your = copy.deepcopy(Fw)
    Fn_cGC_your = copy.deepcopy(Fn)
    Fs_cGC_your = copy.deepcopy(Fs)
if not check_correctGlobalContinuity and useModData:
    print('correctGlobalContinuity:    NOT CHECKED')
# Save reference modified arrays:
if not useModData:
    Fe_cGC = copy.deepcopy(Fe)
    Fw_cGC = copy.deepcopy(Fw)
    Fn_cGC = copy.deepcopy(Fn)
    Fs_cGC = copy.deepcopy(Fs)
# Reset modified arrays:
Fe = copy.deepcopy(Fe_ref)
Fw = copy.deepcopy(Fw_ref)
Fn = copy.deepcopy(Fn_ref)
Fs = copy.deepcopy(Fs_ref)
###############################################################################
# All values should be calculated, so set to zero first to avoid that a
# function that doesn't calculate any value is marked as OK
De*=0
Dw*=0
Dn*=0
Ds*=0
cF.calcD(De, Dw, Dn, Ds,
         mu, nI, nJ, dx_PE, dx_WP, dy_PN, dy_SP, dx_we, dy_sn)
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
aE_uv*=0
aW_uv*=0
aN_uv*=0
aS_uv*=0
aP_uv*=0
cF.calcMomEqCoeffs_FOU_CD(aE_uv, aW_uv, aN_uv, aS_uv, aP_uv,
                          nI, nJ, alphaUV, De, Dw, Dn, Ds,
                          Fe, Fw, Fn, Fs)
if check_calcMomEqCoeffs_FOU_CD and useModData:
    compare(aE_uv, aE_uv_cMEC_FOU_CD, rTol, aTol, 'calcMomEqCoeffs_FOU_CD', 'aE_uv', 'cMEC_FOU_CD', 'cMEC_FOU_CD_your')
    compare(aW_uv, aW_uv_cMEC_FOU_CD, rTol, aTol, 'calcMomEqCoeffs_FOU_CD', 'aW_uv', 'cMEC_FOU_CD', 'cMEC_FOU_CD_your')
    compare(aN_uv, aN_uv_cMEC_FOU_CD, rTol, aTol, 'calcMomEqCoeffs_FOU_CD', 'aN_uv', 'cMEC_FOU_CD', 'cMEC_FOU_CD_your')
    compare(aS_uv, aS_uv_cMEC_FOU_CD, rTol, aTol, 'calcMomEqCoeffs_FOU_CD', 'aS_uv', 'cMEC_FOU_CD', 'cMEC_FOU_CD_your')
    compare(aP_uv, aP_uv_cMEC_FOU_CD, rTol, aTol, 'calcMomEqCoeffs_FOU_CD', 'aP_uv', 'cMEC_FOU_CD', 'cMEC_FOU_CD_your')
    # Save your modified arrays:
    aE_uv_cMEC_FOU_CD_your = copy.deepcopy(aE_uv)
    aW_uv_cMEC_FOU_CD_your = copy.deepcopy(aW_uv)
    aN_uv_cMEC_FOU_CD_your = copy.deepcopy(aN_uv)
    aS_uv_cMEC_FOU_CD_your = copy.deepcopy(aS_uv)
    aP_uv_cMEC_FOU_CD_your = copy.deepcopy(aP_uv)
if not check_calcMomEqCoeffs_FOU_CD and useModData:
    print('calcMomEqCoeffs_FOU_CD:      NOT CHECKED')
# Save reference modified arrays:
if not useModData:
    aE_uv_cMEC_FOU_CD = copy.deepcopy(aE_uv)
    aW_uv_cMEC_FOU_CD = copy.deepcopy(aW_uv)
    aN_uv_cMEC_FOU_CD = copy.deepcopy(aN_uv)
    aS_uv_cMEC_FOU_CD = copy.deepcopy(aS_uv)
    aP_uv_cMEC_FOU_CD = copy.deepcopy(aP_uv)
# Reset modified arrays:
aE_uv = copy.deepcopy(aE_uv_ref)
aW_uv = copy.deepcopy(aW_uv_ref)
aN_uv = copy.deepcopy(aN_uv_ref)
aS_uv = copy.deepcopy(aS_uv_ref)
aP_uv = copy.deepcopy(aP_uv_ref)
###############################################################################
# All values should be calculated, so set to zero first to avoid that a
# function that doesn't calculate any value is marked as OK
aE_uv*=0
aW_uv*=0
aN_uv*=0
aS_uv*=0
aP_uv*=0
cF.calcMomEqCoeffs_Hybrid(aE_uv, aW_uv, aN_uv, aS_uv, aP_uv,
                          nI, nJ, alphaUV, De, Dw, Dn, Ds,
                          Fe, Fw, Fn, Fs, fxe, fxw, fyn, fys)
if check_calcMomEqCoeffs_Hybrid and useModData:
    compare(aE_uv, aE_uv_cMEC_Hybrid, rTol, aTol, 'calcMomEqCoeffs_Hybrid', 'aE_uv', 'cMEC_Hybrid', 'cMEC_Hybrid_your')
    compare(aW_uv, aW_uv_cMEC_Hybrid, rTol, aTol, 'calcMomEqCoeffs_Hybrid', 'aW_uv', 'cMEC_Hybrid', 'cMEC_Hybrid_your')
    compare(aN_uv, aN_uv_cMEC_Hybrid, rTol, aTol, 'calcMomEqCoeffs_Hybrid', 'aN_uv', 'cMEC_Hybrid', 'cMEC_Hybrid_your')
    compare(aS_uv, aS_uv_cMEC_Hybrid, rTol, aTol, 'calcMomEqCoeffs_Hybrid', 'aS_uv', 'cMEC_Hybrid', 'cMEC_Hybrid_your')
    compare(aP_uv, aP_uv_cMEC_Hybrid, rTol, aTol, 'calcMomEqCoeffs_Hybrid', 'aP_uv', 'cMEC_Hybrid', 'cMEC_Hybrid_your')
    # Save your modified arrays:
    aE_uv_cMECH_your = copy.deepcopy(aE_uv)
    aW_uv_cMECH_your = copy.deepcopy(aW_uv)
    aN_uv_cMECH_your = copy.deepcopy(aN_uv)
    aS_uv_cMECH_your = copy.deepcopy(aS_uv)
    aP_uv_cMECH_your = copy.deepcopy(aP_uv)
if not check_calcMomEqCoeffs_Hybrid and useModData:
    print('calcMomEqCoeffs_Hybrid:      NOT CHECKED')
# Save reference modified arrays:
if not useModData:
    aE_uv_cMEC_Hybrid = copy.deepcopy(aE_uv)
    aW_uv_cMEC_Hybrid = copy.deepcopy(aW_uv)
    aN_uv_cMEC_Hybrid = copy.deepcopy(aN_uv)
    aS_uv_cMEC_Hybrid = copy.deepcopy(aS_uv)
    aP_uv_cMEC_Hybrid = copy.deepcopy(aP_uv)
# Reset modified arrays:
aE_uv = copy.deepcopy(aE_uv_ref)
aW_uv = copy.deepcopy(aW_uv_ref)
aN_uv = copy.deepcopy(aN_uv_ref)
aS_uv = copy.deepcopy(aS_uv_ref)
aP_uv = copy.deepcopy(aP_uv_ref)
###############################################################################
# All values should be calculated, so set to zero first to avoid that a
# function that doesn't calculate any value is marked as OK
Su_u*=0
Su_v*=0
cF.calcMomEqSu(Su_u, Su_v,
               nI, nJ, p, dx_WP, dx_PE, dy_SP, dy_PN, dx_we, dy_sn,
               alphaUV, aP_uv, u, v, fxe, fxw, fyn, fys)
if check_calcMomEqSu and useModData:
    compare(Su_u, Su_u_cMES, rTol, aTol, 'calcMomEqSu', 'Su_u', 'cD', 'cMES_your')
    compare(Su_v, Su_v_cMES, rTol, aTol, 'calcMomEqSu', 'Su_v', 'cD', 'cMES_your')
    # Save your modified arrays:
    Su_u_cMES_your = copy.deepcopy(Su_u)
    Su_v_cMES_your = copy.deepcopy(Su_v)
if not check_calcMomEqSu and useModData:
    print('calcMomEqSu:                 NOT CHECKED')
# Save reference modified arrays:
if not useModData:
    Su_u_cMES = copy.deepcopy(Su_u)
    Su_v_cMES = copy.deepcopy(Su_v)
# Reset modified arrays:
Su_u = copy.deepcopy(Su_u_ref)
Su_v = copy.deepcopy(Su_v_ref)
###############################################################################
# Depends on input values, so do not set to zero
# u*=0
cF.solveGaussSeidel(u,
                    nI, nJ, aE_uv, aW_uv, aN_uv, aS_uv, aP_uv, Su_u, nLinSolIter_uv)
if check_solveGaussSeidel_u and useModData:
    compare(u, u_sGS, rTol, aTol, 'solveGaussSeidel', 'u', 'sGS', 'sGS_your')
    # Save your modified arrays:
    u_sGS_your = copy.deepcopy(u)
if not check_solveGaussSeidel_u and useModData:
    print('solveGaussSeidel_u:          NOT CHECKED')
# Save reference modified arrays:
if not useModData:
    u_sGS = copy.deepcopy(u)
# Reset modified arrays:
u = copy.deepcopy(u_ref)
###############################################################################
# Depends on input values, so do not set to zero
# v*=0
cF.solveGaussSeidel(v,
                    nI, nJ, aE_uv, aW_uv, aN_uv, aS_uv, aP_uv, Su_v, nLinSolIter_uv)
if check_solveGaussSeidel_v and useModData:
    compare(v, v_sGS, rTol, aTol, 'solveGaussSeidel', 'v', 'sGS', 'sGS_your')
    # Save your modified arrays:
    v_sGS_your = copy.deepcopy(v)
if not check_solveGaussSeidel_v and useModData:
    print('solveGaussSeidel_v:          NOT CHECKED')
# Save reference modified arrays:
if not useModData:
    v_sGS = copy.deepcopy(v)
# Reset modified arrays:
v = copy.deepcopy(v_ref)
###############################################################################
# Depends on input values, so do not set to zero
# pp*=0
cF.solveGaussSeidel(pp,
                    nI, nJ, aE_pp, aW_pp, aN_pp, aS_pp, aP_pp, Su_pp,
                    nLinSolIter_pp)
if check_solveGaussSeidel_pp and useModData:
    compare(pp, pp_sGS, rTol, aTol, 'solveGaussSeidel', 'pp', 'sGS', 'sGS_your')
    # Save your modified arrays:
    pp_sGS_your = copy.deepcopy(pp)
if not check_solveGaussSeidel_pp and useModData:
    print('solveGaussSeidel_pp:         NOT CHECKED')
# Save reference modified arrays:
if not useModData:
    pp_sGS = copy.deepcopy(pp)
# Reset modified arrays:
pp = copy.deepcopy(pp_ref)
###############################################################################
# All values should be calculated, so set to zero first to avoid that a
# function that doesn't calculate any value is marked as OK
Fe*=0
Fw*=0
Fn*=0
Fs*=0
cF.calcRhieChow_noCorr(Fe, Fw, Fn, Fs,
                       nI, nJ, rho, u, v,
                       dx_we, dy_sn, fxe, fxw, fyn, fys)
if check_calcRhieChow_noCorr and useModData:
    compare(Fe, Fe_cRCnC, rTol, aTol, 'calcRhieChow_noCorr', 'Fe', 'cRCnC', 'cRCnC_your')
    compare(Fw, Fw_cRCnC, rTol, aTol, 'calcRhieChow_noCorr', 'Fw', 'cRCnC', 'cRCnC_your')
    compare(Fn, Fn_cRCnC, rTol, aTol, 'calcRhieChow_noCorr', 'Fn', 'cRCnC', 'cRCnC_your')
    compare(Fs, Fs_cRCnC, rTol, aTol, 'calcRhieChow_noCorr', 'Fs', 'cRCnC', 'cRCnC_your')
    # Save your modified arrays:
    Fe_cRCnC_your = copy.deepcopy(Fe)
    Fw_cRCnC_your = copy.deepcopy(Fw)
    Fn_cRCnC_your = copy.deepcopy(Fn)
    Fs_cRCnC_your = copy.deepcopy(Fs)
if not check_calcRhieChow_noCorr and useModData:
    print('calcRhieChow_noCorr:         NOT CHECKED')
# Save reference modified arrays:
if not useModData:
    Fe_cRCnC = copy.deepcopy(Fe)
    Fw_cRCnC = copy.deepcopy(Fw)
    Fn_cRCnC = copy.deepcopy(Fn)
    Fs_cRCnC = copy.deepcopy(Fs)
# Reset modified arrays:
Fe = copy.deepcopy(Fe_ref)
Fw = copy.deepcopy(Fw_ref)
Fn = copy.deepcopy(Fn_ref)
Fs = copy.deepcopy(Fs_ref)
###############################################################################
# All values should be calculated, so set to zero first to avoid that a
# function that doesn't calculate any value is marked as OK
Fe*=0
Fw*=0
Fn*=0
Fs*=0
cF.calcRhieChow_equiCorr(Fe, Fw, Fn, Fs,
                         nI, nJ, rho, u, v,
                         dx_we, dy_sn, fxe, fxw, fyn, fys, aP_uv, p)
if check_calcRhieChow_equiCorr and useModData:
    compare(Fe, Fe_cRCeC, rTol, aTol, 'calcRhieChow_equiCorr', 'Fe', 'cRCeC', 'cRCeC_your')
    compare(Fw, Fw_cRCeC, rTol, aTol, 'calcRhieChow_equiCorr', 'Fw', 'cRCeC', 'cRCeC_your')
    compare(Fn, Fn_cRCeC, rTol, aTol, 'calcRhieChow_equiCorr', 'Fn', 'cRCeC', 'cRCeC_your')
    compare(Fs, Fs_cRCeC, rTol, aTol, 'calcRhieChow_equiCorr', 'Fs', 'cRCeC', 'cRCeC_your')
    # Save your modified arrays:
    Fe_cRCeC_your = copy.deepcopy(Fe)
    Fw_cRCeC_your = copy.deepcopy(Fw)
    Fn_cRCeC_your = copy.deepcopy(Fn)
    Fs_cRCeC_your = copy.deepcopy(Fs)
if not check_calcRhieChow_equiCorr and useModData:
    print('calcRhieChow_equiCorr:       NOT CHECKED')
# Save reference modified arrays:
if not useModData:
    Fe_cRCeC = copy.deepcopy(Fe)
    Fw_cRCeC = copy.deepcopy(Fw)
    Fn_cRCeC = copy.deepcopy(Fn)
    Fs_cRCeC = copy.deepcopy(Fs)
# Reset modified arrays:
Fe = copy.deepcopy(Fe_ref)
Fw = copy.deepcopy(Fw_ref)
Fn = copy.deepcopy(Fn_ref)
Fs = copy.deepcopy(Fs_ref)
###############################################################################
# All values should be calculated, so set to zero first to avoid that a
# function that doesn't calculate any value is marked as OK
Fe*=0
Fw*=0
Fn*=0
Fs*=0
cF.calcRhieChow_nonEquiCorr(Fe, Fw, Fn, Fs,
                            nI, nJ, rho, u, v,
                            dx_we, dy_sn, fxe, fxw, fyn, fys, aP_uv, p,
                            dx_WP, dx_PE, dy_SP, dy_PN)
if check_calcRhieChow_nonEquiCorr and useModData:
    compare(Fe, Fe_cRCnEC, rTol, aTol, 'calcRhieChow_nonEquiCorr', 'Fe', 'cRCnEC', 'cRCnEC_your')
    compare(Fw, Fw_cRCnEC, rTol, aTol, 'calcRhieChow_nonEquiCorr', 'Fw', 'cRCnEC', 'cRCnEC_your')
    compare(Fn, Fn_cRCnEC, rTol, aTol, 'calcRhieChow_nonEquiCorr', 'Fn', 'cRCnEC', 'cRCnEC_your')
    compare(Fs, Fs_cRCnEC, rTol, aTol, 'calcRhieChow_nonEquiCorr', 'Fs', 'cRCnEC', 'cRCnEC_your')
    # Save your modified arrays:
    Fe_cRCnEC_your = copy.deepcopy(Fe)
    Fw_cRCnEC_your = copy.deepcopy(Fw)
    Fn_cRCnEC_your = copy.deepcopy(Fn)
    Fs_cRCnEC_your = copy.deepcopy(Fs)
if not check_calcRhieChow_nonEquiCorr and useModData:
    print('calcRhieChow_nonEquiCorr:    NOT CHECKED')
# Save reference modified arrays:
if not useModData:
    Fe_cRCnEC = copy.deepcopy(Fe)
    Fw_cRCnEC = copy.deepcopy(Fw)
    Fn_cRCnEC = copy.deepcopy(Fn)
    Fs_cRCnEC = copy.deepcopy(Fs)
# Reset modified arrays:
Fe = copy.deepcopy(Fe_ref)
Fw = copy.deepcopy(Fw_ref)
Fn = copy.deepcopy(Fn_ref)
Fs = copy.deepcopy(Fs_ref)
###############################################################################
# All values should be calculated, so set to 1 first to avoid that a
# function that doesn't calculate any value is marked as OK
# (some should be set to zero by function, so we set to 1 here)
aE_pp*=0
aW_pp*=0
aN_pp*=0
aS_pp*=0
aP_pp*=0
de*=0
dw*=0
dn*=0
ds*=0
aE_pp+=1.0
aW_pp+=1.0
aN_pp+=1.0
aS_pp+=1.0
aP_pp+=1.0
de+=1.0
dw+=1.0
dn+=1.0
ds+=1.0
cF.calcPpEqCoeffs(aE_pp, aW_pp, aN_pp, aS_pp, aP_pp, de, dw, dn, ds,
                  nI, nJ, rho, dx_we, dy_sn, fxe, fxw, fyn, fys, aP_uv)
if check_calcPpEqCoeffs and useModData:
    compare(aE_pp, aE_pp_cPEC, rTol, aTol, 'calcPpEqCoeffs', 'aE_pp', 'cPEC', 'cPEC_your')
    compare(aW_pp, aW_pp_cPEC, rTol, aTol, 'calcPpEqCoeffs', 'aW_pp', 'cPEC', 'cPEC_your')
    compare(aN_pp, aN_pp_cPEC, rTol, aTol, 'calcPpEqCoeffs', 'aN_pp', 'cPEC', 'cPEC_your')
    compare(aS_pp, aS_pp_cPEC, rTol, aTol, 'calcPpEqCoeffs', 'aS_pp', 'cPEC', 'cPEC_your')
    compare(aP_pp, aP_pp_cPEC, rTol, aTol, 'calcPpEqCoeffs', 'aP_pp', 'cPEC', 'cPEC_your')
    compare(de, de_cPEC, rTol, aTol, 'calcPpEqCoeffs', 'de', 'cPEC', 'cPEC_your')
    compare(dw, dw_cPEC, rTol, aTol, 'calcPpEqCoeffs', 'dw', 'cPEC', 'cPEC_your')
    compare(dn, dn_cPEC, rTol, aTol, 'calcPpEqCoeffs', 'dn', 'cPEC', 'cPEC_your')
    compare(ds, ds_cPEC, rTol, aTol, 'calcPpEqCoeffs', 'ds', 'cPEC', 'cPEC_your')
    # Save your modified arrays:
    aE_pp_cPEC_your = copy.deepcopy(aE_pp)
    aW_pp_cPEC_your = copy.deepcopy(aW_pp)
    aN_pp_cPEC_your = copy.deepcopy(aN_pp)
    aS_pp_cPEC_your = copy.deepcopy(aS_pp)
    aP_pp_cPEC_your = copy.deepcopy(aP_pp)
    de_cPEC_your = copy.deepcopy(de)
    dw_cPEC_your = copy.deepcopy(dw)
    dn_cPEC_your = copy.deepcopy(dn)
    ds_cPEC_your = copy.deepcopy(ds)
if not check_calcPpEqCoeffs and useModData:
    print('calcPpEqCoeffs:              NOT CHECKED')
# Save reference modified arrays:
if not useModData:
    aE_pp_cPEC = copy.deepcopy(aE_pp)
    aW_pp_cPEC = copy.deepcopy(aW_pp)
    aN_pp_cPEC = copy.deepcopy(aN_pp)
    aS_pp_cPEC = copy.deepcopy(aS_pp)
    aP_pp_cPEC = copy.deepcopy(aP_pp)
    de_cPEC = copy.deepcopy(de)
    dw_cPEC = copy.deepcopy(dw)
    dn_cPEC = copy.deepcopy(dn)
    ds_cPEC = copy.deepcopy(ds)
# Reset modified arrays:
aE_pp = copy.deepcopy(aE_pp_ref)
aW_pp = copy.deepcopy(aW_pp_ref)
aN_pp = copy.deepcopy(aN_pp_ref)
aS_pp = copy.deepcopy(aS_pp_ref)
aP_pp = copy.deepcopy(aP_pp_ref)
de = copy.deepcopy(de_ref)
dw = copy.deepcopy(dw_ref)
dn = copy.deepcopy(dn_ref)
ds = copy.deepcopy(ds_ref)
###############################################################################
# All values should be calculated, so set to zero first to avoid that a
# function that doesn't calculate any value is marked as OK
Su_pp*=0
cF.calcPpEqSu(Su_pp,
              nI, nJ, Fe, Fw, Fn, Fs)
if check_calcPpEqSu and useModData:
    compare(Su_pp, Su_pp_cPES, rTol, aTol, 'calcPpEqSu', 'Su_pp', 'cPES', 'cPES_your')
    # Save your modified arrays:
    Su_pp_cPES_your = copy.deepcopy(Su_pp)
if not check_calcPpEqSu and useModData:
    print('calcPpEqSu:                  NOT CHECKED')
# Save reference modified arrays:
if not useModData:
    Su_pp_cPES = copy.deepcopy(Su_pp)
# Reset modified arrays:
Su_pp = copy.deepcopy(Su_pp_ref)
###############################################################################
# Depends on input values, so do not set to zero
# pp*=0
cF.solveTDMA(pp,
             nI, nJ, aE_pp, aW_pp, aN_pp, aS_pp, aP_pp, Su_pp,
             nLinSolIter_pp)
if check_solveTDMA and useModData:
    compare(pp, pp_sTDMA, rTol, aTol, 'solveTDMA', 'pp', 'sTDMA', 'sTDMA_your')
    # Save your modified arrays:
    pp_sTDMA_your = copy.deepcopy(pp)
if not check_solveTDMA and useModData:
    print('solveTDMA:                   NOT CHECKED')
# Save reference modified arrays:
if not useModData:
    pp_sTDMA = copy.deepcopy(pp)
# Reset modified arrays:
pp = copy.deepcopy(pp_ref)
###############################################################################
# The level should be changed, so change the level first to avoid that a
# function that doesn't change the level is marked as OK
pp[:,:] += 1.0
cF.setPressureCorrectionLevel(pp,
                              nI, nJ, pRef_i, pRef_j)
if check_setPressureCorrectionLevel and useModData:
    compare(pp, pp_sPCL, rTol, aTol, 'setPressureCorrectionLevel', 'pp', 'sPCL', 'sPCL_your')
    # Save your modified arrays:
    pp_sPCL_your = copy.deepcopy(pp)
if not check_setPressureCorrectionLevel and useModData:
    print('setPressureCorrectionLevel:  NOT CHECKED')
# Save reference modified arrays:
if not useModData:
    pp_sPCL = copy.deepcopy(pp)
# Reset modified arrays:
pp = copy.deepcopy(pp_ref)
###############################################################################
# Keep internal values and set boundary values to zero
pp[:,0]*=0
pp[:,-1]*=0
pp[0,:]*=0
pp[-1,:]*=0
cF.correctPressureCorrectionBC(pp,
                               nI, nJ)
if check_correctPressureCorrectionBC and useModData:
    compare(pp, pp_cPCBC, rTol, aTol, 'correctPressureCorrectionBC', 'pp', 'cD', 'cPCBC_your')
    # Save your modified arrays:
    pp_cPCBC_your = copy.deepcopy(pp)
if not check_correctPressureCorrectionBC and useModData:
    print('correctPressureCorrectionBC: NOT CHECKED')
# Save reference modified arrays:
if not useModData:
    pp_cPCBC = copy.deepcopy(pp)
# Reset modified arrays:
pp = copy.deepcopy(pp_ref)
###############################################################################
# Set p to pp_ref, so that the correction gives 2*pp_ref in  nodes where
# the pressure should be corrected. Also catches the use of = instead of +=
p = copy.deepcopy(pp_ref)
cF.correctPressure(p,
                   nI, nJ, alphaP, pp)
if check_correctPressure and useModData:
    compare(p[1:nI-1,1:nJ-1], p_cP[1:nI-1,1:nJ-1], rTol, aTol, 'correctPressure', 'p', 'cP', 'cP_your')
    # Save your modified arrays:
    p_cP_your = copy.deepcopy(p)
if not check_correctPressure and useModData:
    print('correctPressure:             NOT CHECKED')
# Save reference modified arrays:
if not useModData:
    p_cP = copy.deepcopy(p)
# Reset modified arrays:
p = copy.deepcopy(p_ref)
###############################################################################
# Keep internal values and set boundary values to zero
p[:,0]*=0
p[:,-1]*=0
p[0,:]*=0
p[-1,:]*=0
cF.correctPressureBCandCorners(p,
                               nI, nJ, dx_PE, dx_WP, dy_PN, dy_SP)
if check_correctPressureBCandCorners and useModData:
    compare(p, p_cPBCC, rTol, aTol, 'correctPressureBCandCorners', 'p', 'cPBCC', 'cPBCC_your')
    # Save your modified arrays:
    p_cPBCC_your = copy.deepcopy(p)
if not check_correctPressureBCandCorners and useModData:
    print('correctPressureBCandCorners: NOT CHECKED')
# Save reference modified arrays:
if not useModData:
    p_cPBCC = copy.deepcopy(p)
# Reset modified arrays:
p = copy.deepcopy(p_ref)
###############################################################################
# Set to small value to not cause truncation of correction
# Do not set to zero, which will not catch = instead of +=
u=np.ones((nI,nJ))*1e-6
v=np.ones((nI,nJ))*1e-6
cF.correctVelocity(u, v,
                   nI, nJ, fxe, fxw, fyn, fys, pp, dy_sn, dx_we, aP_uv)
if check_correctVelocity and useModData:
    compare(u, u_cV, rTol, aTol, 'correctVelocity', 'u', 'cV', 'cV_your')
    compare(v, v_cV, rTol, aTol, 'correctVelocity', 'v', 'cV', 'cV_your')
    # Save your modified arrays:
    u_cV_your = copy.deepcopy(u)
    v_cV_your = copy.deepcopy(v)
if not check_correctVelocity and useModData:
    print('correctVelocity:             NOT CHECKED')
# Save reference modified arrays:
if not useModData:
    u_cV = copy.deepcopy(u)
    v_cV = copy.deepcopy(v)
# Reset modified arrays:
u = copy.deepcopy(u_ref)
v = copy.deepcopy(v_ref)
###############################################################################
# Outlet velocities should be set, so set them to zero to avoid that a
# function that doesn't correct any value is marked as OK
for i in range (1,nI-1):
    if Fs[i,1] < 0:
        u[i,1] = 0
        v[i,1] = 0
    if Fn[i,nJ-2] > 0:
        u[i,nJ-2] = 0
        v[i,nJ-2] = 0
for j in range (1,nJ-1):
    if Fw[1,j] < 0:
        u[1,j] = 0
        v[1,j] = 0
    if Fe[nI-2,j] > 0:
        u[nI-2,j] = 0
        v[nI-2,j] = 0
cF.correctOutletVelocity(u, v,
                         nI, nJ, rho, dx_we, dy_sn, nodeX, nodeY, grid_type, caseID)
if check_correctOutletVelocity and useModData:
    compare(u, u_cOV, rTol, aTol, 'correctOutletVelocity', 'u', 'cOV', 'cOV_your')
    compare(v, v_cOV, rTol, aTol, 'correctOutletVelocity', 'v', 'cOV', 'cOV_your')
    # Save your modified arrays:
    u_cOV_your = copy.deepcopy(u)
    v_cOV_your = copy.deepcopy(v)
if not check_correctOutletVelocity and useModData:
    print('correctOutletVelocity:             NOT CHECKED')
# Save reference modified arrays:
if not useModData:
    u_cOV = copy.deepcopy(u)
    v_cOV = copy.deepcopy(v)
# Reset modified arrays:
u = copy.deepcopy(u_ref)
v = copy.deepcopy(v_ref)
###############################################################################
# Set to small value to not cause truncation of correction
# Do not set to zero, which will not catch = instead of +=
Fe=np.ones((nI,nJ))*1e-6
Fw=np.ones((nI,nJ))*1e-6
Fn=np.ones((nI,nJ))*1e-6
Fs=np.ones((nI,nJ))*1e-6
cF.correctFaceFlux(Fe, Fw, Fn, Fs,
                   nI, nJ, rho, dy_sn, dx_we, de, dw, dn, ds, pp)
if check_correctFaceFlux and useModData:
    compare(Fe, Fe_cFF, rTol, aTol, 'correctFaceFlux', 'Fe', 'cFF', 'cFF_your')
    compare(Fw, Fw_cFF, rTol, aTol, 'correctFaceFlux', 'Fw', 'cFF', 'cFF_your')
    compare(Fn, Fn_cFF, rTol, aTol, 'correctFaceFlux', 'Fn', 'cFF', 'cFF_your')
    compare(Fs, Fs_cFF, rTol, aTol, 'correctFaceFlux', 'Fs', 'cFF', 'cFF_your')
    # Save your modified arrays:
    Fe_cFF_your = copy.deepcopy(Fe)
    Fw_cFF_your = copy.deepcopy(Fw)
    Fn_cFF_your = copy.deepcopy(Fn)
    Fs_cFF_your = copy.deepcopy(Fs)
if not check_correctFaceFlux and useModData:
    print('correctFaceFlux:           NOT CHECKED')
# Save reference modified arrays:
if not useModData:
    Fe_cFF = copy.deepcopy(Fe)
    Fw_cFF = copy.deepcopy(Fw)
    Fn_cFF = copy.deepcopy(Fn)
    Fs_cFF = copy.deepcopy(Fs)
# Reset modified arrays:
Fe = copy.deepcopy(Fe_ref)
Fw = copy.deepcopy(Fw_ref)
Fn = copy.deepcopy(Fn_ref)
Fs = copy.deepcopy(Fs_ref)
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
            u_sIVAF = u_sIVAF,
            v_sIVAF = v_sIVAF,
            Fe_sIVAF = Fe_sIVAF,
            Fw_sIVAF = Fw_sIVAF,
            Fn_sIVAF = Fn_sIVAF,
            Fs_sIVAF = Fs_sIVAF,
            Fe_iOF = Fe_iOF,
            Fw_iOF = Fw_iOF,
            Fn_iOF = Fn_iOF,
            Fs_iOF = Fs_iOF,
            Fe_cGC = Fe_cGC,
            Fw_cGC = Fw_cGC,
            Fn_cGC = Fn_cGC,
            Fs_cGC = Fs_cGC,
            De_cD = De_cD,
            Dw_cD = Dw_cD,
            Dn_cD = Dn_cD,
            Ds_cD = Ds_cD,
            aE_uv_cMEC_FOU_CD = aE_uv_cMEC_FOU_CD,
            aW_uv_cMEC_FOU_CD = aW_uv_cMEC_FOU_CD,
            aN_uv_cMEC_FOU_CD = aN_uv_cMEC_FOU_CD,
            aS_uv_cMEC_FOU_CD = aS_uv_cMEC_FOU_CD,
            aP_uv_cMEC_FOU_CD = aP_uv_cMEC_FOU_CD,
            aE_uv_cMEC_Hybrid = aE_uv_cMEC_Hybrid,
            aW_uv_cMEC_Hybrid = aW_uv_cMEC_Hybrid,
            aN_uv_cMEC_Hybrid = aN_uv_cMEC_Hybrid,
            aS_uv_cMEC_Hybrid = aS_uv_cMEC_Hybrid,
            aP_uv_cMEC_Hybrid = aP_uv_cMEC_Hybrid,
            Su_u_cMES = Su_u_cMES,
            Su_v_cMES = Su_v_cMES,
            u_sGS = u_sGS,
            v_sGS = v_sGS,
            pp_sGS = pp_sGS,
            Fe_cRCnC = Fe_cRCnC,
            Fw_cRCnC = Fw_cRCnC,
            Fn_cRCnC = Fn_cRCnC,
            Fs_cRCnC = Fs_cRCnC,
            Fe_cRCeC = Fe_cRCeC,
            Fw_cRCeC = Fw_cRCeC,
            Fn_cRCeC = Fn_cRCeC,
            Fs_cRCeC = Fs_cRCeC,
            Fe_cRCnEC = Fe_cRCnEC,
            Fw_cRCnEC = Fw_cRCnEC,
            Fn_cRCnEC = Fn_cRCnEC,
            Fs_cRCnEC = Fs_cRCnEC,
            aE_pp_cPEC = aE_pp_cPEC,
            aW_pp_cPEC = aW_pp_cPEC,
            aN_pp_cPEC = aN_pp_cPEC,
            aS_pp_cPEC = aS_pp_cPEC,
            aP_pp_cPEC = aP_pp_cPEC,
            de_cPEC = de_cPEC,
            dw_cPEC = dw_cPEC,
            dn_cPEC = dn_cPEC,
            ds_cPEC = ds_cPEC,
            Su_pp_cPES = Su_pp_cPES,
            pp_sTDMA = pp_sTDMA,
            pp_sPCL = pp_sPCL,
            pp_cPCBC = pp_cPCBC,
            p_cP = p_cP,
            p_cPBCC = p_cPBCC,
            u_cV = u_cV,
            v_cV = v_cV,
            u_cOV = u_cOV,
            v_cOV = v_cOV,
            Fe_cFF = Fe_cFF,
            Fw_cFF = Fw_cFF,
            Fn_cFF = Fn_cFF,
            Fs_cFF = Fs_cFF)