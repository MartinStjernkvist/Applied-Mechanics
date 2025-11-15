#%%
# This code can be used to test your functions.
#
# The code requires that:
# * The folder with refData is in the same path as this file
# * The file codeFunctions_template.py is in the same path as this file
#
# Only do changes above "DON'T CHANGE ANYTHING BELOW!".
# Change case-specifics under "Case inputs".
# You may disable some checks under "Functions to check (True / False)",
# but make sure to check all functions at the end.
# You can change the tolerances for the comparisons, if necessary.
#
# The code compares the output of your functions with the output of the
# functions of a reference code, for the input that is given through refData.
# It reports "OK" in the Console if your output is the same as for the
# reference code. If the output differs more than the tolerances, it reports
# "NOT OK", and a few lines saying how much the output differs.
#
# If you get "NOT OK", you can also compare the numbers in the arrays to see
# if it is only a few values or the entire array that differes. The reference
# arrays end with _ref. If there is also an array with some other ending,
# e.g. _cC, that is the array that is compared with. Those are used in cases
# where the array is modified to some other values than the reference values
# when calling the function.
#
# Note that for some functions you MUST get the same output, but for some
# functions you may just have implemented in a slightly different way that
# may also be ok.
# Knowing which of your functions differ from the reference code may help
# you find and correct errors.
#

# Clear all variables when running entire code:
from IPython import get_ipython
get_ipython().run_line_magic('reset', '-sf')
import numpy as np
import copy
import Martin_codeFunctions as cF

# Case inputs:
caseID  = 1
L = 1
H = 0.5
h = 1000
T_inf = 35

# Functions to check (True / False):
check_calcDistances = True
check_calcInterpolationFactors = True
check_setDirichletBCs = True
check_updateConductivityArrays = True
check_updateSourceTerms = True
check_calcCoeffs = True
check_solveGaussSeidel = True
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
k_ref = refData['k']
k_e_ref = refData['k_e']
k_w_ref = refData['k_w']
k_n_ref = refData['k_n']
k_s_ref = refData['k_s']

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
k = copy.deepcopy(k_ref)
k_e = copy.deepcopy(k_e_ref)
k_w = copy.deepcopy(k_w_ref)
k_n = copy.deepcopy(k_n_ref)
k_s = copy.deepcopy(k_s_ref)

if useModData:
    # Load modified arrays
    modData = np.load('refData/Case_'+str(caseID)+'_'+'modArrays.npz')
    T_sDBCs = modData['T_sDBCs']
    k_uCA = modData['k_uCA']
    k_e_uCA = modData['k_e_uCA']
    k_w_uCA = modData['k_w_uCA']
    k_n_uCA = modData['k_n_uCA']
    k_s_uCA = modData['k_s_uCA']
    Su_uST = modData['Su_uST']
    Sp_uST = modData['Sp_uST']
    aE_cC = modData['aE_cC']
    aW_cC = modData['aW_cC']
    aN_cC = modData['aN_cC']
    aS_cC = modData['aS_cC']
    aP_cC = modData['aP_cC']
    T_sGS = modData['T_sGS']
    T_cB = modData['T_cB']

# Set numerical domain size
nI, nJ = np.shape(nodeX_ref)
mI, mJ = np.shape(pointX_ref)

def compare(phi, phi_ref, rTol, aTol, fName, aName):
    fName_aName = fName + ', ' + aName
    try:
        np.testing.assert_allclose(phi, phi_ref, rtol=rTol, atol=aTol, verbose=False, err_msg=fName+': Array '+aName)
        print(fName_aName.ljust(30)+'OK')
    except AssertionError as e:
        print(fName_aName.ljust(30)+'NOT OK')
        print('    -----------------------------------------------------')
        print('    '+str(e).splitlines()[-3])
        print('    '+str(e).splitlines()[-2])
        print('    '+str(e).splitlines()[-1])
        print('    -----------------------------------------------------')

# Test all student-implemented functions:
# ###############################################################################
dx_PE*=0
dx_WP*=0
dy_PN*=0
dy_SP*=0
dx_we*=0
dy_sn*=0
cF.calcDistances(dx_PE, dx_WP, dy_PN, dy_SP, dx_we, dy_sn,
                 nI, nJ, nodeX, nodeY, pointX, pointY)
if check_calcDistances and useModData:
    compare(dx_PE, dx_PE_ref, rTol, aTol, 'calcDistances', 'dx_PE')
    compare(dx_WP, dx_WP_ref, rTol, aTol, 'calcDistances', 'dx_WP')
    compare(dy_PN, dy_PN_ref, rTol, aTol, 'calcDistances', 'dy_PN')
    compare(dy_SP, dy_SP_ref, rTol, aTol, 'calcDistances', 'dy_SP')
    compare(dx_we, dx_we_ref, rTol, aTol, 'calcDistances', 'dx_we')
    compare(dy_sn, dy_sn_ref, rTol, aTol, 'calcDistances', 'dy_sn')
if not check_calcDistances and useModData:
    print('calcDistances:               NOT CHECKED')
# Reset modifed arrays:
dx_PE = copy.deepcopy(dx_PE_ref)
dx_WP = copy.deepcopy(dx_WP_ref)
dy_PN = copy.deepcopy(dy_PN_ref)
dy_SP = copy.deepcopy(dy_SP_ref)
dx_we = copy.deepcopy(dx_we_ref)
dy_sn = copy.deepcopy(dy_sn_ref)
###############################################################################
fxe*=0
fxw*=0
fyn*=0
fys*=0
cF.calcInterpolationFactors(fxe, fxw, fyn, fys,
                            nI, nJ, dx_PE, dx_WP, dy_PN, dy_SP, dx_we, dy_sn)
if check_calcInterpolationFactors and useModData:
    compare(fxe, fxe_ref, rTol, aTol, 'calcInterpolationFactors', 'fxe')
    compare(fxw, fxw_ref, rTol, aTol, 'calcInterpolationFactors', 'fxw')
    compare(fyn, fyn_ref, rTol, aTol, 'calcInterpolationFactors', 'fyn')
    compare(fys, fys_ref, rTol, aTol, 'calcInterpolationFactors', 'fys')
if not check_calcInterpolationFactors and useModData:
    print('calcInterpolationFactors:    NOT CHECKED')
# Reset modified arrays:
fxe = copy.deepcopy(fxe_ref)
fxw = copy.deepcopy(fxw_ref)
fyn = copy.deepcopy(fyn_ref)
fys = copy.deepcopy(fys_ref)
###############################################################################
T*=0
cF.setDirichletBCs(T,
                   nI, nJ, L, H, nodeX, nodeY, caseID)
if check_setDirichletBCs and useModData:
    compare(T, T_sDBCs, rTol, aTol, 'setDirichletBCs', 'T')
if not check_setDirichletBCs and useModData:
    print('setDirichletBCs:             NOT CHECKED')
if not useModData:
    # Save modified arrays:
    T_sDBCs = copy.deepcopy(T)
# Reset modified arrays:
T = copy.deepcopy(T_ref)
###############################################################################
k*=0
k_e*=0
k_w*=0
k_n*=0
k_s*=0
cF.updateConductivityArrays(k, k_e, k_w, k_n, k_s,
                            nI, nJ, nodeX, nodeY, fxe, fxw, fyn, fys, L, H, T, caseID)
if check_updateConductivityArrays and useModData:
    compare(k  , k_uCA  , rTol, aTol, 'updateConductivityArrays', 'k')
    compare(k_e, k_e_uCA, rTol, aTol, 'updateConductivityArrays', 'k_e')
    compare(k_w, k_w_uCA, rTol, aTol, 'updateConductivityArrays', 'k_w')
    compare(k_n, k_n_uCA, rTol, aTol, 'updateConductivityArrays', 'k_n')
    compare(k_s, k_s_uCA, rTol, aTol, 'updateConductivityArrays', 'k_s')
if not check_updateConductivityArrays and useModData:
    print('updateConductivityArrays:    NOT CHECKED')
if not useModData:
    # Save modified arrays:
    k_uCA = copy.deepcopy(k)
    k_e_uCA = copy.deepcopy(k_e)
    k_w_uCA = copy.deepcopy(k_w)
    k_n_uCA = copy.deepcopy(k_n)
    k_s_uCA = copy.deepcopy(k_s)
# Reset modified arrays:
k = copy.deepcopy(k_ref)
k_e = copy.deepcopy(k_e_ref)
k_w = copy.deepcopy(k_w_ref)
k_n = copy.deepcopy(k_n_ref)
k_s = copy.deepcopy(k_s_ref)
###############################################################################
Su*=0
Sp*=0
cF.updateSourceTerms(Su, Sp,
                     nI, nJ, dx_we, dy_sn, dx_WP, dx_PE, dy_SP, dy_PN,
                     T, k_w, k_e, k_s, k_n, h, T_inf, caseID)
if check_updateSourceTerms and useModData:
    compare(Su, Su_uST, rTol, aTol, 'updateSourceTerms', 'Su')
    compare(Sp, Sp_uST, rTol, aTol, 'updateSourceTerms', 'Sp')
if not check_updateSourceTerms and useModData:
    print('updateSourceTerms:           NOT CHECKED')
if not useModData:
    # Save modified arrays:
    Su_uST = copy.deepcopy(Su)
    Sp_uST = copy.deepcopy(Sp)
# Reset modified arrays:
Su = copy.deepcopy(Su_ref)
Sp = copy.deepcopy(Sp_ref)
###############################################################################
aE*=0
aW*=0
aN*=0
aS*=0
aP*=0
cF.calcCoeffs(aE, aW, aN, aS, aP,
              nI, nJ, k_w, k_e, k_s, k_n,
              dy_sn, dx_we, dx_WP, dx_PE, dy_SP, dy_PN, Sp, caseID)
if check_calcCoeffs and useModData:
    compare(aE, aE_cC, rTol, aTol, 'calcCoeffs', 'aE')
    compare(aW, aW_cC, rTol, aTol, 'calcCoeffs', 'aW')
    compare(aN, aN_cC, rTol, aTol, 'calcCoeffs', 'aN')
    compare(aS, aS_cC, rTol, aTol, 'calcCoeffs', 'aS')
    compare(aP, aP_cC, rTol, aTol, 'calcCoeffs', 'aP')
if not check_calcCoeffs and useModData:
    print('calcCoeffs:                  NOT CHECKED')
if not useModData:
    # Save modified arrays:
    aE_cC = copy.deepcopy(aE)
    aW_cC = copy.deepcopy(aW)
    aN_cC = copy.deepcopy(aN)
    aS_cC = copy.deepcopy(aS)
    aP_cC = copy.deepcopy(aP)
# Reset modified arrays:
aE = copy.deepcopy(aE_ref)
aW = copy.deepcopy(aW_ref)
aN = copy.deepcopy(aN_ref)
aS = copy.deepcopy(aS_ref)
aP = copy.deepcopy(aP_ref)
###############################################################################
T*=0
cF.solveGaussSeidel(T,
                    nI, nJ, aE, aW, aN, aS, aP, Su, nLinSolIter)
if check_solveGaussSeidel and useModData:
    compare(T, T_sGS, rTol, aTol, 'solveGaussSeidel', 'T')
if not check_solveGaussSeidel and useModData:
    print('solveGaussSeidel:            NOT CHECKED')
if not useModData:
    # Save modified arrays:
    T_sGS = copy.deepcopy(T)
# Reset modified arrays:
T = copy.deepcopy(T_ref)
###############################################################################
T*=0
cF.correctBoundaries(T,
                     nI, nJ, k_w, k_e, k_s, k_n,
                     dy_sn, dx_we, dx_WP, dx_PE, dy_SP, dy_PN, 
                     h, T_inf, caseID)
if check_correctBoundaries and useModData:
    compare(T, T_cB, rTol, aTol, 'correctBoundaries', 'T')
if not check_correctBoundaries and useModData:
    print('correctBoundaries:           NOT CHECKED')
if not useModData:
    # Save modified arrays:
    T_cB = copy.deepcopy(T)
# Reset modified arrays:
T = copy.deepcopy(T_ref)
###############################################################################

#================ Save modified data ================
if not useModData:
    # Save all arrays in .npz file
    print('Saving modified data')
    np.savez('refData/Case_'+str(caseID)+'_'+'modArrays.npz',
            T_sDBCs = T_sDBCs,
            k_uCA = k_uCA,
            k_e_uCA = k_e_uCA,
            k_w_uCA = k_w_uCA,
            k_n_uCA = k_n_uCA,
            k_s_uCA = k_s_uCA,
            Su_uST = Su_uST,
            Sp_uST = Sp_uST,
            aE_cC = aE_cC,
            aW_cC = aW_cC,
            aN_cC = aN_cC,
            aS_cC = aS_cC,
            aP_cC = aP_cC,
            T_sGS = T_sGS,
            T_cB = T_cB)
#%%