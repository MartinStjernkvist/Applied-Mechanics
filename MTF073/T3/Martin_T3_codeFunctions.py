#%%
# The header of the corresponding code for Task 1 is relevant also here.
# It is not repeated. Please remind yourself if needed.

# Packages needed
import numpy as np
import matplotlib.pyplot as plt
# Set default font size in plots:
plt.rcParams.update({'font.size': 12})
import sys # For sys.exit()
import os # For saving plots

def calcNodePositions(nodeX, nodeY,
                      nI, nJ, pointX, pointY):
    """
    ################################
    # DO NOT CHANGE ANYTHING HERE! #
    ################################
    """
    # Calculates node coordinates.
    # Only changes arrays in first row of argument list!
    # Internal nodes:
    for i in range(0, nI):
        for j in range(0, nJ):
            if i > 0 and i < nI-1:
                nodeX[i,j] = 0.5*(pointX[i,0] + pointX[i-1,0])
            if j > 0 and j < nJ-1:
                nodeY[i,j] = 0.5*(pointY[0,j] + pointY[0,j-1])
    # Boundary nodes:
    nodeX[0,:]  = pointX[0,0]  # Note: corner points needed for contour plot
    nodeY[:,0]  = pointY[0,0]  # Note: corner points needed for contour plot
    nodeX[-1,:] = pointX[-1,0] # Note: corner points needed for contour plot
    nodeY[:,-1] = pointY[0,-1] # Note: corner points needed for contour plot
    
def calcDistances(dx_PE, dx_WP, dy_PN, dy_SP, dx_we, dy_sn,
                  nI, nJ, nodeX, nodeY, pointX, pointY):
    # Calculate distances in first line of argument list.
    # Only change arrays in first row of argument list!
    # Keep 'nan' where values are not needed!
    # ADD CODE HERE
    # for i in range(1, nI-1):
    #     for j in range(1, nJ-1):
    #         dx_PE[i,j] = 0 # ADD CODE HERE
    #         dx_WP[i,j] = 0 # ADD CODE HERE
    #         dy_PN[i,j] = 0 # ADD CODE HERE
    #         dy_SP[i,j] = 0 # ADD CODE HERE
    #         dx_we[i,j] = 0 # ADD CODE HERE
    #         dy_sn[i,j] = 0 # ADD CODE HERE
    """
    --------------------------------
    # ADDED CODE - SOLVED
    --------------------------------
    """
    for i in range(1, nI-1):
        for j in range(1, nJ-1):
            dx_PE[i, j] = nodeX[i + 1, j] - nodeX[i, j] # P->E
            dx_WP[i, j] = nodeX[i, j] - nodeX[i - 1, j] # W->P
            
            dy_PN[i, j] = nodeY[i, j + 1] - nodeY[i, j] # P->N
            dy_SP[i, j] = nodeY[i, j] - nodeY[i, j - 1] # S->P
            
            dx_we[i, j] = pointX[i, 0] - pointX[i - 1, 0] # Length W-E
            dy_sn[i, j] = pointY[0, j] - pointY[0, j - 1] # Length S-N
    
    
def calcInterpolationFactors(fxe, fxw, fyn, fys,
                             nI, nJ, dx_PE, dx_WP, dy_PN, dy_SP, dx_we, dy_sn):
    # Calculate interpolation factors in first row of argument list.
    # Only change arrays in first row of argument list!
    # Keep 'nan' where values are not needed!
    # ADD CODE HERE
    # for i in range(1, nI-1):
    #     for j in range(1, nJ-1):
    #         fxe[i,j] = 0 # ADD CODE HERE
    #         fxw[i,j] = 0 # ADD CODE HERE
    #         fyn[i,j] = 0 # ADD CODE HERE
    #         fys[i,j] = 0 # ADD CODE HERE 
    
    """
    --------------------------------
    # ADDED CODE - SOLVED
    --------------------------------
    """
    for i in range(1, nI-1):
        for j in range(1, nJ-1):
            fxe[i, j] = 0.5 * dx_we[i, j] / dx_PE[i, j] # P->face e / P->E
            fxw[i, j] = 0.5 * dx_we[i, j] / dx_WP[i, j] # face w->P  / W->P
            fyn[i, j] = 0.5 * dy_sn[i, j] / dy_PN[i, j] # P->face n / P->N
            fys[i, j] = 0.5 * dy_sn[i, j] / dy_SP[i, j] # face s->P  / S->P
    
    
def initArrays(u, v, p, Fe, Fw, Fn, Fs):
    """
    ################################
    # DO NOT CHANGE ANYTHING HERE! #
    ################################
    """
    # Sets initial default velocity, pressure and face fluxes
    # The velocity and face flux is later kept to zero at walls 
    # (also in corner nodes, for contour plot).
    u[:,:] = 0
    v[:,:] = 0
    p[:,:] = 0
    # Sets initial default face fluxes.
    # Note that F is here supposed to include the multiplication with area
    # Keeping 'nan' where values are not needed!
    Fe[1:-1, 1:-1] = 0
    Fw[1:-1, 1:-1] = 0
    Fn[1:-1, 1:-1] = 0
    Fs[1:-1, 1:-1] = 0

def setInletVelocityAndFlux(u, v, Fe, Fw, Fn, Fs,
                            nI, nJ, rho, dx_we, dy_sn, nodeX, nodeY, grid_type, caseID):
    # Set inlet boundary values for velocity (u/v) and face fluxes (Fw/Fe/Fn/Fs).
    # The inlet velocity is 1 m/s normal to the inlet for all cases
    # Note that F is here supposed to include the multiplication with area
    # Inlet locations for imported coarse mesh:
    # Cases 1-5:   nodeX = 0, 1.864762 < nodeY < 2.0
    # Cases 6-10:  nodeX = 0, 1.946202 < nodeY < 2.0
    # Cases 11-25: nodeY = H, 1.057008 < nodeX < 1.736459
    # Inlet locations for imported fine mesh:
    # Cases 1-5:   nodeX = 0, 1.864762 < nodeY < 2.0
    # Cases 6-10:  nodeX = 0, 1.968983 < nodeY < 2.0
    # Cases 11-25: nodeY = H, 1.263541 < nodeX < 1.736459
    # match grid_type:
    #     case 'coarse' | 'newCoarse':
    #         # ADD CODE HERE
    #         pass
    #     case 'fine':
    #         # ADD CODE HERE
    #         pass
    #     case _:
    #         sys.exit("Incorrect grid type!")
    
    """
    --------------------------------
    # ADDED CODE - SOLVED
    --------------------------------
    """
    match grid_type:
        case 'coarse' | 'newCoarse':
            for j in range(1,nJ-1):

                i = 1
                if 1.946202 < nodeY[i, j] < 2.0:
                    u[i - 1, j] = 1
                    Fw[i, j] = rho * u[i - 1, j] * dy_sn[i, j]
        case 'fine':
            for j in range(1,nJ-1):

                i = 1
                if 1.968983 < nodeY[i, j] < 2.0:
                    u[i - 1, j] = 1
                    Fw[i, j] = rho * u[i - 1, j] * dy_sn[i, j]
        case _:
            sys.exit("Incorrect grid type!")

def initOutletFlux(Fe, Fw, Fn, Fs,
                   nI, nJ, rho, dx_we, dy_sn, nodeX, nodeY, grid_type, caseID):
    # Initialize outlet flux
    # The outlet fluxes need some non-zero value for the correction that
    # will be done later. To make it easy, we just set a flux
    # corresponding to 1 m/s at all outlet faces.
    # Note that F is here supposed to include the multiplication with area
    # Outlet locations for imported coarse mesh:
    # Cases 1-5:   nodeX = L, 0.0 < nodeY < 0.1352378
    # Cases 6-10:  nodeX = 0, 0.0 < nodeY < 0.03101719
    # Cases 11-20: nodeX = 0, 0.0 < nodeY < 0.122958
    #              nodeX = L, 0.0 < nodeY < 0.122958
    # Cases 21-25: nodeX = L, 0.0 < nodeY < 0.122958
    # Outlet locations for imported fine mesh:
    # Cases 1-5:   nodeX = L, 0.0 < nodeY < 0.1352378
    # Cases 6-10:  nodeX = 0, 0.0 < nodeY < 0.03101719
    # Cases 11-20: nodeX = 0, 0.0 < nodeY < 0.122958
    #              nodeX = L, 0.0 < nodeY < 0.122958
    # Cases 21-25: nodeX = L, 0.0 < nodeY < 0.122958
    # ADD CODE HERE
    # match grid_type:
    #     case 'coarse' | 'fine' | 'newCoarse':
    #         # ADD CODE HERE
    #         pass
    #     case _:
    #         sys.exit("Incorrect grid type!")
    
    """
    --------------------------------
    # ADDED CODE - SOLVED
    --------------------------------
    """
    match grid_type:
        case 'coarse' | 'fine' | 'newCoarse':
            
            for j in range(1,nJ-1):

                i = 1
                if 0 < nodeY[i, j] < 0.03101719:
                    u_out = -1
                    Fw[i, j] = u_out * rho * dy_sn[i, j]
        case _:
            sys.exit("Incorrect grid type!")
    

def correctGlobalContinuity(Fe, Fw, Fn, Fs,
                            nI, nJ):
    # Make sure that outlet flux is equal to inlet flux.
    # Hint: Find inlets/outlets by convective flux into/out of the domain.
    # In this code we don't change the outlet flux later, which forces the
    # flux through the outlet boundary/ies
    # Note that F is here supposed to include the multiplication with area
    # ADD CODE HERE
    pass
    """
    --------------------------------
    # ADDED CODE - SOLVED
    --------------------------------
    """
    
    flux_in = 0
    for j in range(1,nJ-1):

                i = 1
                if Fw[i, j] > 0:
                    flux_in += Fw[i, j]
    
    n_outlets = 0           
    for j in range(1,nJ-1):
                i = 1
                if Fw[i, j] < 0:
                    n_outlets +=1
                    
    for j in range(1,nJ-1):
                i = 1
                if Fw[i, j] < 0:
                    Fw[i, j] = - flux_in / n_outlets
    
    """
    --------------------------------
    # ^^^
    --------------------------------
    """

    # Just for checking (if you want to):
    # globContErr = np.sum(Fe[nI-2,1:nJ-1]) - np.sum(Fw[1,1:nJ-1]) + \
    #               np.sum(Fn[1:nI-1,nJ-2]) - np.sum(Fs[1:nI-1,1])
    # print(globContErr)

def calcD(De, Dw, Dn, Ds,
          gamma, nI, nJ, dx_PE, dx_WP, dy_PN, dy_SP, dx_we, dy_sn):
    # Calculate diffusions conductances in first row of argument list.
    # Note that D is here supposed to include the multiplication with area
    # Only change arrays in first row of argument list!
    # Keep 'nan' where values are not needed!
    # ADD CODE HERE
    # for i in range (1,nI-1):
    #     for j in range(1,nJ-1):
    #         De[i,j] = 0 # ADD CODE HERE
    #         Dw[i,j] = 0 # ADD CODE HERE
    #         Dn[i,j] = 0 # ADD CODE HERE
    #         Ds[i,j] = 0 # ADD CODE HERE       
    """
    --------------------------------
    # ADDED CODE - SOLVED
    --------------------------------
    """
    for i in range (1,nI-1):
        for j in range(1,nJ-1):
            De[i,j] = gamma / dx_PE[i,j] * dy_sn[i,j]
            Dw[i,j] = gamma / dx_WP[i,j] * dy_sn[i,j]
            Dn[i,j] = gamma / dy_PN[i,j] * dx_we[i,j]
            Ds[i,j] = gamma / dy_SP[i,j] * dx_we[i,j]
    
def calcMomEqCoeffs_FOU_CD(aE_uv, aW_uv, aN_uv, aS_uv, aP_uv,
                           nI, nJ, alphaUV, De, Dw, Dn, Ds,
                           Fe, Fw, Fn, Fs):
    """
    ################################
    # OPTIONAL!!! ONLY IF YOU ARE INTERESTED!
    ################################
    """
    # Calculate under-relaxed momentum equation coefficients, based on FOU_CD
    # (First-Order Upwind for convection and Central Differencing for diffusion),
    # using max-functions.
    # Only change arrays in first row of argument list!
    # Keep 'nan' where values are not needed!
    # NEGLECT CONTINUITY ERROR IN CENTRAL COEFFICIENT! (Sp = 0)
    # ADD CODE HERE (IF YOU ARE INTERESTED TO TRY IT OUT)
    # for i in range(1,nI-1):
    #     for j in range(1,nJ-1):
    #         aE_uv[i,j] = 0 # ADD CODE HERE
    #         aW_uv[i,j] = 0 # ADD CODE HERE
    #         aN_uv[i,j] = 0 # ADD CODE HERE
    #         aS_uv[i,j] = 0 # ADD CODE HERE
    #         aP_uv[i,j] = 0 # ADD CODE HERE

def calcMomEqCoeffs_Hybrid(aE_uv, aW_uv, aN_uv, aS_uv, aP_uv,
                           nI, nJ, alphaUV, De, Dw, Dn, Ds,
                           Fe, Fw, Fn, Fs, fxe, fxw, fyn, fys):
    # Calculate under-relaxed momentum equation coefficients, based on Hybrid,
    # using max-functions for non-equidistant mesh.
    # Only change arrays in first row of argument list!
    # Keep 'nan' where values are not needed!
    # NEGLECT CONTINUITY ERROR IN CENTRAL COEFFICIENT! (Sp = 0)
    # ADD CODE HERE
    # for i in range(1,nI-1):
    #     for j in range(1,nJ-1):
    #         aE_uv[i,j] = 0 # ADD CODE HERE
    #         aW_uv[i,j] = 0 # ADD CODE HERE
    #         aN_uv[i,j] = 0 # ADD CODE HERE
    #         aS_uv[i,j] = 0 # ADD CODE HERE
    #         aP_uv[i,j] = 0 # ADD CODE HERE         
    """
    --------------------------------
    # ADDED CODE - SOLVED
    --------------------------------
    """
    for i in range(1,nI-1):
        for j in range(1,nJ-1):
            aE_uv[i,j] = np.max([-Fe[i, j], De[i, j] - fxe[i,j] * Fe[i, j], 0])
            aW_uv[i,j] = np.max([Fw[i, j], Dw[i, j] + fxw[i,j] * Fw[i, j], 0])
            aN_uv[i,j] = np.max([-Fn[i, j], Dn[i, j] - fyn[i,j] * Fn[i, j], 0])
            aS_uv[i,j] = np.max([Fs[i, j], Ds[i, j] + fys[i,j] * Fs[i, j], 0])
    
    # At outlets, set homogeneous Neumann
    for j in range(1, nJ-1):
        i = nI-2

        i = 1
        if Fw[i, j] < 0:
            aW_uv[i, j] = 0
        
    for i in range(1,nI-1):
        j = nJ-2
        
        j = 1
    
    # (Homogeneous) Neumann walls:
    for i in range(1,nI-1):
        j = nJ-2
        if  Fn[i, j + 1] == 0:
            aN_uv[i, j] = 0
            
        j = 1
        if Fs[i, j - 1] == 0:
            aS_uv[i, j] = 0
        
    for j in range(1,nJ-1):
        i = nI-2
        if Fe[i + 1,j] == 0:
            aE_uv[i, j] = 0
            
        i = 1
        if Fw[i - 1, j] == 0:
            aW_uv[i, j] = 0
    
    for i in range(1,nI-1):
        for j in range(1,nJ-1):       
            aP_uv[i,j] = aE_uv[i,j] + aW_uv[i,j] + aN_uv[i,j] + aS_uv[i,j]
            
            aP_uv[i, j] = aP_uv[i, j] / alphaUV
            
def calcMomEqSu(Su_u, Su_v,
                nI, nJ, p, dx_WP, dx_PE, dy_SP, dy_PN, dx_we, dy_sn,
                alphaUV, aP_uv, u, v, fxe, fxw, fyn, fys):
    # Calculate under-relaxed momentum equation source terms,
    # based on linearly interpolated pressure face values.
    # Only change arrays in first row of argument list!
    # Keep 'nan' where values are not needed!
    # NEGLECT CONTINUITY ERROR IN CENTRAL COEFFICIENT! (Sp = 0)
    # ADD CODE HERE
    # for i in range(1,nI-1):
    #     for j in range(1,nJ-1):
    #         Su_u[i,j] = 0 # ADD CODE HERE
    #         Su_v[i,j] = 0 # ADD CODE HERE     
    """
    --------------------------------
    # ADDED CODE - SOLVED
    --------------------------------
    """
    for i in range(1,nI-1):
        for j in range(1,nJ-1):
            
            p_e = fxe[i, j] * p[i + 1, j] + (1 - fxe[i, j]) * p[i, j]
            p_w = fxw[i, j] * p[i - 1, j] + (1 - fxw[i, j]) * p[i, j]
            p_n = fyn[i, j] * p[i, j + 1] + (1 - fyn[i, j]) * p[i, j]
            p_s = fys[i, j] * p[i, j - 1] + (1 - fys[i, j]) * p[i, j]
            
            Su_u[i,j] = - (p_e - p_w) * dy_sn[i, j]
            Su_u[i,j] =  Su_u[i,j] + (1 - alphaUV) * aP_uv[i, j] * u[i, j]
            
            Su_v[i,j] = - (p_n - p_s) * dx_we[i, j]
            Su_v[i,j] = Su_v[i,j] + (1 - alphaUV) * aP_uv[i, j] * v[i, j]
            
            # f_w[1]*p[0]+(f_e[1]-f_w[1])*p[1]-f_e[1]*p[2] + b*dx_we[1], \
            # f_w[2]*p[1]+(f_e[2]-f_w[2])*p[2]-f_e[2]*p[3] + b*dx_we[2], \

def solveGaussSeidel(phi,
                     nI, nJ, aE, aW, aN, aS, aP, Su, nLinSolIter):
    # Implement the Gauss-Seidel solver for general variable phi,
    # so it can be reused for all variables.
    # Do it in two directions, as in Task 2
    # Only change arrays in first row of argument list!
    # ADD CODE HERE
    # pass
    """
    --------------------------------
    # ADDED CODE - SOLVED
    --------------------------------
    """
    for linSolIter in range(nLinSolIter):  
         
        for i in range(1,nI-1):
            for j in range(1,nJ-1):
                phi[i,j] = (aE[i, j] * phi[i + 1, j] +
                            aW[i, j] * phi[i - 1, j] +
                            aN[i, j] * phi[i, j + 1] +
                            aS[i, j] * phi[i, j - 1] +
                            Su[i, j]) / aP[i, j]
            
        for j in range(1,nJ-1):
            for i in range(1,nI-1):
                phi[i,j] = (aE[i, j] * phi[i + 1, j] +
                            aW[i, j] * phi[i - 1, j] +
                            aN[i, j] * phi[i, j + 1] +
                            aS[i, j] * phi[i, j - 1] +
                            Su[i, j]) / aP[i, j]

def calcRhieChow_noCorr(Fe, Fw, Fn, Fs,
                        nI, nJ, rho, u, v,
                        dx_we, dy_sn, fxe, fxw, fyn, fys):
    # Calculate face fluxes for the pressure correction equation source term,
    # using central differencing of velocity on a non-equidistant mesh,
    # without Rhie & Chow correction.
    # Note that F is here supposed to include the multiplication with area
    # Easiest to implement, so start with this.
    # Gives checkerboarding - have a look!
    # DO NOT TOUCH BOUNDARY FLUXES, WHICH ARE SET WITH BOUNDARY CONDITIONS!
    # Only change arrays in first row of argument list!
    # Keep 'nan' where values are not needed!
    # ADD CODE HERE
    pass
    """
    --------------------------------
    # ADDED CODE
    --------------------------------
    """
    

def calcRhieChow_equiCorr(Fe, Fw, Fn, Fs,
                          nI, nJ, rho, u, v,
                          dx_we, dy_sn, fxe, fxw, fyn, fys, aP_uv, p):
    # Calculate face fluxes for pressure correction equation source term,
    # using central differencing of velocity on a non-equidistant mesh,
    # and equidistant implementation of Rhie & Chow correction term.
    # Note that F is here supposed to include the multiplication with area
    # DO NOT TOUCH BOUNDARY FLUXES, WHICH ARE SET WITH BOUNDARY CONDITIONS!
    # Only change arrays in first row of argument list!
    # Keep 'nan' where values are not needed!
    # ADD CODE HERE
    pass
    """
    --------------------------------
    # ADDED CODE
    --------------------------------
    """
    

def calcRhieChow_nonEquiCorr(Fe, Fw, Fn, Fs,
                             nI, nJ, rho, u, v,
                             dx_we, dy_sn, fxe, fxw, fyn, fys, aP_uv, p,
                             dx_WP, dx_PE, dy_SP, dy_PN):
    """
    ################################
    # OPTIONAL!!! ONLY IF YOU ARE INTERESTED!
    ################################
    """
    # Calculate face fluxes for pressure correction equation source term,
    # using central differencing of velocity on a non-equidistant mesh,
    # and non-equidistant implementation of Rhie & Chow correction term.
    # Note that F is here supposed to include the multiplication with area
    # DO NOT TOUCH BOUNDARY FLUXES, WHICH ARE SET WITH BOUNDARY CONDITIONS!
    # Only change arrays in first row of argument list!
    # Keep 'nan' where values are not needed!
    # ADD CODE HERE (IF YOU ARE INTERESTED TO TRY IT OUT)
    pass

def calcPpEqCoeffs(aE_pp, aW_pp, aN_pp, aS_pp, aP_pp, de, dw, dn, ds,
                   nI, nJ, rho, dx_we, dy_sn, fxe, fxw, fyn, fys, aP_uv):
    # Calculate pressure correction equation coefficients.
    # Make sure to treat boundary conditions correctly!
    # Note that de, dw, dn, ds are useful also later,
    # so make sure to set them and use them appropritely later.
    # Only change arrays in first row of argument list!
    # Keep 'nan' where values are not needed!
    # ADD CODE HERE
    pass
    """
    --------------------------------
    # ADDED CODE
    --------------------------------
    """

def calcPpEqSu(Su_pp,
               nI, nJ, Fe, Fw, Fn, Fs):
    # Calculate pressure correction equation source term
    # Only change arrays in first row of argument list!
    # Keep 'nan' where values are not needed!
    # ADD CODE HERE
    pass
    """
    --------------------------------
    # ADDED CODE
    --------------------------------
    """

def fixPp(Su_pp, aP_pp,
          pRef_i, pRef_j, aE_pp, aW_pp, aN_pp, aS_pp):
    # Fix pressure by forcing pp to zero in reference node, through source terms
    # MAKES CONVERGENCE POOR, SO BETTER TO SKIP IT FOR NOW. TRY IF YOU LIKE
    # ADD CODE HERE
    pass
    """
    --------------------------------
    # ADDED CODE
    --------------------------------
    """

def solveTDMA(phi,
              nI, nJ, aE, aW, aN, aS, aP, Su,
              nLinSolIter_phi):
    # Implement the TDMA solver for general variable phi,
    # so it can be reused for all variables.
    # Do it in two directions, as in Task 2.
    # Only change arrays in first row of argument list!
    # ADD CODE HERE
    # pass
    """
    --------------------------------
    # ADDED CODE - SOLVED
    --------------------------------
    """
    
    nan = float("nan")
    P      = np.zeros((nI,nJ))*nan       # Array for TDMA, in nodes
    Q      = np.zeros((nI,nJ))*nan       # Array for TDMA, in nodes
    for linSolIter in range(0,nLinSolIter_phi):
        # March from west to east
        # Sweep from south to north
        for j in range(1, nJ - 1):
            
            i = 1
            
            a = aP[i, j]
            b = aE[i, j]
            c = aW[i, j]
            
            d = (aN[i, j] * phi[i, j + 1] + 
                 aS[i, j] * phi[i, j - 1] + 
                 Su[i, j])
            
            P[i, j] = b / a
            Q[i, j] = (d + c * phi[i - 1, j]) / a
            
            for i in range(2, nI - 1):
                
                a = aP[i, j]
                b = aE[i, j]
                c = aW[i, j]
                d = (aN[i, j] * phi[i, j + 1] + 
                     aS[i, j] * phi[i, j - 1] + 
                     Su[i, j])
                
                denominator = a - c * P[i - 1, j]
                
                P[i, j] = b / denominator
                Q[i, j] = (d + c * Q[i - 1, j]) / denominator
            
            for i in reversed(range(1, nI - 1)):
                phi[i, j] = P[i, j] * phi[i + 1, j] + Q[i, j]
            
        # March from north to south <- THIS IS WRONG, we're marching from south to north
        # Sweep from west to east 
        for i in range(1, nI - 1):
            
            j = 1
            
            a = aP[i, j]
            b = aN[i, j]
            c = aS[i, j]
            d = (aE[i, j] * phi[i + 1, j] + 
                 aW[i, j] * phi[i - 1, j] + 
                 Su[i, j])
            
            P[i, j] = b / a
            Q[i, j] = (d + c * phi[i, j - 1]) / a
            
            for j in range(2, nJ - 1):
            
                a = aP[i, j]
                b = aN[i, j]
                c = aS[i, j]
                d = (aE[i, j] * phi[i + 1, j] + 
                     aW[i, j] * phi[i - 1, j] + 
                     Su[i, j])
                
                denominator = a - c * P[i, j - 1]
                
                P[i, j] = b / denominator
                Q[i, j] = (d + c * Q[i, j - 1]) / denominator
            
            for j in reversed(range(1, nJ - 1)):
                
                phi[i, j] = P[i, j] * phi[i, j + 1] + Q[i, j]

def setPressureCorrectionLevel(pp,
                               nI, nJ, pRef_i, pRef_j):
    # Set pressure correction level explicitly
    # Only change arrays in first row of argument list!
    # ADD CODE HERE
    pass
    """
    --------------------------------
    # ADDED CODE
    --------------------------------
    """

def correctPressureCorrectionBC(pp,
                                nI, nJ):
    # Correct pressure correction homogeneous Neumann boundary conditions
    # Only change arrays in first row of argument list!
    # ADD CODE HERE
    pass
    """
    --------------------------------
    # ADDED CODE
    --------------------------------
    """

def correctPressure(p,
                    nI, nJ, alphaP, pp):
    # Correct pressure, using explicit under-relaxation
    # Only change arrays in first row of argument list!
    # ADD CODE HERE
    pass
    """
    --------------------------------
    # ADDED CODE
    --------------------------------
    """

def correctPressureBCandCorners(p,
                                nI, nJ, dx_PE, dx_WP, dy_PN, dy_SP):
    # Extrapolate pressure to boundaries, using constant gradient,
    # required to get correct Suu in u-mom. equation!
    # Only change arrays in first row of argument list!
    # ADD CODE HERE
    """
    --------------------------------
    # ADDED CODE
    --------------------------------
    """
    
    
    """
    --------------------------------
    # ^^^
    --------------------------------
    """

    # Interpolate pressure to corners (kept so all do the same)
    p[0,0] = 0.5*(p[1,0]+p[0,1])
    p[nI-1,0] = 0.5*(p[nI-2,0]+p[nI-1,1])
    p[0,nJ-1] = 0.5*(p[1,nJ-1]+p[0,nJ-2])
    p[nI-1,nJ-1] = 0.5*(p[nI-2,nJ-1]+p[nI-1,nJ-2])
    
    pass

def correctVelocity(u, v,
                    nI, nJ, fxe, fxw, fyn, fys, pp, dy_sn, dx_we, aP_uv):
    # Correct velocity components using pp solution (DO NOT TOUCH BOUNDARIES!)
    # Only change arrays in first row of argument list!
    # ADD CODE HERE
    pass
    """
    --------------------------------
    # ADDED CODE
    --------------------------------
    """

def correctOutletVelocity(u, v,
                          nI, nJ, rho, dx_we, dy_sn, nodeX, nodeY, grid_type, caseID):
    # Extraplate velocity at outlet (zero gradient for both u and v)
    # Outlet locations for imported coarse mesh:
    # Cases 1-5:   nodeX = L, 0.0 < nodeY < 0.1352378
    # Cases 6-10:  nodeX = 0, 0.0 < nodeY < 0.03101719
    # Cases 11-20: nodeX = 0, 0.0 < nodeY < 0.122958
    #              nodeX = L, 0.0 < nodeY < 0.122958
    # Cases 21-25: nodeX = L, 0.0 < nodeY < 0.122958
    # Outlet locations for imported fine mesh:
    # Cases 1-5:   nodeX = L, 0.0 < nodeY < 0.1352378
    # Cases 6-10:  nodeX = 0, 0.0 < nodeY < 0.03101719
    # Cases 11-20: nodeX = 0, 0.0 < nodeY < 0.122958
    #              nodeX = L, 0.0 < nodeY < 0.122958
    # Cases 21-25: nodeX = L, 0.0 < nodeY < 0.122958
    # ADD CODE HERE
    """
    --------------------------------
    # ADDED CODE
    --------------------------------
    """
    
    match grid_type:
        case 'coarse' | 'fine' | 'newCoarse':
            for j in range(1,nJ-1):

                i = 1
                if 0 < nodeY[i, j] < 0.03101719:
                    pass
        case _:
            sys.exit("Incorrect grid type!")

def correctFaceFlux(Fe, Fw, Fn, Fs,
                    nI, nJ, rho, dy_sn, dx_we, de, dw, dn, ds, pp):
    # Correct face fluxes using pp solution (DO NOT TOUCH BOUNDARIES!)
    # Note that F is here supposed to include the multiplication with area
    # Only change arrays in first row of argument list!
    pass
    """
    --------------------------------
    # ADDED CODE
    --------------------------------
    """

def calcNormalizedResiduals(res_u, res_v, res_c,
                            nI, nJ, iter, u, v,
                            aP_uv, aE_uv, aW_uv, aN_uv, aS_uv, Su_u, Su_v,
                            Fe, Fw, Fn, Fs):
    # Make normalization factors global so they are remembered for next call
    # to this function. Not an ideal way to do it, since they can potentially
    # be changed elsewhere, but we do it like this anyway and make sure not
    # to change them anywhere else.
    # global F_uv, F_c
    # # ADD CODE HERE
    
    # # Compute residuals
    # res_u.append(0) # U momentum residual
    # res_v.append(0) # V momentum residual
    # res_c.append(0) # Continuity residual/error
    # for i in range(1,nI-1):
    #     for j in range(1,nJ-1):
    #         res_u[-1] = 1 # ADD CODE HERE
    #         res_v[-1] = 1 # ADD CODE HERE
    #         res_c[-1] = 1 # ADD CODE HERE
    """
    --------------------------------
    # ADDED CODE
    --------------------------------
    """
    
    global F_uv, F_c
    
    # Compute residuals
    res_u.append(0) # U momentum residual
    res_v.append(0) # V momentum residual
    res_c.append(0) # Continuity residual/error
    for i in range(1,nI-1):
        for j in range(1,nJ-1):
            res_u[-1] = 1 
            res_v[-1] = 1 
            res_c[-1] = 1 
    
    """
    --------------------------------
    # ^^^
    --------------------------------
    """
    
    # Normalization with first non-normalized residual:        
    # Same normalization factor for u,v (based on largest initial)
    # Separate normalization factor for continuity residual/error
    if iter == 0:
        F_uv = max(res_u[-1],res_v[-1])
        F_c = res_c[-1]
    res_u[-1] = res_u[-1] / F_uv
    res_v[-1] = res_v[-1] / F_uv
    res_c[-1] = res_c[-1] / F_c

def createDefaultPlots(
               nI, nJ, pointX, pointY, nodeX, nodeY, pRef_i, pRef_j,
               caseID, grid_type, u, v, uTask2, vTask2, p,
               iter, res_u, res_v, res_c):
    """
    ################################
    # DO NOT CHANGE ANYTHING HERE! #
    ################################
    """
    # (Do not change any input arrays!)
    if not os.path.isdir('Figures'):
        os.makedirs('Figures')
    
    # Plot mesh
    plt.figure()
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.title('Computational mesh\n(reference pressure node marked)')
    plt.axis('equal')
    plt.vlines(pointX[:,0],pointY[0,0],pointY[0,-1],colors = 'k',linestyles = 'dashed')
    plt.hlines(pointY[0,:],pointX[0,0],pointX[-1,0],colors = 'k',linestyles = 'dashed')
    plt.plot(nodeX, nodeY, 'ro')
    plt.plot(nodeX[pRef_i,pRef_j], nodeY[pRef_i,pRef_j], 'bo')
    plt.show()
    plt.savefig('Figures/Case_'+str(caseID)+'_'+grid_type+'_mesh.png')
    
    # Plot velocity vectors
    plt.figure()
    plt.quiver(nodeX.T, nodeY.T, u.T, v.T)
    plt.title('Velocity vectors')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.axis('equal')
    plt.show()
    plt.savefig('Figures/Case_'+str(caseID)+'_'+grid_type+'_velocityVectors.png')
    
    # Plot u-velocity contour
    plt.figure()
    tempmap=plt.contourf(nodeX.T,nodeY.T,u.T,cmap='coolwarm',levels=30)
    cbar=plt.colorbar(tempmap)
    cbar.set_label('$[m/s]$')
    plt.title('U velocity $[m/s]$')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.axis('equal')
    plt.tight_layout()
    plt.show()
    plt.savefig('Figures/Case_'+str(caseID)+'_'+grid_type+'_uVelocityContour.png')
    
    # Plot v-velocity contour
    plt.figure()
    tempmap=plt.contourf(nodeX.T,nodeY.T,v.T,cmap='coolwarm',levels=30)
    cbar=plt.colorbar(tempmap)
    cbar.set_label('$[m/s]$')
    plt.title('V velocity $[m/s]$')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.axis('equal')
    plt.tight_layout()
    plt.show()
    plt.savefig('Figures/Case_'+str(caseID)+'_'+grid_type+'_vVelocityContour.png')
    
    # Plot pressure contour
    plt.figure()
    tempmap=plt.contourf(nodeX.T,nodeY.T,p.T,cmap='coolwarm',levels=30)
    cbar=plt.colorbar(tempmap)
    cbar.set_label('$[Pa]$')
    plt.plot(nodeX[pRef_i,pRef_j], nodeY[pRef_i,pRef_j], 'bo')
    plt.title('Pressure $[Pa]$\n(reference pressure node marked)')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.axis('equal')
    plt.tight_layout()
    plt.show()
    plt.savefig('Figures/Case_'+str(caseID)+'_'+grid_type+'_pressureContour.png')
    
    # Plot velocity validation
    plt.figure()
    plt.plot(nodeX[:,int(nJ/2)], u[:,int(nJ/2)], 'b', label = 'U')
    plt.plot(nodeX[:,int(nJ/2)], uTask2[:,int(nJ/2)], 'bx', label = 'U ref')
    plt.plot(nodeX[:,int(nJ/2)], v[:,int(nJ/2)], 'r', label = 'V')
    plt.plot(nodeX[:,int(nJ/2)], vTask2[:,int(nJ/2)], 'rx', label = 'V ref')
    plt.title('Velocity validation (horizontal centerline)')
    plt.xlabel('x [m]')
    plt.ylabel('Velocity [m/s]')
    plt.legend()
    plt.show()
    plt.savefig('Figures/Case_'+str(caseID)+'_'+grid_type+'_velocityValidation.png')
    
    # Plot residual convergence
    plt.figure()
    plt.semilogy(range(0,iter+1), res_u, 'blue', label = 'U')
    plt.semilogy(range(0,iter+1), res_v, 'red', label = 'V')
    plt.semilogy(range(0,iter+1), res_c, 'green', label = 'Continuity')
    plt.title('Residual convergence')
    plt.xlabel('Iterations')
    plt.ylabel('Residual [-]')
    plt.legend()
    plt.show()
    plt.savefig('Figures/Case_'+str(caseID)+'_'+grid_type+'_residualConvergence.png')

def createAdditionalPlots():
    pass
#%%