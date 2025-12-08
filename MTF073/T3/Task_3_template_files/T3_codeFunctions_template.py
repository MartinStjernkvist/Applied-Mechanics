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
    ################################
    # DO NOT CHANGE ANYTHING HERE! #
    ################################
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
    for i in range(1, nI-1):
        for j in range(1, nJ-1):
            dx_PE[i,j] = 0 # ADD CODE HERE
            dx_WP[i,j] = 0 # ADD CODE HERE
            dy_PN[i,j] = 0 # ADD CODE HERE
            dy_SP[i,j] = 0 # ADD CODE HERE
            dx_we[i,j] = 0 # ADD CODE HERE
            dy_sn[i,j] = 0 # ADD CODE HERE
    
def calcInterpolationFactors(fxe, fxw, fyn, fys,
                             nI, nJ, dx_PE, dx_WP, dy_PN, dy_SP, dx_we, dy_sn):
    # Calculate interpolation factors in first row of argument list.
    # Only change arrays in first row of argument list!
    # Keep 'nan' where values are not needed!
    # ADD CODE HERE
    for i in range(1, nI-1):
        for j in range(1, nJ-1):
            fxe[i,j] = 0 # ADD CODE HERE
            fxw[i,j] = 0 # ADD CODE HERE
            fyn[i,j] = 0 # ADD CODE HERE
            fys[i,j] = 0 # ADD CODE HERE 

def initArrays(u, v, p, Fe, Fw, Fn, Fs):
    ################################
    # DO NOT CHANGE ANYTHING HERE! #
    ################################
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
    match grid_type:
        case 'coarse' | 'newCoarse':
            # ADD CODE HERE
            pass
        case 'fine':
            # ADD CODE HERE
            pass
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
    match grid_type:
        case 'coarse' | 'fine' | 'newCoarse':
            # ADD CODE HERE
            pass
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
    for i in range (1,nI-1):
        for j in range(1,nJ-1):
            De[i,j] = 0 # ADD CODE HERE
            Dw[i,j] = 0 # ADD CODE HERE
            Dn[i,j] = 0 # ADD CODE HERE
            Ds[i,j] = 0 # ADD CODE HERE

def calcMomEqCoeffs_FOU_CD(aE_uv, aW_uv, aN_uv, aS_uv, aP_uv,
                           nI, nJ, alphaUV, De, Dw, Dn, Ds,
                           Fe, Fw, Fn, Fs):
    # OPTIONAL!!! ONLY IF YOU ARE INTERESTED!
    # Calculate under-relaxed momentum equation coefficients, based on FOU_CD
    # (First-Order Upwind for convection and Central Differencing for diffusion),
    # using max-functions.
    # Only change arrays in first row of argument list!
    # Keep 'nan' where values are not needed!
    # NEGLECT CONTINUITY ERROR IN CENTRAL COEFFICIENT! (Sp = 0)
    # ADD CODE HERE (IF YOU ARE INTERESTED TO TRY IT OUT)
    for i in range(1,nI-1):
        for j in range(1,nJ-1):
            aE_uv[i,j] = 0 # ADD CODE HERE
            aW_uv[i,j] = 0 # ADD CODE HERE
            aN_uv[i,j] = 0 # ADD CODE HERE
            aS_uv[i,j] = 0 # ADD CODE HERE
            aP_uv[i,j] = 0 # ADD CODE HERE

def calcMomEqCoeffs_Hybrid(aE_uv, aW_uv, aN_uv, aS_uv, aP_uv,
                           nI, nJ, alphaUV, De, Dw, Dn, Ds,
                           Fe, Fw, Fn, Fs, fxe, fxw, fyn, fys):
    # Calculate under-relaxed momentum equation coefficients, based on Hybrid,
    # using max-functions for non-equidistant mesh.
    # Only change arrays in first row of argument list!
    # Keep 'nan' where values are not needed!
    # NEGLECT CONTINUITY ERROR IN CENTRAL COEFFICIENT! (Sp = 0)
    # ADD CODE HERE
    for i in range(1,nI-1):
        for j in range(1,nJ-1):
            aE_uv[i,j] = 0 # ADD CODE HERE
            aW_uv[i,j] = 0 # ADD CODE HERE
            aN_uv[i,j] = 0 # ADD CODE HERE
            aS_uv[i,j] = 0 # ADD CODE HERE
            aP_uv[i,j] = 0 # ADD CODE HERE

def calcMomEqSu(Su_u, Su_v,
                nI, nJ, p, dx_WP, dx_PE, dy_SP, dy_PN, dx_we, dy_sn,
                alphaUV, aP_uv, u, v, fxe, fxw, fyn, fys):
    # Calculate under-relaxed momentum equation source terms,
    # based on linearly interpolated pressure face values.
    # Only change arrays in first row of argument list!
    # Keep 'nan' where values are not needed!
    # NEGLECT CONTINUITY ERROR IN CENTRAL COEFFICIENT! (Sp = 0)
    # ADD CODE HERE
    for i in range(1,nI-1):
        for j in range(1,nJ-1):
            Su_u[i,j] = 0 # ADD CODE HERE
            Su_v[i,j] = 0 # ADD CODE HERE

def solveGaussSeidel(phi,
                     nI, nJ, aE, aW, aN, aS, aP, Su, nLinSolIter):
    # Implement the Gauss-Seidel solver for general variable phi,
    # so it can be reused for all variables.
    # Do it in two directions, as in Task 2
    # Only change arrays in first row of argument list!
    # ADD CODE HERE
    pass

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

def calcRhieChow_nonEquiCorr(Fe, Fw, Fn, Fs,
                             nI, nJ, rho, u, v,
                             dx_we, dy_sn, fxe, fxw, fyn, fys, aP_uv, p,
                             dx_WP, dx_PE, dy_SP, dy_PN):
    # OPTIONAL!!! ONLY IF YOU ARE INTERESTED!
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

def calcPpEqSu(Su_pp,
               nI, nJ, Fe, Fw, Fn, Fs):
    # Calculate pressure correction equation source term
    # Only change arrays in first row of argument list!
    # Keep 'nan' where values are not needed!
    # ADD CODE HERE
    pass

def fixPp(Su_pp, aP_pp,
          pRef_i, pRef_j, aE_pp, aW_pp, aN_pp, aS_pp):
    # Fix pressure by forcing pp to zero in reference node, through source terms
    # MAKES CONVERGENCE POOR, SO BETTER TO SKIP IT FOR NOW. TRY IF YOU LIKE
    # ADD CODE HERE
    pass

def solveTDMA(phi,
              nI, nJ, aE, aW, aN, aS, aP, Su,
              nLinSolIter_phi):
    # Implement the TDMA solver for general variable phi,
    # so it can be reused for all variables.
    # Do it in two directions, as in Task 2.
    # Only change arrays in first row of argument list!
    # ADD CODE HERE
    pass

def setPressureCorrectionLevel(pp,
                               nI, nJ, pRef_i, pRef_j):
    # Set pressure correction level explicitly
    # Only change arrays in first row of argument list!
    # ADD CODE HERE
    pass

def correctPressureCorrectionBC(pp,
                                nI, nJ):
    # Correct pressure correction homogeneous Neumann boundary conditions
    # Only change arrays in first row of argument list!
    # ADD CODE HERE
    pass

def correctPressure(p,
                    nI, nJ, alphaP, pp):
    # Correct pressure, using explicit under-relaxation
    # Only change arrays in first row of argument list!
    # ADD CODE HERE
    pass

def correctPressureBCandCorners(p,
                                nI, nJ, dx_PE, dx_WP, dy_PN, dy_SP):
    # Extrapolate pressure to boundaries, using constant gradient,
    # required to get correct Suu in u-mom. equation!
    # Only change arrays in first row of argument list!
    # ADD CODE HERE

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
    match grid_type:
        case 'coarse' | 'fine' | 'newCoarse':
            pass
        case _:
            sys.exit("Incorrect grid type!")

def correctFaceFlux(Fe, Fw, Fn, Fs,
                    nI, nJ, rho, dy_sn, dx_we, de, dw, dn, ds, pp):
    # Correct face fluxes using pp solution (DO NOT TOUCH BOUNDARIES!)
    # Note that F is here supposed to include the multiplication with area
    # Only change arrays in first row of argument list!
    pass

def calcNormalizedResiduals(res_u, res_v, res_c,
                            nI, nJ, iter, u, v,
                            aP_uv, aE_uv, aW_uv, aN_uv, aS_uv, Su_u, Su_v,
                            Fe, Fw, Fn, Fs):
    # Make normalization factors global so they are remembered for next call
    # to this function. Not an ideal way to do it, since they can potentially
    # be changed elsewhere, but we do it like this anyway and make sure not
    # to change them anywhere else.
    global F_uv, F_c
    # ADD CODE HERE
    
    # Compute residuals
    res_u.append(0) # U momentum residual
    res_v.append(0) # V momentum residual
    res_c.append(0) # Continuity residual/error
    for i in range(1,nI-1):
        for j in range(1,nJ-1):
            res_u[-1] = 1 # ADD CODE HERE
            res_v[-1] = 1 # ADD CODE HERE
            res_c[-1] = 1 # ADD CODE HERE
    
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
    ################################
    # DO NOT CHANGE ANYTHING HERE! #
    ################################
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
