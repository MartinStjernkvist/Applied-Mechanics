#%%
# The header of the corresponding code for Task 1 is relevant also here.
# It is not repeated. Please remind yourself if needed.

# Packages needed
import numpy as np
import matplotlib.pyplot as plt
# Set default font size in plots:
plt.rcParams.update({'font.size': 12})
import os # For saving plots

def createMesh(pointX, pointY,
               mI, mJ, pointXvector, pointYvector):
    """
    ################################
    # DO NOT CHANGE ANYTHING HERE! #
    ################################
    """
    # Only changes arrays in first row of argument list!
    # Sets point coordinates for Task 2 cases.
    for i in range(0, mI):
        for j in range(0, mJ):
            pointX[i,j] = pointXvector[i]
            pointY[i,j] = pointYvector[j]
    
def calcNodePositions(nodeX, nodeY,
                      nI, nJ, pointX, pointY):
    """
    ################################
    # DO NOT CHANGE ANYTHING HERE! #
    ################################
    """
    # Only changes arrays in first row of argument list!
    # Calculates node coordinates.
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
            # dx_PE[i,j] = 0 # ADD CODE HERE
            # dx_WP[i,j] = 0 # ADD CODE HERE
            # dy_PN[i,j] = 0 # ADD CODE HERE
            # dy_SP[i,j] = 0 # ADD CODE HERE
            # dx_we[i,j] = 0 # ADD CODE HERE
            # dy_sn[i,j] = 0 # ADD CODE HERE
            """
            --------------------------------
            # ADDED CODE
            --------------------------------
            """     
            dx_PE[i,j] = 0 
            dx_WP[i,j] = 0 
            dy_PN[i,j] = 0 
            dy_SP[i,j] = 0 
            dx_we[i,j] = 0 
            dy_sn[i,j] = 0 

def calcInterpolationFactors(fxe, fxw, fyn, fys,
                             nI, nJ, dx_PE, dx_WP, dy_PN, dy_SP, dx_we, dy_sn):
    # Calculate interpolation factors in first row of argument list.
    # Only change arrays in first row of argument list!
    # Keep 'nan' where values are not needed!
    # ADD CODE HERE
    for i in range(1, nI-1):
        for j in range(1, nJ-1):
            # fxe[i,j] = 0 # ADD CODE HERE
            # fxw[i,j] = 0 # ADD CODE HERE
            # fyn[i,j] = 0 # ADD CODE HERE
            # fys[i,j] = 0 # ADD CODE HERE 
            """
            --------------------------------
            # ADDED CODE
            --------------------------------
            """
            fxe[i,j] = 0
            fxw[i,j] = 0
            fyn[i,j] = 0
            fys[i,j] = 0 

def initArray(T,
              T_init):
    """
    ################################
    # DO NOT CHANGE ANYTHING HERE! #
    ################################
    """
    # Sets initial default temperature
    # (in all nodes, for contour plot).
    T[:,:] = T_init

def setDirichletBCs(T,
                    nI, nJ, u, v, T_in, T_west, T_east, T_south, T_north):
    # Set Dirichlet boundary conditions
    # (only changes arrays in first row of argument list)
    # Note that a value is needed in all nodes for contour plot
    # Inlets (found by velocity into domain):
    # for i in range(nI):
    #     j = nJ-1
    #     # ADD CODE HERE
    #     j = 0
    #     # ADD CODE HERE
    # for j in range(nJ):
    #     i = nI-1
    #     # ADD CODE HERE
    #     i = 0
    #     # ADD CODE HERE
    # # Outlets:
    #     # Homogeneous Neumann:
    #     # Set coefficients later, default value already set
    # # Walls (found by zero velocity), Dirichlet or initial guess:
    # for i in range(nI):
    #     j = nJ-1
    #     # ADD CODE HERE
    #     j = 0
    #     # ADD CODE HERE
    # for j in range(nJ):
    #     i = nI-1
    #     # ADD CODE HERE
    #     i = 0
    #     # ADD CODE HERE
    """
    --------------------------------
    # ADDED CODE
    --------------------------------
    """
    for i in range(nI):
        j = nJ-1
        
        j = 0
        
    for j in range(nJ):
        i = nI-1
        
        i = 0
        
    # Outlets:
        # Homogeneous Neumann:
        # Set coefficients later, default value already set
        
    # Walls (found by zero velocity), Dirichlet or initial guess:
    for i in range(nI):
        j = nJ-1
        
        j = 0
        
    for j in range(nJ):
        i = nI-1
        
        i = 0
        

def calcSourceTerms(Su, Sp,
                    nI, nJ, q_wall, Cp, u, v, dx_we, dy_sn, rho, deltaT, T_o, caseID):
    # Calculate constant source terms
    # (only change arrays in first row of argument list)
    # Keep 'nan' where values are not needed!
    
    # # Default values:
    # for i in range(1,nI-1):
    #     for j in range(1,nJ-1):
    #         Su[i,j] = 0 # ADD CODE HERE
    #         Sp[i,j] = 0 # ADD CODE HERE

    # # Heat rate walls (found by zero velocity):
    # for i in range(1,nI-1):
    #     j = nJ-2
    #     # ADD CODE HERE
    #     j = 1
    #     # ADD CODE HERE
    # for j in range(1,nJ-1):
    #     i = nI-2
    #     # ADD CODE HERE
    #     i = 1
    #     # ADD CODE HERE

    # # Time term:
    # for i in range(1,nI-1):
    #     for j in range(1,nJ-1):
    #         pass # ADD CODE HERE
    """
    --------------------------------
    # ADDED CODE
    --------------------------------
    """
    # Default values:
    for i in range(1,nI-1):
        for j in range(1,nJ-1):
            Su[i,j] = 0 
            Sp[i,j] = 0 

    # Heat rate walls (found by zero velocity):
    for i in range(1,nI-1):
        j = nJ-2
        
        j = 1
        
    for j in range(1,nJ-1):
        i = nI-2
        
        i = 1
        

    # Time term:
    for i in range(1,nI-1):
        for j in range(1,nJ-1):
            pass 
    

def calcD(De, Dw, Dn, Ds,
          gamma, nI, nJ, dx_PE, dx_WP, dy_PN, dy_SP, dx_we, dy_sn):
    # Calculate diffusions conductances in first row of argument list.
    # Note that D is here supposed to include the multiplication with area
    # Only change arrays in first row of argument list!
    # Keep 'nan' where values are not needed!
    # ADD CODE HERE
    for i in range (1,nI-1):
        for j in range(1,nJ-1):
            # De[i,j] = 0 # ADD CODE HERE
            # Dw[i,j] = 0 # ADD CODE HERE
            # Dn[i,j] = 0 # ADD CODE HERE
            # Ds[i,j] = 0 # ADD CODE HERE
            """
            --------------------------------
            # ADDED CODE
            --------------------------------
            """
            De[i,j] = 0 
            Dw[i,j] = 0 
            Dn[i,j] = 0 
            Ds[i,j] = 0 

def calcF(Fe, Fw, Fn, Fs,
          rho, nI, nJ, dx_we, dy_sn, fxe, fxw, fyn, fys, u, v):
    # Calculate constant convective (F) coefficients by linear interpolation
    # of velocity in nodes to faces
    # Note that F is here supposed to include the multiplication with area
    # (only changes arrays in first row of argument list)
    # Keep 'nan' where values are not needed!
    # ADD CODE HERE
    for i in range(1,nI-1):
        for j in range(1,nJ-1):
            Fe[i,j] = 0
            Fw[i,j] = 0
            Fn[i,j] = 0
            Fs[i,j] = 0
            """
            --------------------------------
            # ADDED CODE
            --------------------------------
            """

def calcHybridCoeffs(aE, aW, aN, aS, aP,
                     nI, nJ, De, Dw, Dn, Ds, Fe, Fw, Fn, Fs,
                     fxe, fxw, fyn, fys, dy_sn, Sp, u, v,
                     nodeX, nodeY, L, H, caseID):
    # (only changes arrays in first row of argument list)
    # # Calculate constant Hybrid scheme coefficients (not taking into account boundary conditions)
    # for i in range(1,nI-1):
    #     for j in range(1,nJ-1):
    #         aE[i,j] = 0 # ADD CODE HERE
    #         aW[i,j] = 0 # ADD CODE HERE
    #         aN[i,j] = 0 # ADD CODE HERE
    #         aS[i,j] = 0 # ADD CODE HERE
            
    # # At outlets (found by velocity out of domain), set homogeneous Neumann
    # for j in range(1,nJ-1):
    #     i = nI-2
    #     # ADD CODE HERE
    #     i = 1
    #     # ADD CODE HERE
    # for i in range(1,nI-1):
    #     j = nJ-2
    #     # ADD CODE HERE
    #     j = 1
    #     # ADD CODE HERE
    
    # # (Homogeneous) Neumann walls (found by zero velocity):
    # for i in range(1,nI-1):
    #     j = nJ-2
    #     # ADD CODE HERE
    #     j = 1
    #     # ADD CODE HERE
    # for j in range(1,nJ-1):
    #     i = nI-2
    #     # ADD CODE HERE
    #     i = 1
    #     # ADD CODE HERE
    
    # for i in range(1,nI-1):
    #     for j in range(1,nJ-1):       
    #         aP[i,j] = 0 # ADD CODE HERE
    """
    --------------------------------
    # ADDED CODE
    --------------------------------
    """
    # Calculate constant Hybrid scheme coefficients (not taking into account boundary conditions)
    for i in range(1,nI-1):
        for j in range(1,nJ-1):
            aE[i,j] = 0 
            aW[i,j] = 0 
            aN[i,j] = 0
            aS[i,j] = 0
            
    # At outlets (found by velocity out of domain), set homogeneous Neumann
    for j in range(1,nJ-1):
        i = nI-2

        i = 1
        
    for i in range(1,nI-1):
        j = nJ-2
        
        j = 1
        
    
    # (Homogeneous) Neumann walls (found by zero velocity):
    for i in range(1,nI-1):
        j = nJ-2
        
        j = 1
        
    for j in range(1,nJ-1):
        i = nI-2
        
        i = 1
        
    
    for i in range(1,nI-1):
        for j in range(1,nJ-1):       
            aP[i,j] = 0 

def solveGaussSeidel(phi,
                     nI, nJ, aE, aW, aN, aS, aP, Su, nLinSolIter):
    # Implement the Gauss-Seidel solver for general variable phi,
    # so it can be reused for all variables.
    # Do it in two directions
    # Only change arrays in first row of argument list!
    # ADD CODE HERE
    # for linSolIter in range(nLinSolIter):   
    #     for i in range(1,nI-1):
    #         for j in range(1,nJ-1):
    #             pass # ADD CODE HERE
    #     for j in range(1,nJ-1):
    #         for i in range(1,nI-1):
    #             pass # ADD CODE HERE
    """
    --------------------------------
    # ADDED CODE
    --------------------------------
    """
    for linSolIter in range(nLinSolIter):  
         
        for i in range(1,nI-1):
            for j in range(1,nJ-1):
                pass 
            
        for j in range(1,nJ-1):
            for i in range(1,nI-1):
                pass 

def solveTDMA(phi, P, Q,
              nI, nJ, aE, aW, aN, aS, aP, Su, nLinSolIter):
    # Implement the Gauss-Seidel solver for general variable phi,
    # so it can be reused for all variables.
    # Do it in two directions
    # Only change arrays in first row of argument list!
    # ADD CODE HERE
    # for linSolIter in range(0,nLinSolIter):
    #     # March from west to east
    #     # Sweep from south to north
    #     for j in range(1,nJ-1):
    #         # ADD CODE HERE

    #         for i in range(2,nI-2):
    #             pass # ADD CODE HERE
                
    #         pass# ADD CODE HERE
            
    #         for i in reversed(range(1,nI-1)):
    #             pass # ADD CODE HERE
            
    #     # March from north to south
    #     # Sweep from west to east 
    #     for i in range(1,nI-1):
    #         pass # ADD CODE HERE
            
    #         for j in range(2,nJ-2):
    #             pass # ADD CODE HERE
                
    #         pass # ADD CODE HERE
            
    #         for j in reversed(range(1,nJ-1)):
    #             pass # ADD CODE HERE
    """
    --------------------------------
    # ADDED CODE
    --------------------------------
    """
    for linSolIter in range(0,nLinSolIter):
        # March from west to east
        # Sweep from south to north
        for j in range(1,nJ-1):
            

            for i in range(2,nI-2):
                pass 
                
            pass
            
            for i in reversed(range(1,nI-1)):
                pass 
            
        # March from north to south
        # Sweep from west to east 
        for i in range(1,nI-1):
            pass 
            
            for j in range(2,nJ-2):
                pass 
                
            pass 
            
            for j in reversed(range(1,nJ-1)):
                pass 

def correctBoundaries(T,
                      nI, nJ, q_wall, k, dx_PE, dx_WP, dy_PN, dy_SP,
                      u, v, nodeX, nodeY, L, H, caseID):
    # Only change arrays in first row of argument list!

    # Copy T to walls where (non-)homogeneous Neumann is applied
    # Note that specified heat flux is positive INTO computational domain!
    # ADD CODE HERE
    
    # Copy T to outlets (where homogeneous Neumann should always be applied):
    # ADD CODE HERE
    
    """
    --------------------------------
    # ADDED CODE
    --------------------------------
    """


    """
    ################################
    # DO NOT CHANGE ANYTHING BELOW!
    ################################
    """
    # Set cornerpoint values to average of neighbouring boundary points
    T[0,0]   = 0.5*(T[1,0]+T[0,1])     # DO NOT CHANGE
    T[-1,0]  = 0.5*(T[-2,0]+T[-1,1])   # DO NOT CHANGE
    T[0,-1]  = 0.5*(T[1,-1]+T[0,-2])   # DO NOT CHANGE
    T[-1,-1] = 0.5*(T[-2,-1]+T[-1,-2]) # DO NOT CHANGE

def calcNormalizedResiduals(res,
                            nI, nJ, explCorrIter, T,
                            aP, aE, aW, aN, aS, Su, Sp):
    # # Compute and print residuals (taking into account normalization):
    # # Non-normalized residual:
    # r0 = 1.0 # ADD CODE HERE

    # # Append residual at present iteration to list of all residuals, for plotting:
    # res.append(r0)
    
    # print('iteration: %5d, res = %.5e' % (explCorrIter, res[-1]/res[0]))
    
    """
    --------------------------------
    # ADDED CODE
    --------------------------------
    """
    # Compute and print residuals (taking into account normalization):
    # Non-normalized residual:
    r0 = 1.0

    # Append residual at present iteration to list of all residuals, for plotting:
    res.append(r0)
    
    print('iteration: %5d, res = %.5e' % (explCorrIter, res[-1]/res[0]))

def probe(nodeX, nodeY, T, probeX, probeY, method='linear'):
    # method (str): interpolation method ('linear', 'nearest', 'cubic')
    """
    ################################
    # DO NOT CHANGE ANYTHING HERE! #
    ################################
    """
    from scipy.interpolate import griddata
    # Flatten the grid for griddata
    points = np.column_stack((nodeX.ravel(), nodeY.ravel()))
    values = T.ravel()
    # Combine probe coordinates into (M, 2) array
    probes = np.column_stack((probeX, probeY))
    # Perform interpolation
    probe = griddata(points, values, probes, method=method)
    return probe

def createDefaultPlots(
                       nI, nJ, pointX, pointY, nodeX, nodeY,
                       dx_WP, dx_PE, dy_SP, dy_PN, Fe, Fw, Fn, Fs,
                       aE, aW, aN, aS, L, H, T, u, v, k,
                       explCorrIter, res, grid_type, caseID):
    """
    ################################
    # DO NOT CHANGE ANYTHING HERE! #
    ################################
    """
    # (Do not change any input arrays!)
    if not os.path.isdir('Figures'):
        os.makedirs('Figures')

    nan = float("nan")
    
    # Plot mesh
    plt.figure()
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.title('Computational mesh \n (Corner nodes only needed for visualization)')
    plt.axis('equal')
    plt.vlines(pointX[:,0],pointY[0,0],pointY[0,-1],colors = 'k',linestyles = 'dashed')
    plt.hlines(pointY[0,:],pointX[0,0],pointX[-1,0],colors = 'k',linestyles = 'dashed')
    plt.plot(nodeX, nodeY, 'ro')
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
    
    # Plot temperature contour
    plt.figure()
    # plt.contourf(nodeX.T, nodeY.T, T.T)
    tempmap=plt.contourf(nodeX.T,nodeY.T,T.T,cmap='coolwarm',levels=30)
    cbar=plt.colorbar(tempmap)
    cbar.set_label('Temperature $[K]$')
    plt.title('Temperature $[K]$')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.axis('equal')
    plt.tight_layout()
    plt.show()
    plt.savefig('Figures/Case_'+str(caseID)+'_'+grid_type+'_temperatureDistribution.png')
    
    # Plot heat flux vectors NORMAL TO WALL boundary face centers ONLY (not in corners)
    # Use temperature gradient just inside domain (note difference to set heat flux)
    qX = np.zeros((nI,nJ))*nan # Array for heat flux in x-direction, in nodes
    qY = np.zeros((nI,nJ))*nan # Array for heat flux in y-direction, in nodes
    for j in range(1,nJ-1):
        i = 0
        if u[i,j] == 0 and v[i,j] == 0:
            qX[i,j] = 1 # ADD CODE HERE
            qY[i,j] = 0
        i = nI-1
        if u[i,j] == 0 and v[i,j] == 0:
            qX[i,j] = 1 # ADD CODE HERE
            qY[i,j] = 0
    for i in range(1,nI-1):
        j = 0
        if u[i,j] == 0 and v[i,j] == 0:
            qX[i,j] = 0
            qY[i,j] = 1 # ADD CODE HERE
        j = nJ-1
        if u[i,j] == 0 and v[i,j] == 0:
            qX[i,j] = 0
            qY[i,j] = 1 # ADD CODE HERE
            
    plt.figure()
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.title('Wall-normal heat flux vectors\n (from internal temperature gradient)')
    plt.gca().set_aspect('equal', adjustable='box')
    tempmap=plt.contourf(nodeX.T,nodeY.T,T.T,cmap='coolwarm',levels=30)
    cbar=plt.colorbar(tempmap)
    cbar.set_label('Temperature $[K]$')
    plt.quiver(nodeX, nodeY, qX, qY, color="black")
    plt.xlim(-0.5*L, 3/2*L)
    plt.ylim(-0.5*H, 3/2*H)
    plt.tight_layout()
    plt.show()
    plt.savefig('Figures/Case_'+str(caseID)+'_'+grid_type+'_wallHeatFlux.png')
    
    # Plot residual convergence
    plt.figure()
    plt.title('Residual convergence')
    plt.xlabel('Iterations')
    plt.ylabel('Residual [-]')
    resLength = np.arange(0,len(res),1)
    normalized = [x / res[0] for x in res]
    plt.plot(resLength, normalized)
    plt.grid()
    plt.yscale('log')
    plt.show()
    plt.savefig('Figures/Case_'+str(caseID)+'_'+grid_type+'_residualConvergence.png')    

def createTimeEvolutionPlots(
                             probeX, probeY, probeValues, caseID, grid_type):
    # Convert list of arrays to a 2D array: shape (n_steps, n_probes)
    data = np.vstack(probeValues)  # rows = time steps, columns = probe points
    n_steps, n_probes = data.shape
    # Plot evolution for each probe point
    plt.figure()
    for i in range(n_probes):
        plt.plot(range(1, n_steps+1), data[:, i], label=f'Probe {i+1} ({probeX[i]}, {probeY[i]})')
    plt.xlabel('Time Step')
    plt.ylabel('Interpolated Value')
    plt.title('Evolution of Probe Values Over Time')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    plt.savefig('Figures/Case_'+str(caseID)+'_'+grid_type+'_timeEvolution.png')

def createAnimatedPlots(
                       nodeX, nodeY, savedT):
    # Create animated plot
    from matplotlib.animation import FuncAnimation
    fig, ax = plt.subplots()
    # Compute global min and max for consistent color scale
    vmin = min(arr.min() for arr in savedT)
    vmax = max(arr.max() for arr in savedT)
    # Initial contour plot
    tempmap = ax.contourf(nodeX.T, nodeY.T, savedT[0].T,
                          cmap='coolwarm', levels=30, vmin=vmin, vmax=vmax)
    # Add colorbar once
    cbar = fig.colorbar(tempmap, ax=ax)
    cbar.set_label('Temperature [K]')
    # Set static labels and aspect ratio
    ax.set_title('Temperature [K]')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_aspect('equal')
    fig.tight_layout()
    # Update function: redraw contour
    def update(frame):
        ax.clear()  # Clear axis completely
        # Redraw contour for current frame
        ax.contourf(nodeX.T, nodeY.T, savedT[frame].T,
                    cmap='coolwarm', levels=30, vmin=vmin, vmax=vmax)
        # Reset labels and aspect ratio after clearing
        ax.set_title('Temperature [K]')
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.set_aspect('equal')
        return []  # Nothing to return for blit=False
    # Create animation with looping
    ani = FuncAnimation(fig, update, frames=len(savedT), interval=100,
                        blit=False, repeat=True)
    # plt.show()
    # Save as GIF (works without ffmpeg)
    ani.save('animated_contour.gif', writer='pillow')

def createAdditionalPlots():
    pass
#%%