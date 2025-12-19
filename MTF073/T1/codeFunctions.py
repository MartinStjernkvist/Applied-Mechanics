#%%
# This file should not be executed by itself. It only contains the
# functions needed for the main code. Some of the functions are
# pre-coded (marked "DO NOT CHANGE ANYTHING HERE!"), and some of
# the functions you need to implement yourself (marked "ADD CODE HERE").
# You can easily find those strings using Ctrl-f.
#
# MAKE SURE THAT YOU ONLY CHANGE ARRAYS IN THE FIRST ROW OF THE ARGUMENT LISTS!
# DO NOT CHANGE THE ARGUMENT LISTS OR FUNCTION NAMES!
# ... with the exception of function createAdditionalPlots, which is prepared
# for your additional plots and post-processing.
#
# Special notes for functions:
# * Functions generally only have access to the variables supplied as arguments
#   or local variables created in the function.
# * Arrays are "mutable", meaning that if they are supplied as arguments to a
#   function, any change to the array in the function also happens to the
#   original array used when calling the function. This is not the case for
#   scalars, which are "non-mutable". One way to do similar changes of global
#   scalars inside functions is to define them as "global" in the function.
#   This should be avoided if possible, since it is not good coding and can
#   potentially lead to problems.
# * Although an array is supplied as argument to a function, the following
#   creates a NEW local array rather than changing the supplied array:
#       aP = aE + aW + aN + aS - Sp
#   The correct way to change the array in a function is either by looping
#   over all the components, or using:
#       aP[:,:] = aE[:,:] + aW[:,:] + aN[:,:] + aS[:,:] - Sp[:,:]
# * Although an array is supplied as argument to a function, the following
#   creates a NEW local array rather than changing the supplied array:
#       p = p + pp*alphaP
#   The correct way to change the array in a function is either by looping
#   over all the components, or using:
#       p += pp*alphaP
#   or, to be more clear that some of the variables are arrays:
#       p[:,:] += pp[:,:]*alphaP
#   or, also working:
#       p[:,:] = p[:,:] + pp[:,:]*alphaP

# Packages needed
import numpy as np
import matplotlib.pyplot as plt
# Set default font size in plots:
plt.rcParams.update({'font.size': 12})
import math # Only used for mesh example 1
import os # For saving plots

def createEquidistantMesh(pointX, pointY,
                          mI, mJ, L, H):
    """
    ################################
    # DO NOT CHANGE ANYTHING HERE! #
    ################################
    """
    # Only changes arrays in first row of argument list!
    # Calculate mesh point coordinates:
    # Equation for line: yy = kk*xx + mm
    # Use it for yy as position in x or y direction and xx a i or j
    # We here always start at x=y=0, so (for x-direction):
    # x = kk*i
    # Determine kk from end points, as kk = (L-0)/(mI-1)
    # Same for y-direction
    for i in range(0, mI):
        for j in range(0, mJ):
            pointX[i,j] = i*L/(mI - 1)
            pointY[i,j] = j*H/(mJ - 1)

def createNonEquidistantMesh(pointX, pointY,
                             mI, mJ, L, H):
    ########################################
    # ADD CODE HERE - ADAPT FOR YOUR CASE! #
    ########################################
    # Only change arrays in first row of argument list!
    # Below you find some examples of how to implement non-equidistant
    # meshes. None of them might be ideal, and they all have pros and cons.
    # Play with them and modify as you like (and need).
    # Toggle commenting of a set of lines by marking them and pressing Ctrl-1.
    ###########
    # Example 1
    ###########
    # Use a non-linear function that starts at 0 and ends at L or H
    # The shape of the function determines the distribution of points
    # We here use the cos function for two-sided clustering
    # for i in range(0, mI):
    #     for j in range(0, mJ):
    #         pointX[i,j] = -L*(np.cos(math.pi*(i/(mI-1)))-1)/2
    #         pointY[i,j] = -H*(np.cos(math.pi*(j/(mJ-1)))-1)/2
    ###############
    # Example 2
    ###############
    # growing_rate = 1.2 # growing rate for non-equidistant mesh
    # tangens_growing_rate=np.tanh(growing_rate)
    # for i in range(0,mI):
    #     s=(i)/(mI-1)
    #     pointX[i,:]=(np.tanh(growing_rate*s)/tangens_growing_rate)*L
    # for j in range(0,mJ):
    #     s=(2*(j+1)-mJ-1)/(mJ-1)
    #     pointY[:,j]=(1+np.tanh(growing_rate*s)/tangens_growing_rate)*0.5*H
    ###############
    # Example 3
    ###############
    # r = 0.85
    # dx = L*(1-r)/(1-r**(mI-1))
    # dy = H*(1-r)/(1-r**(mJ-1))
    # pointX[0,:] = 0
    # pointY[:,0] = 0
    # for i in range(1, mI):
    #     for j in range(mJ):
    #         pointX[i,j] = pointX[i-1, j] + (r**(i-1)) * dx
    # for j in range(1, mJ):
    #     for i in range(mI):
    #         pointY[i,j] = pointY[i, j-1] + (r**(j-1)) * dy
    ###############
    # Example 4
    ###############
    # procent_increase = 1.15
    # inc = 1 / procent_increase
    # dx_mid = 1 / mI
    # dx = np.zeros((mI, 1))
    # for i in range(0, mI):
    #     if i < (mI - 1) / 2 or i == (mI - 1) / 2:
    #         dx[i] = dx_mid * inc ** (mI - i)
    #         dx[-1 - i] = dx_mid * inc ** (mI - i)
    #     for j in range(0, mJ):
    #         pointY[i, j] = j * H / (mJ - 1)
    # for i, d_x in enumerate(dx):
    #     pointX[i, :] = np.sum(dx[0:i])
    # pointX = (pointX / pointX[-1, 0]) * L
    ###############
    # Example 5
    ###############
    # First and second value in linspace must add to 2 (any combination works)
    # dx = np.linspace(1.7, 0.3, mI + 1) * L / (mI - 1)
    # dy = np.linspace(0.3, 1.7, mJ + 1) * H / (mJ - 1)
    # pointX = np.zeros((mI, mJ))
    # pointY = np.zeros((mI, mJ))
    # for i in range(mI):
    #     for j in range(mJ):
    #         # For the mesh points
    #         if i > 0: pointX[i, j] = pointX[i - 1, j] + dx[i]
    #         if j > 0: pointY[i, j] = pointY[i, j - 1] + dy[j]
    
    """
    --------------------------------
    # ADDED CODE
    --------------------------------
    """    
    ###############
    # Example 2
    ###############
    # growing_rate = 1.2 # growing rate for non-equidistant mesh
    # tangens_growing_rate=np.tanh(growing_rate)
    # for i in range(0,mI):
    #     s=(i)/(mI-1)
    #     pointX[i,:]=(np.tanh(growing_rate*s)/tangens_growing_rate)*L
    # for j in range(0,mJ):
    #     s=(2*(j+1)-mJ-1)/(mJ-1)
    #     pointY[:,j]=(1+np.tanh(growing_rate*s)/tangens_growing_rate)*0.5*H
        
    ###########
    # MODIFIED Example 1
    ###########
    xi_list = []
    x_list = []
    y_list = []
    x_refinement_list = []
    y_refinement_list = []
    refinement_list = []
    # Use a non-linear function that starts at 0 and ends at L or H
    # The shape of the function determines the distribution of points
    # We here use the cos function for two-sided clustering
    for i in range(0, mI):
        for j in range(0, mJ):
            xi = i / (mI - 1)
            eta = j / (mJ - 1)
            
            # pointX[i,j] = -L*(np.cos(math.pi*(i/(mI-1)))-1)/2
            # pointY[i,j] = -H*(np.cos(math.pi*(j/(mJ-1)))-1)/2
            
            pointX[i,j] = -L * (np.cos(math.pi * xi) - 1) / 2
            # pointY[i,j] = H * (1 - np.cos((math.pi / 2) * eta))
            pointY[i,j] = -H * (np.cos(math.pi * eta) - 1) / 2
            
            pointRefinement = -(np.cos(math.pi * xi) - 1) / 2
            
            xi_list.append(xi)
            x_list.append(xi * L)
            y_list.append(eta * H)
            x_refinement_list.append(pointX[i,j])
            y_refinement_list.append(pointY[i,j])
            refinement_list.append(pointRefinement)
            
    plt.figure()
    plt.scatter(x_list, x_refinement_list, color='green')
    plt.xlabel('x values')
    plt.ylabel('refinement curve')
    plt.savefig('Figures/x_refinement.png')
    plt.show()
    
    plt.figure()
    plt.scatter(y_list, y_refinement_list, color='red')
    plt.xlabel('y values')
    plt.ylabel('refinement curve')
    plt.savefig('Figures/y_refinement.png')
    plt.show()
    
    plt.figure()
    plt.scatter(xi_list, refinement_list, color='purple')
    plt.xlabel('normalized coordinate values for x & y')
    plt.ylabel('refinement curve')
    plt.savefig('Figures/refinement.png')
    plt.show()

    return pointX, pointY

def calcNodePositions(nodeX, nodeY,
                      nI, nJ, pointX, pointY):
    """
    ################################
    # DO NOT CHANGE ANYTHING HERE! #
    ################################
    """
    # Only changes arrays in first row of argument list!
    # Calculates node coordinates.
    # Same for equidistant and non-equidistant meshes.
    # Internal nodes:
    for i in range(0, nI):
        for j in range(0, nJ):
            if i > 0 and i < nI-1:
                nodeX[i,j] = 0.5*(pointX[i,0] + pointX[i-1,0])
            if j > 0 and j < nJ-1:
                nodeY[i,j] = 0.5*(pointY[0,j] + pointY[0,j-1])
    # Boundary nodes:
    nodeX[0,:]  = pointX[0,0]  # Note: corner points only needed for contour plot
    nodeY[:,0]  = pointY[0,0]  # Note: corner points only needed for contour plot
    nodeX[-1,:] = pointX[-1,0] # Note: corner points only needed for contour plot
    nodeY[:,-1] = pointY[0,-1] # Note: corner points only needed for contour plot
    
def calcDistances(dx_PE, dx_WP, dy_PN, dy_SP, dx_we, dy_sn,
                  nI, nJ, nodeX, nodeY, pointX, pointY):
    # Calculate distances in first line of argument list.
    # Only change arrays in first row of argument list!
    # Keep 'nan' where values are not needed!
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
            fxe[i, j] = 0.5 * dx_we[i, j] / dx_PE[i, j] # P->face e / P->E
            fxw[i, j] = 0.5 * dx_we[i, j] / dx_WP[i, j] # face w->P  / W->P
            fyn[i, j] = 0.5 * dy_sn[i, j] / dy_PN[i, j] # P->face n / P->N
            fys[i, j] = 0.5 * dy_sn[i, j] / dy_SP[i, j] # face s->P  / S->P

def initArray(T):
    """
    ################################
    # DO NOT CHANGE ANYTHING HERE! #
    ################################
    """
    # Initialize dependent variable array
    # Only change arrays in first row of argument list!
    # Note that a value is needed in all nodes for contour plot
    T[:,:] = 0

def setDirichletBCs(T,
                    nI, nJ, L, H, nodeX, nodeY, caseID):
    # Set Dirichlet boundary conditions according to your case
    # Only change arrays in first row of argument list!
    # Note that a value is needed in all nodes for contour plot
    # Note: caseID is used only for testing.
    # ADD CODE HERE
    # pass # Comment this line when you have added your code! 
    
    """
    --------------------------------
    # ADDED CODE
    --------------------------------
    """
    if caseID == 8:
        
        # boundary 2: x = L
        i_right = nI - 1
        for j in range(nJ):
            y = nodeY[i_right, j]
            if np.isnan(y):
                continue
            T2 = 5 * (y / H - 1) + 15 * np.cos(np.pi * y / H)
            T[nI - 1, j] = T2
            
        # boundary 4: x = 0
        i_left = 0
        for j in range(nJ):
            y = nodeY[i_left, j]
            if np.isnan(y):
                continue
            T4 = 15
            T[i_left, j] = T4

def updateConductivityArrays(k, k_e, k_w, k_n, k_s,
                             nI, nJ, nodeX, nodeY, fxe, fxw, fyn, fys, L, H, T, caseID):
    # Update conductivity arrays according to your case
    # Only change arrays in first row of argument list!
    # Keep 'nan' where values are not needed!
    # Note: caseID is used only for testing.
    # ADD CODE HERE.
    # pass # Comment this line when you have added your code!
    """
    --------------------------------
    # ADDED CODE
    --------------------------------
    """
    x_min, x_max = 0.7, 1.1
    y_min, y_max = 0.3, 0.4     
            
    for i in range(0, nI):
        for j in range(0, nJ):
            if x_min < nodeX[i, j] < x_max and y_min < nodeY[i, j] < y_max:
                k[i, j] = 0.01
            else:
                k[i, j] = 20
            
    for i in range(1, nI - 1):
        for j in range(1, nJ - 1):
            # East face
            k_e[i, j] = fxe[i, j] * k[i + 1, j] + (1 - fxe[i, j]) * k[i, j]
            
            # West face
            k_w[i, j] = fxw[i, j] * k[i - 1, j] + (1 - fxw[i, j]) * k[i, j]
            
            # North face
            k_n[i, j] = fyn[i, j] * k[i, j + 1] + (1 - fyn[i, j]) * k[i, j]
            
            # South face
            k_s[i, j] = fys[i, j] * k[i, j - 1] + (1 - fys[i, j]) * k[i, j]

def updateSourceTerms(Su, Sp,
                      nI, nJ, dx_we, dy_sn, dx_WP, dx_PE, dy_SP, dy_PN, \
                      T, k_w, k_e, k_s, k_n, h, T_inf, caseID):
    # Update source terms according to your case
    # Only change arrays in first row of argument list!
    # Keep 'nan' where values are not needed!
    # Note: caseID is used only for testing.
    # ADD CODE HERE.
    # pass # Comment this line when you have added your code!

    """
    --------------------------------
    # ADDED CODE
    --------------------------------
    """
    for i in range(1,nI-1):
        for j in range(1,nJ-1):
            
            A = dx_we
            # See page 28, Ch.4
            if j == 1:
                Sp[i, j] = - (h * k_s[i, j] * A[i, j] **2 / dy_SP[i, j]) / (h * A[i, j] + k_s[i, j] * A[i, j] / dy_SP[i, j])
                
                Su[i, j] = - T_inf * ((h * A[i, j])**2 / (h * A[i, j] + k_s[i, j] * A[i, j] / dy_SP[i, j]) - h * A[i, j])
                
            else: 
                Sp[i, j] = 0
                Su[i, j] = 0
    
def calcCoeffs(aE, aW, aN, aS, aP,
               nI, nJ, k_w, k_e, k_s, k_n,
               dy_sn, dx_we, dx_WP, dx_PE, dy_SP, dy_PN, Sp, caseID):
    # Calculate coefficients according to your case
    # Only change arrays in first row of argument list!
    # Keep 'nan' where values are not needed!
    # Note: caseID is used only for testing.
    # Inner node neighbour coefficients:
    # (not caring about special treatment at boundaries):
    for i in range(1,nI-1):
        for j in range(1,nJ-1):
            # aE[i,j] = 0 # ADD CODE HERE
            # aW[i,j] = 0 # ADD CODE HERE
            # aN[i,j] = 0 # ADD CODE HERE
            # aS[i,j] = 0 # ADD CODE HERE
            
            """
            --------------------------------
            # ADDED CODE
            --------------------------------
            """
            A_EW = dy_sn
            A_NS = dx_we
            
            aE[i,j] = k_e[i,j] * A_EW[i,j] / dx_PE[i,j]
            aW[i,j] = k_w[i,j] * A_EW[i,j] / dx_WP[i,j]
            aN[i,j] = k_n[i,j] * A_NS[i,j] / dy_PN[i,j]
            aS[i,j] = k_s[i,j] * A_NS[i,j] / dy_SP[i,j]
            
            if j == nJ - 2:
                aN[i,j] = 0
                
            if j == 1:
                aS[i,j] = 0
                
                
    # Modifications of aE and aW inside east and west boundaries:
    # ADD CODE HERE IF NECESSARY
    # Modifications of aN and aS inside north and south boundaries:
    # ADD CODE HERE IF NECESSARY
                
    # Inner node central coefficients:
    for i in range(1,nI-1):
        for j in range(1,nJ-1):
            # aP[i,j] = 0 # ADD CODE HERE
            
            """
            --------------------------------
            # ADDED CODE
            --------------------------------
            """
            if j == nJ-1:
                aP[i,j] = 0
            else:
                aP[i,j] = aE[i,j] + aW[i,j] + aN[i,j] + aS[i,j] - Sp[i,j] # Sp should be 0

def solveGaussSeidel(phi,
                     nI, nJ, aE, aW, aN, aS, aP, Su, nLinSolIter_phi):
    # Implement the Gauss-Seidel solver for general variable phi,
    # so it can be reused for any variable.
    # Do it only in one direction.
    # Only change arrays in first row of argument list!
    for linSolIter in range(nLinSolIter_phi):   
        for i in range(1,nI-1):
            for j in range(1,nJ-1):
                # phi[i,j] = 0 # ADD CODE HERE
                
                """
                --------------------------------
                # ADDED CODE
                --------------------------------
                """
                # See page 13, Ch.4
                phi[i,j] = (aE[i,j] * phi[i + 1, j] + 
                            aW[i,j] * phi[i - 1, j] + 
                            aN[i,j] * phi[i, j + 1] + 
                            aS[i,j] * phi[i, j - 1] + 
                            Su[i,j]) / aP[i,j] 

def correctBoundaries(T,
                      nI, nJ, k_w, k_e, k_s, k_n,
                      dy_sn, dx_we, dx_WP, dx_PE, dy_SP, dy_PN, 
                      h, T_inf, caseID):
    # Copy T to boundaries (and corners) where homegeneous Neumann is applied
    # Only change arrays in first row of argument list!
    # ADD CODE HERE IF NECESSARY
    # pass # Comment this line when you have added your code!
    """
    --------------------------------
    # ADDED CODE
    --------------------------------
    """
    
    A = dx_we
    
    # for i in range(1,nI-1):
    #     for j in range(1,nJ-1):
                
    #             if j == 1:
    #                 # TW = (h * A[i, j] * T_inf + k_s[i, j] * A[i, j] / dy_SP[i, j] * T[i, j]) / (k_s[i, j] * A[i, j] / dy_SP[i, j] + h * A[i, j])
                    
    #                 # T[i, j] = h * A[i, j] * (TW - T_inf)  / (k_s[i, j] * A[i, j]) * dy_SP[i, j] + TW
    #                 pass
    
    j_top = nJ - 1
    j_int = nJ - 2

    for i in range(1, nI - 1):
        T[i, j_top] = T[i, j_int]

    for i in range(1, nI - 1):
        k_face = k_s[i, 1]
        dy = dy_SP[i, 1]

        if np.isnan(k_face) or np.isnan(dy):
            continue

        T_P = T[i, 1]

        T_wall = (k_face * T_P / dy + h * T_inf) / (h + k_face / dy)
        T[i, 0] = T_wall

def calcNormalizedResiduals(res, glob_imbal_plot,
                            nI, nJ, explCorrIter, T, \
                            aP, aE, aW, aN, aS, Su, Sp, F_data=1, T_data=1, file_name="blank"):
    # Calculate and print normalized residuals, and sane 
    # Only change arrays in first row of argument list!
    # Normalize as shown in lecture notes, using:
    #   Din: Diffusive heat rate into the domain
    #   Dout: Diffusive heat rate out of the domain
    #   Sin: Source heat rate into the domain
    #   Sout: Source heat rate out of the domain
    # Non-normalized residual:
    # r0 = 0
    # for i in range(1,nI-1):
    #     for j in range(1,nJ-1):
    #         r0 += 0 # ADD CODE HERE
    # # Calculate normalization factor as
    # # F =  Din + Sin
    # # Calculate normalized residual:
    # r = 1 # ADD CODE HERE
    # # Append residual at present iteration to list of all residuals, for plotting:
    # res.append(r)
    # print('iteration: %5d, res = %.5e' % (explCorrIter, r))

    # # Calculate the global imbalance as
    # # glob_imbal = abs((Din - Dout + Sin - Sout)/(Din + Sin))
    # # glob_imbal_plot.append(glob_imbal)
    # glob_imbal_plot.append(1) # Comment when you have added your code above!
    
    """
    --------------------------------
    # ADDED CODE
    --------------------------------
    """
    r0 = 0
    for i in range(1, nI - 1):
        for j in range(1, nJ - 1):
            balance = (aP[i, j] * T[i, j] 
                       - (aE[i, j] * T[i + 1, j]
                        + aW[i, j] * T[i - 1, j] 
                        + aN[i, j] * T[i, j + 1]
                        + aS[i, j] * T[i, j - 1] 
                        + Su[i, j]))
            r0 += abs(balance)
            
    # Calculate normalization factor
    Din = 0
    Dout = 0
    Sin = 0
    Sout = 0

    i = nI - 2
    for j in range(1, nJ - 1):
        east = aE[i, j] * (T[i + 1, j] - T[i, j])
        Din += max(east, 0)
        Dout += abs(min(east, 0))
    
    i = 1
    for j in range(1, nJ - 1):
        west = aW[i, j] * (T[i - 1, j] - T[i, j])
        Din += max(west, 0)
        Dout += abs(min(west, 0))
    
    j = nJ - 2
    for i in range(1, nI - 1):
        north = aN[i, j] * (T[i, j + 1] - T[i, j])
        Din += max(north, 0)
        Dout += abs(min(north, 0))
    
    j = 1
    for i in range(1, nI - 1):
        south = aS[i, j] * (T[i, j - 1] - T[i, j])
        Din += max(south, 0)
        Dout += abs(min(south, 0))
    
    for i in range(1, nI - 1):
        for j in range(1, nJ - 1):
            source = Su[i, j] + T[i, j] * Sp[i, j]
            Sin += max(source, 0)
            Sout += abs(min(source, 0))
    
    F = Din + Sin
    print('Din: ', Din)
    print('Sin: ', Sin)
    print('fraction of q_in and q_out:', (Din + Sin) / (Dout + Sout))
    F_data.append(F)
        
    # Calculate normalized residual:
    r = r0 / F
    # Append residual at present iteration to list of all residuals, for plotting:
    res.append(r)
    print('iteration: %5d, res = %.5e' % (explCorrIter, r))
    
    # Calculate the global imbalance
    glob_imbal = abs((Din - Dout + Sin - Sout)/(Din + Sin))
    glob_imbal_plot.append(glob_imbal)
    
    # i_selected = nI-2
    # j_selected = int(nJ / 2)
    i_selected = int(nI / 2)
    j_selected = 1
    T_value = T[i_selected, j_selected]
    T_data.append(T_value)
    
    resLength = np.arange(0,len(res),1)
    
    np.savez(file_name + '.npz', 
         reslen = resLength, 
         T_data=T_data)
    
def createDefaultPlots(
                       nI, nJ, pointX, pointY, nodeX, nodeY,
                       L, H, T, k,
                       explCorrIter, res, glob_imbal_plot, caseID):
    """
    ################################
    # DO NOT CHANGE ANYTHING HERE! #
    ################################
    """
    # Does not change the values of any input arrays!
    if not os.path.isdir('Figures'):
        os.makedirs('Figures')

    nan = float("nan")
    
    # Plot mesh
    plt.figure()
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.title('Computational mesh \n (Corner nodes only needed for visualization)')
    plt.axis('equal')
    plt.vlines(pointX[:,0],0,H,colors = 'k',linestyles = 'dashed')
    plt.hlines(pointY[0,:],0,L,colors = 'k',linestyles = 'dashed')
    plt.plot(nodeX, nodeY, 'ro')
    plt.tight_layout()
    plt.savefig('Figures/Case_'+str(caseID)+'_mesh_results.png')
    plt.show()
    
    # Plot temperature contour
    plt.figure()
    plt.title('Temperature distribution')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.axis('equal')
    tempmap=plt.contourf(nodeX.T,nodeY.T,T.T,cmap='coolwarm',levels=30)
    cbar=plt.colorbar(tempmap)
    cbar.set_label('Temperature [K]')
    plt.tight_layout()
    plt.savefig('Figures/Case_'+str(caseID)+'_temperatureDistribution_results.png')
    plt.show()
    
    # Plot residual convergence
    plt.figure()
    plt.title('Residual convergence')
    plt.xlabel('Iterations')
    plt.ylabel('Residual [-]')
    resLength = np.arange(0,len(res),1)
    plt.plot(resLength, res)
    plt.grid()
    plt.yscale('log')
    plt.savefig('Figures/Case_'+str(caseID)+'_residualConvergence_results.png')
    plt.show()
    
    # Plot heat flux vectors in nodes (not at boundaries)
    # qX = np.zeros((nI,nJ))*nan # Array for heat flux in x-direction, in nodes
    # qY = np.zeros((nI,nJ))*nan # Array for heat flux in y-direction, in nodes
    # for i in range(1,nI-1):
    #     for j in range(1,nJ-1):
                # qX[i,j] = 1 # ADD CODE HERE
                # qY[i,j] = 1 # ADD CODE HERE
                
    """
    --------------------------------
    # ADDED CODE
    --------------------------------
    """
    # Plot heat flux vectors in nodes (not at boundaries)
    qX = np.zeros((nI, nJ)) * nan  # Array for heat flux in x-direction, in nodes
    qY = np.zeros((nI, nJ)) * nan  # Array for heat flux in y-direction, in nodes
    for i in range(1, nI - 1):
        for j in range(1, nJ - 1):
            dTdx = (T[i+1,j] - T[i-1,j]) / (nodeX[i+1,j] - nodeX[i-1,j])
            dTdy = (T[i,j+1] - T[i,j-1]) / (nodeY[i,j+1] - nodeY[i,j-1])

            qX[i, j] = -k[i,j] * dTdx
            qY[i, j] = -k[i,j] * dTdy
                
    plt.figure()
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.title('Heat flux')
    plt.gca().set_aspect('equal', adjustable='box')
    tempmap=plt.contourf(nodeX.T,nodeY.T,T.T,cmap='coolwarm',levels=30)
    cbar=plt.colorbar(tempmap)
    cbar.set_label('Temperature [K]')
    plt.quiver(nodeX, nodeY, qX, qY, color="black")
    plt.xlim(-0.2*L, 1.2*L)
    plt.ylim(-0.2*H, 1.2*H)
    plt.tight_layout()
    plt.savefig('Figures/Case_'+str(caseID)+'_heatFlux_results.png')
    plt.show()
    
    # Plot heat flux vectors NORMAL TO WALL boundary face centers ONLY (not in corners)
    # Use temperature gradient just inside domain (note difference to set heat flux)
    # qX = np.zeros((nI,nJ))*nan # Array for heat flux in x-direction, in nodes
    # qY = np.zeros((nI,nJ))*nan # Array for heat flux in y-direction, in nodes
    # for j in range(1,nJ-1):
    #     i = 0
        # qX[i,j] = 1 # ADD CODE HERE
        # qY[i,j] = 0 # ADD CODE HERE
        # i = nI-1
        # qX[i,j] = 1 # ADD CODE HERE
        # qY[i,j] = 0 # ADD CODE HERE
        
    # for i in range(1,nI-1):
    #     j = 0
        # qX[i,j] = 0 # ADD CODE HERE
        # qY[i,j] = 1 # ADD CODE HERE
        # j = nJ-1
        # qX[i,j] = 0 # ADD CODE HERE
        # qY[i,j] = 1 # ADD CODE HERE
        
    """
    --------------------------------
    # ADDED CODE
    --------------------------------
    """
        
        # Plot heat flux vectors NORMAL TO WALL boundary face centers ONLY (not in corners)
    # Use temperature gradient just inside domain (note difference to set heat flux)
    qX = np.zeros((nI, nJ)) * nan  # Array for heat flux in x-direction, in nodes
    qY = np.zeros((nI, nJ)) * nan  # Array for heat flux in y-direction, in nodes
    
    for j in range(1, nJ - 1):
        
        i = 0
        dTdx_left = (T[i + 1, j] - T[i, j]) / (nodeX[i + 1, j] - nodeX[i, j])
        qX[i, j] = -k[i, j] * dTdx_left
        qY[i, j] = 0.0
        
        i = nI - 1
        dTdx_right = (T[i, j] - T[i - 1, j]) / (nodeX[i, j] - nodeX[i - 1, j])
        qX[i, j] = -k[i, j] * dTdx_right
        qY[i, j] = 0.0
        
    for i in range(1, nI - 1):
        
        j = 0
        dTdy_bottom = (T[i, j + 1] - T[i, j]) / (nodeY[i, j + 1] - nodeY[i, j])
        qX[i, j] = 0.0
        qY[i, j] = -k[i, j] * dTdy_bottom
        
        j = nJ - 1
        dTdy_top = (T[i, j] - T[i, j - 1]) / (nodeY[i, j] - nodeY[i, j - 1])
        qX[i, j] = 0.0
        qY[i, j] = -k[i, j] * dTdy_top
        
    plt.figure()
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.title('Wall-normal heat flux \n (from internal temperature gradient)')
    plt.gca().set_aspect('equal', adjustable='box')
    tempmap=plt.contourf(nodeX.T,nodeY.T,T.T,cmap='coolwarm',levels=30)
    cbar=plt.colorbar(tempmap)
    cbar.set_label('Temperature [K]')
    plt.quiver(nodeX, nodeY, qX, qY, color="black")
    plt.xlim(-0.2*L, 1.2*L)
    plt.ylim(-0.2*H, 1.2*H)
    plt.tight_layout()
    plt.savefig('Figures/Case_'+str(caseID)+'_wallHeatFlux_results.png')
    plt.show()

    # Plot global heat rate imbalance convergence
    plt.figure()
    plt.title('Global heat rate imbalance convergence')
    plt.xlabel('Iterations')
    plt.ylabel('Global heat rate imbalance [-]')
    glob_imbal_plotLength = np.arange(0,len(glob_imbal_plot),1)
    plt.plot(glob_imbal_plotLength, glob_imbal_plot)
    plt.grid()
    plt.yscale('log')
    plt.savefig('Figures/Case_'+str(caseID)+'_globalHeatRateImbalanceConvergence_results.png')
    plt.show()

def createAdditionalPlots(nI, nJ, pointX, pointY, nodeX, nodeY,
                       L, H, T, k,
                       explCorrIter, res, glob_imbal_plot, caseID, F_data, T_data):
    # ADD CODE HERE IF NECESSARY
    # Also add needed arguments to the function - and then also add those
    # arguments for the same function in the main code.
    # Don't change the values of any arrays supplied as arguments!
    # pass # Comment this line when you have added your code!
    """
    --------------------------------
    # ADDED CODE
    --------------------------------
    """
    # Plot F
    plt.figure()
    plt.title('F vs iterations')
    plt.xlabel('Iterations')
    plt.ylabel('F')
    resLength = np.arange(0,len(res),1)
    plt.plot(resLength, F_data)
    plt.grid()
    # plt.yscale('log')
    plt.savefig('Figures/Case_'+str(caseID)+'_FConvergence_results.png')
    plt.show()
    
    # Plot T
    plt.figure()
    plt.title('Temperature vs iterations')
    plt.xlabel('Iterations')
    plt.ylabel('T')
    resLength = np.arange(0,len(res),1)
    plt.plot(resLength, T_data)
    plt.grid()
    # plt.yscale('log')
    plt.savefig('Figures/Case_'+str(caseID)+'_TemperatureConvergence_results.png')
    plt.show()
    
    # # Compare convergence for different equidistant meshes
    # with np.load('4x4.npz') as data:
    #     resLength_4x4, T_data_4x4 = data['reslen'], data['T_data']
    
    # with np.load('8x8.npz') as data:
    #     resLength_8x8, T_data_8x8 = data['reslen'], data['T_data']
    
    # with np.load('16x16.npz') as data:
    #     resLength_16x16, T_data_16x16 = data['reslen'], data['T_data']
    
    # plt.figure()
    # plt.title('Temperature vs iterations')
    # plt.xlabel('Iterations')
    # plt.ylabel('T')
    
    # plt.plot(resLength_4x4, T_data_4x4, label = "equidistant, 4x4")
    # plt.plot(resLength_8x8, T_data_8x8, label = "equidistant, 8x8")
    # plt.plot(resLength_16x16, T_data_16x16, label = "equidistant, 16x16")
    
    # plt.grid()
    # plt.legend()
    # plt.savefig('Figures/Case_'+str(caseID)+'_TemperatureConvergence_equidistant.png')
    # plt.show()
    
    # # Comparison equidistant vs non-equidistant, iterations
    # plt.figure()
    # plt.title('Temperature vs iterations')
    # plt.xlabel('Iterations')
    # plt.ylabel('T')
    # resLength = np.arange(0,len(res),1)
    # plt.plot(resLength, T_data)
    # plt.grid()
    # # plt.yscale('log')
    # plt.savefig('Figures/Case_'+str(caseID)+'_TemperatureConvergence_results.png')
    # plt.show()
        
    # with np.load('32x32.npz') as data:
    #     resLength_32x32, T_data_32x32 = data['reslen'], data['T_data']
    
    # with np.load('noneq.npz') as data:
    #     resLength_noneq, T_data_noneq = data['reslen'], data['T_data']
    
    # plt.figure()
    # plt.title('Temperature vs iterations')
    # plt.xlabel('Iterations')
    # plt.ylabel('T')
    
    # plt.plot(resLength_32x32, T_data_32x32, label = "equidistant, 32x32")
    # plt.plot(resLength_noneq, T_data_noneq, label = "non-equidistant")
    
    # plt.grid()
    # plt.legend()
    # plt.savefig('Figures/Case_'+str(caseID)+'_TemperatureConvergence_comparison.png')
    # plt.show()
    
    # Comparison equidistant vs non-equidistant, refinement
    plt.figure()
    plt.title('Temperature vs meshgrid')
    plt.xlabel('Mesh sizing (both directions)')
    plt.ylabel('T')
    
    mesh_list = []
    T_data, T_data_noneq = [], []
    
    for i in range(5):
        mesh = 4 * 2**i
        with np.load(str(mesh) + '.npz') as data:
            mesh_list.append(mesh)
            T = data['T_data']
            T_data.append(float(T[-1]))
    
    for i in range(5):
        mesh = 4 * 2**i
        with np.load('non' + str(mesh) + '.npz') as data:
            T = data['T_data']
            T_data_noneq.append(float(T[-1]))
    
    plt.plot(mesh_list, T_data,'o-', label = "equidistant")
    plt.plot(mesh_list, T_data_noneq, 'o-', label = "non-equidistant")
    
    plt.grid()
    plt.legend()
    plt.savefig('Figures/Case_'+str(caseID)+'_TemperatureConvergence_comparison_meshing.png')
    plt.show()
    
#%%