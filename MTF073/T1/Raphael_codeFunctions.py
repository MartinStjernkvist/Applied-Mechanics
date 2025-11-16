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
import math  # Only used for mesh example 1
import os  # For saving plots


def createEquidistantMesh(pointX, pointY,
                          mI, mJ, L, H):
    ################################
    # DO NOT CHANGE ANYTHING HERE! #
    ################################
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
            pointX[i, j] = i * L / (mI - 1)
            pointY[i, j] = j * H / (mJ - 1)


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
    growing_rate = 1.2  # growing rate for non-equidistant mesh
    tangens_growing_rate = np.tanh(growing_rate)
    for i in range(0, mI):
        s = (i) / (mI - 1)
        pointX[i, :] = (np.tanh(growing_rate * s) / tangens_growing_rate) * L
    for j in range(0, mJ):
        s = (2 * (j + 1) - mJ - 1) / (mJ - 1)
        pointY[:, j] = (1 + np.tanh(growing_rate * s) / tangens_growing_rate) * 0.5 * H
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


def calcNodePositions(nodeX, nodeY,
                      nI, nJ, pointX, pointY):
    ################################
    # DO NOT CHANGE ANYTHING HERE! #
    ################################
    # Only changes arrays in first row of argument list!
    # Calculates node coordinates.
    # Same for equidistant and non-equidistant meshes.
    # Internal nodes:
    for i in range(0, nI):
        for j in range(0, nJ):
            if i > 0 and i < nI - 1:
                nodeX[i, j] = 0.5 * (pointX[i, 0] + pointX[i - 1, 0])
            if j > 0 and j < nJ - 1:
                nodeY[i, j] = 0.5 * (pointY[0, j] + pointY[0, j - 1])
    # Boundary nodes:
    nodeX[0, :] = pointX[0, 0]  # Note: corner points only needed for contour plot
    nodeY[:, 0] = pointY[0, 0]  # Note: corner points only needed for contour plot
    nodeX[-1, :] = pointX[-1, 0]  # Note: corner points only needed for contour plot
    nodeY[:, -1] = pointY[0, -1]  # Note: corner points only needed for contour plot


def calcDistances(dx_PE, dx_WP, dy_PN, dy_SP, dx_we, dy_sn, \
                  nI, nJ, nodeX, nodeY, pointX, pointY):
    # Calculate distances in first line of argument list.
    # Only change arrays in first row of argument list!
    # Keep 'nan' where values are not needed!
    for i in range(1, nI - 1):
        for j in range(1, nJ - 1):
            #centres
            dx_PE[i, j] = nodeX[i+1,j] - nodeX[i,j] # P->E
            dx_WP[i, j] = nodeX[i,j] - nodeX[i-1,j] # W->P
            dy_PN[i, j] = nodeY[i,j+1] - nodeY[i,j] # P->N
            dy_SP[i, j] = nodeY[i,j] - nodeY[i,j-1] # S->P
            #sides
            dx_we[i, j] = pointX[i,0] - pointX[i-1,0] # Length W-E
            dy_sn[i, j] = pointY[0,j] - pointY[0,j-1] # Length S-N


def calcInterpolationFactors(fxe, fxw, fyn, fys, \
                             nI, nJ, dx_PE, dx_WP, dy_PN, dy_SP, dx_we, dy_sn):
    # Calculate interpolation factors in first row of argument list.
    # Only change arrays in first row of argument list!
    # Keep 'nan' where values are not needed!
    for i in range(1, nI - 1):
        for j in range(1, nJ - 1):
            fxe[i, j] = 0.5*dx_we[i,j] / dx_PE[i,j] # P->face e / P->E
            fxw[i, j] = 0.5*dx_we[i,j] / dx_WP[i,j] # face w->P  / W->P
            fyn[i, j] = 0.5*dy_sn[i,j] / dy_PN[i,j] # P->face n / P->N
            fys[i, j] = 0.5*dy_sn[i,j] / dy_SP[i,j] # face s->P  / S->P


def initArray(T):
    ################################
    # DO NOT CHANGE ANYTHING HERE! #
    ################################
    # Initialize dependent variable array
    # Only change arrays in first row of argument list!
    # Note that a value is needed in all nodes for contour plot
    T[:, :] = 0


def setDirichletBCs(T, \
                    nI, nJ, L, H, nodeX, nodeY, caseID):
    # Set Dirichlet boundary conditions according to your case
    # Only change arrays in first row of argument list!
    # Note that a value is needed in all nodes for contour plot
    # Note: caseID is used only for testing.
    if caseID == 8:
        # boundary 2: x = L
        i_right=nI-1
        for j in range(nJ):
            y=nodeY[i_right,j]
            if np.isnan(y):
                continue
            T2=5.0*(y/H-1)+15*math.cos(math.pi*y/H)
            T[i_right,j]=T2
        # boundary 4: x = 0
        i_left=0
        for j in range(nJ):
            if np.isnan(nodeY[i_left,j]):
                continue
            T[i_left,j]=15 #T4
    # CODE DONE


def updateConductivityArrays(k, k_e, k_w, k_n, k_s, \
                             nI, nJ, nodeX, nodeY, fxe, fxw, fyn, fys, L, H, T, caseID):
    # Update conductivity arrays according to your case
    # Only change arrays in first row of argument list!
    # Keep 'nan' where values are not needed!
    # Note: caseID is used only for testing.

    if caseID == 8:
        for i in range(nI):
            for j in range(nJ):
                x = nodeX[i, j]
                y = nodeY[i, j]

                if np.isnan(x) or np.isnan(y):
                    continue

                if (0.7 < x < 1.1) and (0.3 < y < 0.4):
                    k[i, j] = 0.01
                else:
                    k[i, j] = 20

        for i in range(1, nI - 1):
            for j in range(1, nJ - 1):
                k_e[i,j]=(1.0-fxe[i,j])*k[i,j]+fxe[i,j]*k[i+1,j] # Between P=(i,j) and E=(i+1,j)
                k_w[i,j]=(1.0-fxw[i,j])*k[i-1,j]+fxw[i,j]*k[i,j] # Between W=(i-1,j) and P=(i,j)
                k_n[i,j]=(1.0-fyn[i,j])*k[i,j]+fyn[i,j]*k[i,j+1] # Between P=(i,j) and N=(i,j+1)
                k_s[i,j]=(1.0-fys[i,j])*k[i,j-1]+fys[i,j]*k[i,j] # Between S=(i,j-1) and P=(i,j)
    # CODE DONE


def updateSourceTerms(Su, Sp, \
                      nI, nJ, dx_we, dy_sn, dx_WP, dx_PE, dy_SP, dy_PN, \
                      T, k_w, k_e, k_s, k_n, h, T_inf, caseID):
    # Update source terms according to your case
    # Only change arrays in first row of argument list!
    # Keep 'nan' where values are not needed!
    # Note: caseID is used only for testing.
    for i in range(1, nI - 1):
        for j in range(1, nJ - 1):
            V = dx_we[i,j]*dy_sn[i,j]
            b = 0.0
            Su[i,j]=-b*V
            Sp[i,j]=0.0
    # Comment this line when you have added your code!


def calcCoeffs(aE, aW, aN, aS, aP, \
               nI, nJ, k_w, k_e, k_s, k_n, \
               dy_sn, dx_we, dx_WP, dx_PE, dy_SP, dy_PN, Sp, caseID):
    # Calculate coefficients according to your case
    # Only change arrays in first row of argument list!
    # Keep 'nan' where values are not needed!
    # Note: caseID is used only for testing.
    # Inner node neighbour coefficients:
    # (not caring about special treatment at boundaries):
    for i in range(1, nI - 1):
        for j in range(1, nJ - 1):
            aE[i, j] = k_e[i,j] * dy_sn[i,j] / dx_PE[i,j]
            aW[i, j] = k_w[i,j] * dy_sn[i,j] / dx_WP[i,j]
            aN[i, j] = k_n[i,j] * dx_we[i,j] / dy_PN[i,j]
            aS[i, j] = k_s[i,j] * dx_we[i,j] / dy_SP[i,j]
    # Modifications of aE and aW inside east and west boundaries:
    # ADD CODE HERE IF NECESSARY
    # Modifications of aN and aS inside north and south boundaries:
    # ADD CODE HERE IF NECESSARY

    # Inner node central coefficients:
    for i in range(1, nI - 1):
        for j in range(1, nJ - 1):
            aP[i, j] = aE[i,j] + aW[i,j] + aN[i,j] + aS[i,j] - Sp[i,j]


def solveGaussSeidel(phi, \
                     nI, nJ, aE, aW, aN, aS, aP, Su, nLinSolIter_phi):
    # Implement the Gauss-Seidel solver for general variable phi,
    # so it can be reused for any variable.
    # Do it only in one direction.
    # Only change arrays in first row of argument list!
    for linSolIter in range(nLinSolIter_phi):
        for i in range(1, nI - 1):
            for j in range(1, nJ - 1):
                phi[i, j] = (aE[i,j]*phi[i+1,j]+aW[i,j]*phi[i-1,j]+aN[i,j]*phi[i,j+1]+aS[i,j]*phi[i,j-1]+Su[i,j]) / aP[i,j]


def correctBoundaries(T, \
                      nI, nJ, k_w, k_e, k_s, k_n, \
                      dy_sn, dx_we, dx_WP, dx_PE, dy_SP, dy_PN,
                      h, T_inf, caseID):
    # Copy T to boundaries (and corners) where homegeneous Neumann is applied
    # Only change arrays in first row of argument list!
    if caseID == 8:
        j_top = nJ - 1
        j_int = nJ - 2

        for i in range(1, nI - 1):
            T[i,j_top] = T[i,j_int]
    # CODE DONE


def calcNormalizedResiduals(res, glob_imbal_plot, \
                            nI, nJ, explCorrIter, T, \
                            aP, aE, aW, aN, aS, Su, Sp):
    # Calculate and print normalized residuals, and sane
    # Only change arrays in first row of argument list!
    # Normalize as shown in lecture notes, using:
    #   Din: Diffusive heat rate into the domain
    #   Dout: Diffusive heat rate out of the domain
    #   Sin: Source heat rate into the domain
    #   Sout: Source heat rate out of the domain
    # Non-normalized residual:
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
    # Calculate normalization factor as
    F = 0.0
    for i in range(1, nI - 1):
        for j in range(1, nJ - 1):
            F += abs(aP[i, j] * T[i, j])

    if F <= 1e-30:
        F = 1.0
    # Calculate normalized residual:
    r = r0 / F
    # Append residual at present iteration to list of all residuals, for plotting:
    res.append(r)
    print('iteration: %5d, res = %.5e' % (explCorrIter, r))

    # Calculate the global imbalance as
    # glob_imbal = abs((Din - Dout + Sin - Sout)/(Din + Sin))
    # glob_imbal_plot.append(glob_imbal)
    glob_imbal_plot.append(r)  # CODE DONE


def createDefaultPlots( \
        nI, nJ, pointX, pointY, nodeX, nodeY, \
        L, H, T, k, \
        explCorrIter, res, glob_imbal_plot, caseID):
    ################################
    # DO NOT CHANGE ANYTHING HERE! #
    ################################
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
    plt.vlines(pointX[:, 0], 0, H, colors='k', linestyles='dashed')
    plt.hlines(pointY[0, :], 0, L, colors='k', linestyles='dashed')
    plt.plot(nodeX, nodeY, 'ro')
    plt.tight_layout()
    plt.show()
    plt.savefig('Figures/Case_' + str(caseID) + '_mesh.png')

    # Plot temperature contour
    plt.figure()
    plt.title('Temperature distribution')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.axis('equal')
    tempmap = plt.contourf(nodeX.T, nodeY.T, T.T, cmap='coolwarm', levels=30)
    cbar = plt.colorbar(tempmap)
    cbar.set_label('Temperature [K]')
    plt.tight_layout()
    plt.show()
    plt.savefig('Figures/Case_' + str(caseID) + '_temperatureDistribution.png')

    # Plot residual convergence
    plt.figure()
    plt.title('Residual convergence')
    plt.xlabel('Iterations')
    plt.ylabel('Residual [-]')
    resLength = np.arange(0, len(res), 1)
    plt.plot(resLength, res)
    plt.grid()
    plt.yscale('log')
    plt.show()
    plt.savefig('Figures/Case_' + str(caseID) + '_residualConvergence.png')

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
    tempmap = plt.contourf(nodeX.T, nodeY.T, T.T, cmap='coolwarm', levels=30)
    cbar = plt.colorbar(tempmap)
    cbar.set_label('Temperature [K]')
    plt.quiver(nodeX, nodeY, qX, qY, color="black")
    plt.xlim(-0.2 * L, 1.2 * L)
    plt.ylim(-0.2 * H, 1.2 * H)
    plt.tight_layout()
    plt.show()
    plt.savefig('Figures/Case_' + str(caseID) + '_heatFlux.png')

    # Plot heat flux vectors NORMAL TO WALL boundary face centers ONLY (not in corners)
    # Use temperature gradient just inside domain (note difference to set heat flux)
    qX = np.zeros((nI, nJ)) * nan  # Array for heat flux in x-direction, in nodes
    qY = np.zeros((nI, nJ)) * nan  # Array for heat flux in y-direction, in nodes
    for j in range(1, nJ - 1):
        i = 0
        dTdx_left = (T[i+1,j] - T[i,j]) / (nodeX[i+1,j] - nodeX[i,j])
        qX[i, j] = -k[i,j] * dTdx_left
        qY[i, j] = 0.0
        i = nI - 1
        dTdx_right = (T[i,j] - T[i-1,j]) / (nodeX[i,j] - nodeX[i-1,j])
        qX[i, j] = -k[i,j] * dTdx_right
        qY[i, j] = 0.0
    for i in range(1, nI - 1):
        j = 0
        dTdy_bottom = (T[i,j+1] - T[i,j]) / (nodeY[i,j+1] - nodeY[i,j])
        qX[i, j] = 0.0
        qY[i, j] = -k[i,j] * dTdy_bottom
        j = nJ - 1
        dTdy_top = (T[i,j] - T[i,j-1]) / (nodeY[i,j] - nodeY[i,j-1])
        qX[i, j] = 0.0
        qY[i, j] = -k[i,j] * dTdy_top

    plt.figure()
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.title('Wall-normal heat flux \n (from internal temperature gradient)')
    plt.gca().set_aspect('equal', adjustable='box')
    tempmap = plt.contourf(nodeX.T, nodeY.T, T.T, cmap='coolwarm', levels=30)
    cbar = plt.colorbar(tempmap)
    cbar.set_label('Temperature [K]')
    plt.quiver(nodeX, nodeY, qX, qY, color="black")
    plt.xlim(-0.2 * L, 1.2 * L)
    plt.ylim(-0.2 * H, 1.2 * H)
    plt.tight_layout()
    plt.show()
    plt.savefig('Figures/Case_' + str(caseID) + '_wallHeatFlux.png')

    # Plot global heat rate imbalance convergence
    plt.figure()
    plt.title('Global heat rate imbalance convergence')
    plt.xlabel('Iterations')
    plt.ylabel('Global heat rate imbalance [-]')
    glob_imbal_plotLength = np.arange(0, len(glob_imbal_plot), 1)
    plt.plot(glob_imbal_plotLength, glob_imbal_plot)
    plt.grid()
    plt.yscale('log')
    plt.show()
    plt.savefig('Figures/Case_' + str(caseID) + '_globalHeatRateImbalanceConvergence.png')


def createAdditionalPlots():
    # ADD CODE HERE IF NECESSARY
    # Also add needed arguments to the function - and then also add those
    # arguments for the same function in the main code.
    # Don't change the values of any arrays supplied as arguments!
    pass  # Comment this line when you have added your code!
