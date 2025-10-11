# Purpose:
# Generates a 2D rectangular mesh
#
# Input:
# p1    - Lower left point of rectangle [x1, y1]
# p2    - Upper right point of rectangle [x2, y2]
# nelx  - Number of elments in x direction
# nely  - Number of elments in y direction
# ndofs - Number of degrees of freedom per node
#
# Output:
# Edof  - Connectivity matrix for mesh, cf. Calfem Toolbox
# Ex    - Elementwise x-coordinates, cf. Calfem Toolbox
# Ey    - Elementwise y-coordinates, cf. Calfem Toolbox
# Bi    - Matrix containing boundary dofs for segment i (i=1,2,3,4)
#         First column -> 1st dofs, second column -> 2nd dofs and so on
#         size = (num boundary nodes on segment) x ndofs
#         B1 = Bottom side     B2 = Right side
#         B3 = Upper side      B4 = Left side
#
import numpy as np


def ex_ey_quadmesh(p1, p2, nelx, nely, ndofs):

    xv = np.linspace(p1[0], p2[0], nelx + 1)
    yv = np.linspace(p1[1], p2[1], nely + 1)

    nel = nelx * nely
    Ex = np.zeros((nel, 4))
    Ey = np.zeros((nel, 4))

    for m in range(0, nely):

        for n in range(0, nelx):

            Ex[n + m * nelx, 0] = xv[n]
            Ex[n + m * nelx, 1] = xv[n + 1]
            Ex[n + m * nelx, 2] = xv[n + 1]
            Ex[n + m * nelx, 3] = xv[n]
            #
            Ey[n + m * nelx, 0] = yv[m]
            Ey[n + m * nelx, 1] = yv[m]
            Ey[n + m * nelx, 2] = yv[m + 1]
            Ey[n + m * nelx, 3] = yv[m + 1]

    # return x- and y-wise coordinates
    return Ex, Ey


def edof_quadmesh(nelx, nely, ndofs):

    Edof = np.zeros((nelx * nely, 4 * ndofs), "i")

    for m in range(0, nely):
        for n in range(0, nelx):

            Edof[n + m * nelx, 0] = n * ndofs + 1 + m * (nelx + 1) * ndofs
            Edof[n + m * nelx, 1] = n * ndofs + 2 + m * (nelx + 1) * ndofs
            Edof[n + m * nelx, 2] = (n + 1) * ndofs + 1 + m * (nelx + 1) * ndofs
            Edof[n + m * nelx, 3] = (n + 1) * ndofs + 2 + m * (nelx + 1) * ndofs
            #
            Edof[n + m * nelx, 4] = (n + 1) * ndofs + 1 + (m + 1) * (nelx + 1) * ndofs
            Edof[n + m * nelx, 5] = (n + 1) * ndofs + 2 + (m + 1) * (nelx + 1) * ndofs
            Edof[n + m * nelx, 6] = n * ndofs + 1 + (m + 1) * (nelx + 1) * ndofs
            Edof[n + m * nelx, 7] = n * ndofs + 2 + (m + 1) * (nelx + 1) * ndofs

    return Edof


def B1B2B3B4_quadmesh(nelx, nely, ndofs):

    # lower boundary, dofs
    B1 = np.linspace(1, (nelx + 1) * ndofs, (nelx + 1) * ndofs)
    B1 = B1.astype(int)

    B2 = np.zeros(((nely + 1) * ndofs), "i")
    nn = 0
    for n in range(0, nely + 1):
        B2[nn] = (nelx + 1) * ndofs * (n + 1) - 1
        if ndofs > 1:
            B2[nn + 1] = (nelx + 1) * ndofs * (n + 1) + 0
        nn = nn + ndofs

    B3 = (
        np.linspace(1, (nelx + 1) * ndofs, (nelx + 1) * ndofs)
        + (nelx + 1) * ndofs * nely
    )
    B3 = B3.astype(int)

    B4 = np.zeros(((nely + 1) * ndofs), "i")
    nn = 0
    for n in range(0, nely + 1):
        B4[nn] = (nelx + 1) * ndofs * n + 1
        if ndofs > 1:
            B4[nn + 1] = (nelx + 1) * ndofs * n + 2
        nn = nn + ndofs

    P1 = np.zeros((2), "i")
    P2 = np.zeros((2), "i")
    P3 = np.zeros((2), "i")
    P4 = np.zeros((2), "i")
    for m in range(0, 2):
        P1[m] = B1[m]
        P2[m] = B2[m]
        P4[m] = B3[m]
    P3[0] = B3[-1] - 1
    P3[1] = B3[-1]

    return B1, B2, B3, B4, P1, P2, P3, P4


def quadmesh(p1, p2, nelx, nely, ndofs):

    Ex, Ey = ex_ey_quadmesh(p1, p2, nelx, nely, ndofs)
    Edof = edof_quadmesh(nelx, nely, ndofs)
    B1, B2, B3, B4, P1, P2, P3, P4 = B1B2B3B4_quadmesh(nelx, nely, ndofs)

    return Ex, Ey, Edof, B1, B2, B3, B4, P1, P2, P3, P4
