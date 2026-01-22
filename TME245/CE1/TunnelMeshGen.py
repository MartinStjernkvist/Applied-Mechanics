# -*- coding: utf-8 -*-
"""
Created on Wed Jan 14 14:43:38 2026

@author: fagmar
"""

import numpy as np


def TunnelMeshGen(H, B, D, b, h, r, Nr, Nt, elemtype):

    '''
Generates a rectangular mesh of half the tunnel
call:
[Edof,Coord,Ex,Ey,LeftSide_nodes,TopSide_nodes,RightSide_nodes,...
BottomSide_nodes,TopLeftElem_no,TopRighty_node,h]=RectangleMeshGen(Lx,Ly,Nt,Nr,elemtype)
Input -------------------------------------------------------------------

H is the domain size in y-direction

B is the domain size in x-direction

D is the distance between bottom of domain and tunnel floor

b is the tunnel (half) width 

h is the tunnel maximum height

r is the radius of corner fillet

Nr is the number of element side lengths from tunnel to domain top/bottom edge, 
i.e. defines the number of elements along the top left and bottom left boundary
(minimum 1)

Nt is the number of element side lengths in tunnel inside
(minimum 5)

Set to 1 for CST elements(triangles) and set as 2 for 4-node bilinear
elements

Output ------------------------------------------------------------------

Edof is the element topology matrix (for 2 dof per node)
Edof[i-1,:] = [elem_i, glob_dof1, glob_dof2, glob_dof3, glob_dof4 ...]

Coord is the global coordinate matrix
Coord[i-1,:] = [x-coord_nodei, y-coord_nodei]

Ex is the element nodal coordinates in x-direction
Ex[i-1,:] = [x-coord_node1_elemi, x-coord_node2_elemi, x-coord_node3_elemi ]

Ey is the element nodal coordinates in y-direction
Ey[i-1,:] = [y-coord_node1_elemi, y-coord_node2_elemi, y-coord_node3_elemi ]

LeftSide_nodes is a list of the nodes on the left boundary
TopSide_nodes is a list of the nodes on the top boundary
RightSide_nodes is a list of the nodes on the right boundary
BottomSide_nodes is a list of the nodes on the bottom boundary
'''

    if H < 0.9 * (h + D) or B < 0.9 * b or b < r or h < r:
        raise ValueError("Inappropriate geometry chosen")

    # Tunnel boundary length
    L_inside = b + (h - r) + np.pi * r / 2 + (b - r)

    # Initial division
    s_start = np.array([
        0,
        b / L_inside,
        (b + (h - r)) / L_inside,
        (b + (h - r) + np.pi * r / 4) / L_inside,
        (b + (h - r) + np.pi * r / 2) / L_inside,
        1.0
    ])

    s = s_start.copy()

    # Divide largest segment
    for _ in range(6, Nt + 1):
        ds = np.diff(s)
        ind = np.argmax(ds)
        s = np.insert(s, ind + 1, 0.5 * (s[ind] + s[ind + 1]))

    s_seg1 = np.where(s < s_start[1])[0]
    s_seg2 = np.where((s > s_start[1]) & (s < s_start[3]))[0]
    s_seg3 = np.where(s > s_start[3])[0]

    X_squares = np.zeros((Nr + 1, Nt + 1))
    Y_squares = np.zeros((Nr + 1, Nt + 1))

    # Segment 1
    for i in s_seg1:
        y1, y2 = D, 0
        x1 = s[i] / s[len(s_seg1)] * b
        x2 = s[i] / s[len(s_seg1)] * B
        X_squares[:, i] = np.linspace(x1, x2, Nr + 1)
        Y_squares[:, i] = np.linspace(y1, y2, Nr + 1)

    X_squares[:, s_seg1[-1] + 1] = np.linspace(b, B, Nr + 1)
    Y_squares[:, s_seg1[-1] + 1] = np.linspace(D, 0, Nr + 1)

    # Segment 2
    for i in s_seg2:
        x2 = B
        y2 = (s[i] - s_start[1]) / (s_start[3] - s_start[1]) * H

        if s[i] < s_start[2]:
            x1 = b
            y1 = (s[i] - s_start[1]) / (s_start[2] - s_start[1]) * (h - r) + D
        else:
            v = (s[i] - s_start[2]) * L_inside / r
            x1 = b - r + r * np.cos(v)
            y1 = (D + h) - r + r * np.sin(v)

        X_squares[:, i] = np.linspace(x1, x2, Nr + 1)
        Y_squares[:, i] = np.linspace(y1, y2, Nr + 1)

    X_squares[:, s_seg2[-1] + 1] = np.linspace(b - (1 - 1 / np.sqrt(2)) * r, B, Nr + 1)
    Y_squares[:, s_seg2[-1] + 1] = np.linspace(D + h - (1 - 1 / np.sqrt(2)) * r, H, Nr + 1)

    # Segment 3
    for i in s_seg3:
        y2 = H
        x2 = (1 - (s[i] - s_start[3]) / (1 - s_start[3])) * B

        if s[i] < s_start[4]:
            v = (s[i] - s_start[2]) * L_inside / r
            x1 = b - r + r * np.cos(v)
            y1 = (D + h) - r + r * np.sin(v)
        else:
            y1 = D + h
            x1 = (1 - (s[i] - s_start[4]) / (1 - s_start[4])) * (b - r)

        X_squares[:, i] = np.linspace(x1, x2, Nr + 1)
        Y_squares[:, i] = np.linspace(y1, y2, Nr + 1)

    # ------------------------------------------------------------------
    # ELEMENT TYPES
    # ------------------------------------------------------------------
    if elemtype == 1:  # Triangles (CST)

        X_mids = 0.25 * (
            X_squares[:-1, :-1] + X_squares[:-1, 1:] +
            X_squares[1:, :-1] + X_squares[1:, 1:]
        )
        Y_mids = 0.25 * (
            Y_squares[:-1, :-1] + Y_squares[:-1, 1:] +
            Y_squares[1:, :-1] + Y_squares[1:, 1:]
        )

        X_coord = X_squares[0, :].copy()
        Y_coord = Y_squares[0, :].copy()

        for k in range(Nr):
            X_coord = np.concatenate([X_coord, X_mids[k, :], X_squares[k + 1, :]])
            Y_coord = np.concatenate([Y_coord, Y_mids[k, :], Y_squares[k + 1, :]])

        Row_nod_top = np.vstack([
            np.arange(0, Nt),
            np.arange(Nt + 1, 2 * Nt + 1),
            np.arange(1, Nt + 1)
        ]).T

        Row_nod_left = np.vstack([
            np.arange(0, Nt),
            np.arange(2 * Nt + 1, 3 * Nt + 1),
            np.arange(Nt + 1, 2 * Nt + 1)
        ]).T

        Row_nod_right = np.vstack([
            np.arange(1, Nt + 1),
            np.arange(Nt + 1, 2 * Nt + 1),
            np.arange(2 * Nt + 2, 3 * Nt + 2)
        ]).T

        Row_nod_bottom = np.vstack([
            np.arange(Nt + 1, 2 * Nt + 1),
            np.arange(2 * Nt + 1, 3 * Nt + 1),
            np.arange(2 * Nt + 2, 3 * Nt + 2)
        ]).T

        Row_nod = np.vstack([Row_nod_top, Row_nod_left, Row_nod_right, Row_nod_bottom])

        Enod = []
        for k in range(Nr):
            Enod.append(Row_nod + k * (2 * Nt + 1))
        Enod = np.vstack(Enod)

        el_numbers = np.arange(1, Enod.shape[0] + 1)
        Enod = np.column_stack([el_numbers, Enod + 1])

        Edof = np.column_stack([
            Enod[:, 0],
            2 * Enod[:, 1] - 1, 2 * Enod[:, 1],
            2 * Enod[:, 2] - 1, 2 * Enod[:, 2],
            2 * Enod[:, 3] - 1, 2 * Enod[:, 3]
        ])


#        Edof = np.column_stack([
#            2 * Enod[:, 1] - 1, 2 * Enod[:, 1],
#            2 * Enod[:, 2] - 1, 2 * Enod[:, 2],
#            2 * Enod[:, 3] - 1, 2 * Enod[:, 3]
#        ])

        Ex = X_coord[Enod[:, 1:] - 1]
        Ey = Y_coord[Enod[:, 1:] - 1]

    elif elemtype == 2:  # Bilinear quads

        X_coord = X_squares.T.reshape(-1)
        Y_coord = Y_squares.T.reshape(-1)

        Row_nod = np.vstack([
            np.arange(0, Nt),
            np.arange(Nt + 1, 2 * Nt + 1),
            np.arange(Nt + 2, 2 * Nt + 2),
            np.arange(1, Nt + 1)
        ]).T

        Enod = []
        for k in range(Nr):
            Enod.append(Row_nod + k * (Nt + 1))
        Enod = np.vstack(Enod)

        el_numbers = np.arange(1, Enod.shape[0] + 1)
        Enod = np.column_stack([el_numbers, Enod + 1])

        Edof = np.column_stack([
            Enod[:, 0],
            2 * Enod[:, 1] - 1, 2 * Enod[:, 1],
            2 * Enod[:, 2] - 1, 2 * Enod[:, 2],
            2 * Enod[:, 3] - 1, 2 * Enod[:, 3],
            2 * Enod[:, 4] - 1, 2 * Enod[:, 4]
        ])
#        Edof = np.column_stack([
#            2 * Enod[:, 1] - 1, 2 * Enod[:, 1],
#            2 * Enod[:, 2] - 1, 2 * Enod[:, 2],
#            2 * Enod[:, 3] - 1, 2 * Enod[:, 3],
#            2 * Enod[:, 4] - 1, 2 * Enod[:, 4]
#        ])

        Ex = X_coord[Enod[:, 1:] - 1]
        Ey = Y_coord[Enod[:, 1:] - 1]

    else:
        raise ValueError("elemtype must be 1 (triangles) or 2 (quads)")

    Coord = np.column_stack([X_coord, Y_coord])

    eps = np.finfo(float).eps
    LeftSide_nodes = np.where(np.abs(X_coord) < 100 * eps)[0] + 1
    RightSide_nodes = np.where(np.abs(X_coord - B) < 100 * eps)[0] + 1
    BottomSide_nodes = np.where(np.abs(Y_coord) < 100 * eps)[0] + 1
    TopSide_nodes = np.where(np.abs(Y_coord - H) < 100 * eps)[0] + 1

    return (Edof, Coord, Ex, Ey,
            LeftSide_nodes, TopSide_nodes,
            RightSide_nodes, BottomSide_nodes)

#Edof, Coord, Ex, Ey,\
#LeftSide_nodes, TopSide_nodes,\
#RightSide_nodes, BottomSide_nodes = TunnelMeshGen(45, 15, 6, 6, 7, 2, 5, 10, 1)
