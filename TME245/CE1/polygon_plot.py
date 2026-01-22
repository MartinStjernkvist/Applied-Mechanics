# -*- coding: utf-8 -*-
"""
Created on Wed Jan 14 16:15:43 2026

@author: fagmar
"""

import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import matplotlib.tri as tri
import numpy as np

# Example polygons (list of Nx2 arrays)
polygons = [
    np.array([[0,0], [1,0], [1,1]]),
    np.array([[1,0], [2,0], [2,1]]),
    np.array([[0,1], [1,1], [1,2]])
]

# Example. Plot polygons without colouring (e.g. undeformed and deformed mesh)
fig1, ax1 = plt.subplots()

pc1 = PolyCollection(
    polygons,
    facecolors='none',
    edgecolors='k'
)

ax1.add_collection(pc1)
ax1.autoscale()
ax1.set_title("Undeformed mesh")
fig1.colorbar(pc1, ax=ax1)


# Example. Plot polygons with colouring by values (e.g. stress plot)

# One value per polygon
stress_values = np.array([1, 2, 3])

fig2, ax2 = plt.subplots()

pc2 = PolyCollection(
    polygons,
    array=stress_values, # values used for coloring 
    cmap='turbo', 
    edgecolors='k')

ax2.add_collection(pc2)
ax2.autoscale()
ax2.set_title("Max tensile stress")
fig2.colorbar(pc2, ax=ax2)

# Example. Plot polygons with interpolated colouring by vertext values
# (e.g. nodal displacements)

nodal_xcoords = np.array([0,1,1,0])
nodal_ycoords = np.array([0,0,1,1])

triangle_nodes = [[0,1,2],[0,2,3]]

nodal_values = np.array([0, 1, 2, 3])

triang = tri.Triangulation(nodal_xcoords, nodal_ycoords, triangle_nodes)

fig3, ax3 = plt.subplots()
plt.tripcolor(triang, nodal_values, shading='gouraud', cmap='jet')
plt.triplot(triang, color='k', linewidth=1.0)
plt.colorbar()
plt.show()
