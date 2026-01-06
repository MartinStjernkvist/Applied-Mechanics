"""
Problem 2 
Anonumous code: 
"""

# General packages
import numpy as np
import scipy.io
import matplotlib.pyplot as plt

# CALFEM packages
import calfem.core as cfc
import calfem.vis_mpl as cfv

# Load mesh data
mesh = scipy.io.loadmat('mesh_task2.mat') 

Coord = mesh['Coord']                         # [x, y] coords for each node
Dofs = mesh['Dofs']                           
Edof = mesh['Edof']                           # [element number, dof1, dof2, dof3]
Ex = mesh['Ex']                                
Ey = mesh['Ey']                               
right_dofs = mesh['right_dofs']                                                   
left_dofs = mesh['left_dofs']                          

# Plot the mesh
plotpar = np.array([1, 1, 2]) # parameters for line style, color, marker 
# cfv.eldraw2(Ex, Ey, plotpar)
cfv.eldraw2(Ex, Ey)
plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.show()


#-------------------------------
# Start your implementation here
