import numpy as np
from sympy import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define symbols
F, q0, a, b, E, nu, h, r, A1, A2, A3, A4 = symbols('F q0 a b E nu h r A1 A2 A3 A4', real = True)

D = 1/12/(1-nu**2)*E*(h**3) # bending stiffness

q = q0*(r-a)/(b-a)          # distributed load

# Formulate general solutions
w = integrate(1/r*integrate(r*integrate( 1/r*integrate(q*r/D,r),r),r),r)+\
    A1*r**2*log(r/b)+A2*r**2+A3*log(r/b)+A4 # deflection field

wprime = diff(w,r)                          # rotation field

Mr   = D*(-diff(wprime,r) - nu/r*wprime )   # radial bending moment field
Mphi = D*(-1/r*wprime - nu*diff(wprime,r))  # circumferential bending moment field
V    = diff(Mr,r) + 1/r*(Mr-Mphi)           # shear force field

# Apply the boundary conditions
eqns = [
Mr.subs(r,a),     # inner boundary radial bending moment free
V.subs(r,a)-F,    # inner boundary shear force applied
w.subs(r,b),      # outer boundary deflection fixed
wprime.subs(r,b), # outer boundary rotation fixed
]

# Solve for unknown constants
unknowns = (A1,A2,A3,A4)
sol= solve(eqns,unknowns)

# Formulate the deflection field
w_ = simplify(w.subs(sol)) # constants substituted

print("w(r) = ", w_)

# Plot the deflection field for a given set of parameters
wp_ = simplify(w_.subs({F:1., q0:0., E:200e3, nu:0.3, a:100, b:500, h:4})) # parameters substituted

r_  = np.linspace(100., 500., 401)
wr_ = [wp_.subs({r:val}) for val in r_]

plt.figure()
plt.grid()
plt.plot(r_, wr_, "b-")
plt.xlabel(r"$r$ [mm]")
plt.ylabel(r"$w$ [mm]")

plt.show()

# ChatGPT generated code for 360 degree revolution of the solution
# Create mesh in polar coordinates
theta = np.linspace(0, 2*np.pi, 200)
r_3d = np.linspace(100., 500., 200)
R, Theta = np.meshgrid(r_3d, theta)

# Convert to Cartesian for plotting
X = R * np.cos(Theta)
Y = R * np.sin(Theta)

# Convert symbolic expression to a fast numpy function
w_func = lambdify(r, wp_, "numpy")

# === scale factor for deflection ===
scale = 50.0   # <<-- adjust this to exaggerate or reduce displacement
W = scale * w_func(R)

# Create figure
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot surface
surf = ax.plot_surface(X, Y, W, cmap='viridis', linewidth=0, antialiased=True)

# Labels
ax.set_xlabel("x [mm]")
ax.set_ylabel("y [mm]")
ax.set_zlabel(f"w × {scale} [mm]")
ax.set_title(f"Deflection field w(r, θ) × {scale}")

# Colorbar
fig.colorbar(surf, shrink=0.5, aspect=10, label=f"w × {scale} [mm]")

# === Make all axes equal ===
def set_axes_equal(ax):
    """Make axes of 3D plot have equal scale."""
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])

    max_range = max([x_range, y_range, z_range]) / 2.0

    mid_x = np.mean(x_limits)
    mid_y = np.mean(y_limits)
    mid_z = np.mean(z_limits)

    ax.set_xlim3d([mid_x - max_range, mid_x + max_range])
    ax.set_ylim3d([mid_y - max_range, mid_y + max_range])
    ax.set_zlim3d([mid_z - max_range, mid_z + max_range])

set_axes_equal(ax)

plt.show()